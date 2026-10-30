#ifdef _Py_TIER2

/*
 * This file contains the support code for CPython's uops optimizer.
 * It also performs some simple optimizations.
 * It performs a traditional data-flow analysis[1] over the trace of uops.
 * Using the information gained, it chooses to emit, or skip certain instructions
 * if possible.
 *
 * [1] For information on data-flow analysis, please see
 * https://clang.llvm.org/docs/DataFlowAnalysisIntro.html
 *
 * */
#include "Python.h"
#include "opcode.h"
#include "pycore_call.h"          // _Py_METH_CALL_FLAGS
#include "pycore_dict.h"
#include "pycore_interp.h"
#include "pycore_opcode_metadata.h"
#include "pycore_opcode_utils.h"
#include "pycore_pystate.h"       // _PyInterpreterState_GET()
#include "pycore_pyatomic_ft_wrappers.h" // FT_ATOMIC_*
#include "pycore_tstate.h"        // _PyThreadStateImpl
#include "pycore_uop_metadata.h"
#include "pycore_long.h"
#include "pycore_interpframe.h"  // _PyFrame_GetCode
#include "pycore_optimizer.h"
#include "pycore_object.h"
#include "pycore_function.h"
#include "pycore_uop_ids.h"
#include "pycore_range.h"
#include "pycore_unicodeobject.h"
#include "pycore_ceval.h"
#include "pycore_floatobject.h"
#include "pycore_setobject.h"
#include "pycore_typeobject.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef Py_DEBUG
    extern const char *_PyUOpName(int index);
    extern void _PyUOpPrint(const _PyUOpInstruction *uop);
    extern void _PyUOpSymPrint(JitOptRef ref);
    static const char *const DEBUG_ENV = "PYTHON_OPT_DEBUG";
    static inline int get_lltrace(void) {
        char *uop_debug = Py_GETENV(DEBUG_ENV);
        int lltrace = 0;
        if (uop_debug != NULL && *uop_debug >= '0') {
            lltrace = *uop_debug - '0';  // TODO: Parse an int and all that
        }
        return lltrace;
    }
    #define DPRINTF(level, ...) \
    if (get_lltrace() >= (level)) { printf(__VA_ARGS__); }

static void
dump_abstract_stack(_Py_UOpsAbstractFrame *frame, JitOptRef *stack_pointer)
{
    printf("    locals=[");
    for (int i = 0 ; i < frame->locals_len; i++) {
        if (i > 0) {
            printf(", ");
        }
        _PyUOpSymPrint(frame->locals[i]);
    }
    printf("]\n");
    if (stack_pointer < frame->stack) {
        printf("    stack=%d\n", (int)(stack_pointer - frame->stack));
    }
    else {
        printf("    stack=[");
        for (JitOptRef *ptr = frame->stack; ptr < stack_pointer; ptr++) {
            if (ptr != frame->stack) {
                printf(", ");
            }
            _PyUOpSymPrint(*ptr);
        }
        printf("]\n");
    }
    fflush(stdout);
}

static void
dump_uop(JitOptContext *ctx, const char *label, int index,
              const _PyUOpInstruction *instr, JitOptRef *stack_pointer)
{
    if (get_lltrace() >= 3) {
        printf("%4d %s: ", index, label);
        _PyUOpPrint(instr);
        printf("\n");
        if (get_lltrace() >= 5 && ctx->frame->code != ((PyCodeObject *)&_Py_InitCleanup)) {
            dump_abstract_stack(ctx->frame, stack_pointer);
        }
    }
}

static void
dump_uops(JitOptContext *ctx, const char *label,
          _PyUOpInstruction *start, JitOptRef *stack_pointer)
{
    int current_len = uop_buffer_length(&ctx->out_buffer);
    int added_count = (int)(ctx->out_buffer.next - start);
    for (int j = 0; j < added_count; j++) {
        dump_uop(ctx, label, current_len - added_count + j, &start[j], stack_pointer);
    }
}

#define DUMP_UOP dump_uop
#define DUMP_UOPS dump_uops

#else
    #define DPRINTF(level, ...)
    #define DUMP_UOP(ctx, label, index, instr, stack_pointer)
    #define DUMP_UOPS(ctx, label, start, stack_pointer)
#endif

static int
get_mutations(PyObject* dict) {
    assert(PyDict_CheckExact(dict));
    PyDictObject *d = (PyDictObject *)dict;
    uint64_t tag = FT_ATOMIC_LOAD_UINT64_RELAXED(d->_ma_watcher_tag);
    return (tag >> DICT_MAX_WATCHERS) & ((1 << DICT_WATCHED_MUTATION_BITS) - 1);
}

static void
increment_mutations(PyObject* dict) {
    assert(PyDict_CheckExact(dict));
    PyDictObject *d = (PyDictObject *)dict;
    FT_ATOMIC_ADD_UINT64(d->_ma_watcher_tag, 1ULL << DICT_MAX_WATCHERS);
}

/* The first two dict watcher IDs are reserved for CPython,
 * so we don't need to check that they haven't been used */
#define BUILTINS_WATCHER_ID 0
#define GLOBALS_WATCHER_ID  1
#define TYPE_WATCHER_ID  0

static int
globals_watcher_callback(PyDict_WatchEvent event, PyObject* dict,
                         PyObject* key, PyObject* new_value)
{
    RARE_EVENT_STAT_INC(watched_globals_modification);
    assert(get_mutations(dict) < _Py_MAX_ALLOWED_GLOBALS_MODIFICATIONS);
    _Py_Executors_InvalidateDependency(_PyInterpreterState_GET(), dict, 1);
    increment_mutations(dict);
    PyDict_Unwatch(GLOBALS_WATCHER_ID, dict);
    return 0;
}

static int
type_watcher_callback(PyTypeObject* type)
{
    _Py_Executors_InvalidateDependency(_PyInterpreterState_GET(), type, 1);
    PyType_Unwatch(TYPE_WATCHER_ID, (PyObject *)type);
    return 0;
}

static int
_setup_optimizer_watchers(void *Py_UNUSED(arg))
{
    PyInterpreterState *interp = _PyInterpreterState_GET();
    FT_ATOMIC_STORE_PTR_RELEASE(
        interp->dict_state.watchers[GLOBALS_WATCHER_ID],
        globals_watcher_callback);
    interp->type_watchers[TYPE_WATCHER_ID] = type_watcher_callback;
    return 0;
}

static void
watch_type(PyTypeObject *type, _PyBloomFilter *filter)
{
    if (_Py_IsImmortal(type) && (type->tp_flags & Py_TPFLAGS_IMMUTABLETYPE)) {
        return;
    }
    PyType_Watch(TYPE_WATCHER_ID, (PyObject *)type);
    _Py_BloomFilter_Add(filter, type);
}

static PyObject *
convert_global_to_const(_PyUOpInstruction *inst, PyObject *obj)
{
    assert(inst->opcode == _LOAD_GLOBAL_MODULE || inst->opcode == _LOAD_GLOBAL_BUILTINS || inst->opcode == _LOAD_ATTR_MODULE);
    assert(PyDict_CheckExact(obj));
    PyDictObject *dict = (PyDictObject *)obj;
    assert(dict->ma_keys->dk_kind == DICT_KEYS_UNICODE);
    PyDictUnicodeEntry *entries = DK_UNICODE_ENTRIES(dict->ma_keys);
    int64_t index = inst->operand1;
    assert(index <= UINT16_MAX);
    if ((int)index >= dict->ma_keys->dk_nentries) {
        return NULL;
    }
    PyDictKeysObject *keys = dict->ma_keys;
    if (keys->dk_version != inst->operand0) {
        return NULL;
    }
    PyObject *res = entries[index].me_value;
    if (res == NULL) {
        return NULL;
    }
    if (_Py_IsImmortal(res)) {
        inst->opcode = _LOAD_CONST_INLINE_BORROW;
    } else {
#ifdef Py_GIL_DISABLED
        PyUnstable_Object_EnableDeferredRefcount(res);
        if (!_PyObject_HasDeferredRefcount(res)) {
            return NULL;
        }
#endif
        inst->opcode = _LOAD_CONST_INLINE;
    }
    inst->operand0 = (uint64_t)res;
    return res;
}

static bool
incorrect_keys(PyObject *obj, uint32_t version)
{
    if (!PyDict_CheckExact(obj)) {
        return true;
    }
    PyDictObject *dict = (PyDictObject *)obj;
    return dict->ma_keys->dk_version != version;
}


#define STACK_LEVEL()     ((int)(stack_pointer - ctx->frame->stack))
#define STACK_SIZE()      ((int)(ctx->frame->stack_len))

static inline int
is_terminator_uop(const _PyUOpInstruction *uop)
{
    int opcode = uop->opcode;
    return (
        opcode == _EXIT_TRACE ||
        opcode == _JUMP_TO_TOP ||
        opcode == _DYNAMIC_EXIT ||
        opcode == _DEOPT
    );
}

#define CURRENT_FRAME_IS_INIT_SHIM() (ctx->frame->code == ((PyCodeObject *)&_Py_InitCleanup))

#define GETLOCAL(idx)          ((ctx->frame->locals[idx]))

#define REPLACE_OP(INST, OP, ARG, OPERAND)    \
    (INST)->opcode = OP;            \
    (INST)->oparg = ARG;            \
    (INST)->operand0 = OPERAND;

#define ADD_OP(OP, ARG, OPERAND) add_op(ctx, this_instr, (OP), (ARG), (OPERAND))

static inline void
add_op(JitOptContext *ctx, _PyUOpInstruction *this_instr,
       uint16_t opcode, uint16_t oparg, uintptr_t operand0)
{
    _PyUOpInstruction *out = ctx->out_buffer.next;
    assert(out < ctx->out_buffer.end);
    out->opcode = (opcode);
    out->format = this_instr->format;
    out->oparg = (oparg);
    out->target = this_instr->target;
    out->operand0 = (operand0);
    out->operand1 = this_instr->operand1;
#ifdef Py_STATS
    out->fitness = this_instr->fitness;
#endif
    ctx->out_buffer.next++;
}

/* Shortened forms for convenience, used in optimizer_bytecodes.c */
#define sym_is_not_null _Py_uop_sym_is_not_null
#define sym_is_const _Py_uop_sym_is_const
#define sym_is_safe_const _Py_uop_sym_is_safe_const
#define sym_is_not_container _Py_uop_sym_is_not_container
#define sym_get_const _Py_uop_sym_get_const
#define sym_new_const_steal _Py_uop_sym_new_const_steal
#define sym_get_const_as_stackref _Py_uop_sym_get_const_as_stackref
#define sym_new_unknown _Py_uop_sym_new_unknown
#define sym_new_not_null _Py_uop_sym_new_not_null
#define sym_new_type _Py_uop_sym_new_type
#define sym_is_null _Py_uop_sym_is_null
#define sym_new_const _Py_uop_sym_new_const
#define sym_new_null _Py_uop_sym_new_null
#define sym_has_type _Py_uop_sym_has_type
#define sym_get_type _Py_uop_sym_get_type
#define sym_get_probable_type _Py_uop_sym_get_probable_type
#define sym_matches_type _Py_uop_sym_matches_type
#define sym_matches_type_version _Py_uop_sym_matches_type_version
#define sym_get_type_version _Py_uop_sym_get_type_version
#define sym_set_unshared_local _Py_uop_sym_set_unshared_local
#define sym_is_unshared_local _Py_uop_sym_is_unshared_local
#define sym_set_null(SYM) _Py_uop_sym_set_null(ctx, SYM)
#define sym_set_non_null(SYM) _Py_uop_sym_set_non_null(ctx, SYM)
#define sym_set_type(SYM, TYPE) _Py_uop_sym_set_type(ctx, SYM, TYPE)
#define sym_set_type_version(SYM, VERSION) _Py_uop_sym_set_type_version(ctx, SYM, VERSION)
#define sym_set_const(SYM, CNST) _Py_uop_sym_set_const(ctx, SYM, CNST)
#define sym_set_compact_int(SYM) _Py_uop_sym_set_compact_int(ctx, SYM)
#define sym_is_bottom _Py_uop_sym_is_bottom
#define sym_truthiness _Py_uop_sym_truthiness
#define frame_new _Py_uop_frame_new
#define frame_new_from_symbol _Py_uop_frame_new_from_symbol
#define frame_pop _Py_uop_frame_pop
#define sym_new_tuple _Py_uop_sym_new_tuple
#define sym_tuple_getitem _Py_uop_sym_tuple_getitem
#define sym_tuple_length _Py_uop_sym_tuple_length
#define sym_is_immortal _Py_uop_symbol_is_immortal
#define sym_is_compact_int _Py_uop_sym_is_compact_int
#define sym_new_compact_int _Py_uop_sym_new_compact_int
#define sym_new_truthiness _Py_uop_sym_new_truthiness
#define sym_new_predicate _Py_uop_sym_new_predicate
#define sym_apply_predicate_narrowing _Py_uop_sym_apply_predicate_narrowing
#define sym_set_recorded_type(SYM, TYPE) _Py_uop_sym_set_recorded_type(ctx, SYM, TYPE)
#define sym_set_recorded_value(SYM, VAL) _Py_uop_sym_set_recorded_value(ctx, SYM, VAL)
#define sym_set_recorded_gen_func(SYM, VAL) _Py_uop_sym_set_recorded_gen_func(ctx, SYM, VAL)
#define sym_get_probable_func_code _Py_uop_sym_get_probable_func_code
#define sym_get_probable_value _Py_uop_sym_get_probable_value
#define sym_set_stack_depth(DEPTH, SP) _Py_uop_sym_set_stack_depth(ctx, DEPTH, SP)

/* Comparison oparg masks */
#define COMPARE_LT_MASK 2
#define COMPARE_GT_MASK 4
#define COMPARE_EQ_MASK 8

#define JUMP_TO_LABEL(label) goto label;

static int
check_stack_bounds(JitOptContext *ctx, JitOptRef *stack_pointer, int offset, int opcode)
{
    int stack_level = (int)(stack_pointer + (offset) - ctx->frame->stack);
    int should_check = !CURRENT_FRAME_IS_INIT_SHIM() ||
        (opcode == _RETURN_VALUE) ||
        (opcode == _RETURN_GENERATOR) ||
        (opcode == _YIELD_VALUE);
    if (should_check && (stack_level < 0 || stack_level > STACK_SIZE() + MAX_CACHED_REGISTER)) {
        ctx->contradiction = true;
        ctx->done = true;
        return 1;
    }
    return 0;
}

#define CHECK_STACK_BOUNDS(offset) \
    if (check_stack_bounds(ctx, stack_pointer, offset, opcode)) { \
        break; \
    } \

static int
optimize_to_bool(
    _PyUOpInstruction *this_instr,
    JitOptContext *ctx,
    JitOptRef value,
    JitOptRef *result_ptr,
    uint16_t prefix, uint16_t suffix)
{
    if (sym_matches_type(value, &PyBool_Type)) {
        ADD_OP(_NOP, 0, 0);
        *result_ptr = value;
        return 1;
    }
    int truthiness = sym_truthiness(ctx, value);
    if (truthiness >= 0) {
        PyObject *load = truthiness ? Py_True : Py_False;
        if (prefix != _NOP) {
            ADD_OP(prefix, 0, 0);
        }
        ADD_OP(_LOAD_CONST_INLINE_BORROW, 0, (uintptr_t)load);
        if (suffix != _NOP) {
            ADD_OP(suffix, 2, 0);
        }
        *result_ptr = sym_new_const(ctx, load);
        return 1;
    }
    return 0;
}

static void
optimize_dict_known_hash(
    JitOptContext *ctx, _PyBloomFilter *dependencies, _PyUOpInstruction *this_instr,
    PyObject *sub, uint16_t opcode)
{
    if (PyUnicode_CheckExact(sub) || PyLong_CheckExact(sub) || PyBytes_CheckExact(sub)
            || PyFloat_CheckExact(sub) || PyComplex_CheckExact(sub)) {
        // PyObject_Hash can't fail on these types
        ADD_OP(opcode, 0, PyObject_Hash(sub));
    }
    else if (PyTuple_CheckExact(sub)) {
        // only use known hash variant when hash of tuple is already computed
        // since computing it can call arbitrary code
        Py_hash_t hash = ((PyTupleObject *)sub)->ob_hash;
        if (hash != -1) {
            ADD_OP(opcode, 0, hash);
        }
    }
    else if (Py_TYPE(sub)->tp_hash == PyBaseObject_Type.tp_hash) {
        // for user-defined objects which don't override tp_hash
        Py_hash_t hash = PyObject_Hash(sub);
        ADD_OP(opcode, 0, hash);
        watch_type(Py_TYPE(sub), dependencies);
    }
}

static void
eliminate_pop_guard(_PyUOpInstruction *this_instr, JitOptContext *ctx, bool exit)
{
    ADD_OP(_POP_TOP, 0, 0);
    if (exit) {
        REPLACE_OP((this_instr+1), _EXIT_TRACE, 0, 0);
        this_instr[1].target = this_instr->target;
    }
}

static JitOptRef
lookup_attr(JitOptContext *ctx, _PyBloomFilter *dependencies, _PyUOpInstruction *this_instr,
            PyTypeObject *type, PyObject *name,
            uint16_t prefix, uint16_t suffix)
{
    // The cached value may be dead, so we need to do the lookup again... :(
    if (type && PyType_Check(type)) {
        PyObject *lookup = _PyType_LookupRef(type, name);
        if (lookup) {
            bool immortal = _Py_IsImmortal(lookup) || (type->tp_flags & Py_TPFLAGS_IMMUTABLETYPE);
#ifdef Py_GIL_DISABLED
            if (!immortal) {
                PyUnstable_Object_EnableDeferredRefcount(lookup);
                if (!_PyObject_HasDeferredRefcount(lookup)) {
                    Py_DECREF(lookup);
                    return sym_new_not_null(ctx);
                }
            }
#endif
            if (prefix != _NOP) {
                ADD_OP(prefix, 0, 0);
            }
            ADD_OP(immortal ? _LOAD_CONST_INLINE_BORROW : _LOAD_CONST_INLINE,
                   0, (uintptr_t)lookup);
            if (suffix != _NOP) {
                ADD_OP(suffix, 2, 0);
            }
            if ((type->tp_flags & Py_TPFLAGS_IMMUTABLETYPE) == 0) {
                watch_type(type, dependencies);
            }
            JitOptRef res = sym_new_const(ctx, lookup);
            if (immortal) {
                res = PyJitRef_Borrow(res);
            }
            Py_DECREF(lookup);
            return res;
        }
    }
    return sym_new_not_null(ctx);
}

static void
optimize_pop_top(JitOptContext *ctx, _PyUOpInstruction *this_instr, JitOptRef value)
{
    PyTypeObject *typ = sym_get_type(value);
    if (PyJitRef_IsBorrowed(value) ||
        sym_is_immortal(PyJitRef_Unwrap(value)) ||
        sym_is_null(value)) {
        ADD_OP(_POP_TOP_NOP, 0, 0);
    }
    else if (typ == &PyLong_Type) {
        ADD_OP(_POP_TOP_INT, 0, 0);
    }
    else if (typ == &PyFloat_Type) {
        ADD_OP(_POP_TOP_FLOAT, 0, 0);
    }
    else if (typ == &PyUnicode_Type) {
        ADD_OP(_POP_TOP_UNICODE, 0, 0);
    }
    else {
        ADD_OP(_POP_TOP, 0, 0);
    }
}

/* Look up name via super (normal case from supercheck where
   su_obj_type = Py_TYPE(obj)). */
static JitOptRef
lookup_super_attr(JitOptContext *ctx, _PyBloomFilter *dependencies,
                  _PyUOpInstruction *this_instr,
                  PyTypeObject *su_type, PyTypeObject *obj_type,
                  PyObject *name,
                  uint16_t immortal, uint16_t mortal, uint16_t suffix)
{
    if (su_type == NULL || obj_type == NULL) {
        return sym_new_not_null(ctx);
    }
    /* Normal case: obj_type must be a subtype of su_type */
    if (!PyType_IsSubtype(obj_type, su_type)) {
        return sym_new_not_null(ctx);
    }
    PyObject *lookup = _PySuper_LookupDescr(su_type, obj_type, name);
    if (lookup == NULL) {
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }
        return sym_new_not_null(ctx);
    }
    if ((Py_TYPE(lookup)->tp_flags & Py_TPFLAGS_METHOD_DESCRIPTOR) == 0) {
        Py_DECREF(lookup);
        return sym_new_not_null(ctx);
    }
    int opcode = mortal;
    if (_Py_IsImmortal(lookup) || (obj_type->tp_flags & Py_TPFLAGS_IMMUTABLETYPE)) {
        opcode = immortal;
    }
#ifdef Py_GIL_DISABLED
    else {
        PyUnstable_Object_EnableDeferredRefcount(lookup);
        if (!_PyObject_HasDeferredRefcount(lookup)) {
            Py_DECREF(lookup);
            return sym_new_not_null(ctx);
        }
    }
#endif
    ADD_OP(_SWAP, 3, 0);
    ADD_OP(_POP_TOP, 0, 0);
    ADD_OP(_POP_TOP, 0, 0);
    ADD_OP(opcode, 0, (uintptr_t)lookup);
    if (suffix != _NOP) {
        ADD_OP(suffix, 2, 0);
    }
    // if obj_type is immutable, then all its superclasses are immutable
    if ((obj_type->tp_flags & Py_TPFLAGS_IMMUTABLETYPE) == 0) {
        watch_type(su_type, dependencies);
        watch_type(obj_type, dependencies);
    }
    JitOptRef res = sym_new_const_steal(ctx, lookup);
    if (opcode == immortal) {
        res = PyJitRef_Borrow(res);
    }
    return res;
}

static
PyCodeObject *
get_current_code_object(JitOptContext *ctx)
{
    return (PyCodeObject *)ctx->frame->code;
}

static PyObject *
get_co_name(JitOptContext *ctx, int index)
{
    return PyTuple_GET_ITEM(get_current_code_object(ctx)->co_names, index);
}

static int
get_test_bit_for_bools(void) {
#ifdef Py_STACKREF_DEBUG
    uintptr_t false_bits = _Py_STACKREF_FALSE_INDEX;
    uintptr_t true_bits = _Py_STACKREF_TRUE_INDEX;
#else
    uintptr_t false_bits = (uintptr_t)&_Py_FalseStruct;
    uintptr_t true_bits = (uintptr_t)&_Py_TrueStruct;
#endif
    for (int i = 4; i < 8; i++) {
        if ((true_bits ^ false_bits) & (uintptr_t)(1 << i)) {
            return i;
        }
    }
    return 0;
}

static int
test_bit_set_in_true(int bit) {
#ifdef Py_STACKREF_DEBUG
    uintptr_t true_bits = _Py_STACKREF_TRUE_INDEX;
#else
    uintptr_t true_bits = (uintptr_t)&_Py_TrueStruct;
#endif
    assert((true_bits ^ ((uintptr_t)&_Py_FalseStruct)) & (uintptr_t)(1 << bit));
    return true_bits & (uintptr_t)(1 << bit);
}

#ifdef Py_DEBUG
void
_Py_opt_assert_within_stack_bounds(
    _Py_UOpsAbstractFrame *frame, JitOptRef *stack_pointer,
    const char *filename, int lineno
) {
    if (frame->code == ((PyCodeObject *)&_Py_InitCleanup)) {
        return;
    }
    int level = (int)(stack_pointer - frame->stack);
    if (level < 0) {
        printf("Stack underflow (depth = %d) at %s:%d\n", level, filename, lineno);
        fflush(stdout);
        abort();
    }
    int size = (int)(frame->stack_len) + MAX_CACHED_REGISTER;
    if (level > size) {
        printf("Stack overflow (depth = %d) at %s:%d\n", level, filename, lineno);
        fflush(stdout);
        abort();
    }
}
#endif

#ifdef Py_DEBUG
#define ASSERT_WITHIN_STACK_BOUNDS(F, L) _Py_opt_assert_within_stack_bounds(ctx->frame, stack_pointer, (F), (L))
#else
#define ASSERT_WITHIN_STACK_BOUNDS(F, L) (void)0
#endif

/* >0 (length) for success, 0 for not ready, clears all possible errors. */
static int
optimize_uops(
    _PyThreadStateImpl *tstate,
    _PyUOpInstruction *trace,
    int trace_len,
    int curr_stacklen,
    _PyUOpInstruction *output,
    _PyBloomFilter *dependencies
)
{
    assert(!PyErr_Occurred());
    assert(tstate->jit_tracer_state != NULL);
    PyFunctionObject *func = tstate->jit_tracer_state->initial_state.func;

    JitOptContext *ctx = &tstate->jit_tracer_state->opt_context;
    uint32_t opcode = UINT16_MAX;

    uop_buffer_init(&ctx->out_buffer, output, UOP_MAX_TRACE_LENGTH);

    // Make sure that watchers are set up
    PyInterpreterState *interp = _PyInterpreterState_GET();
    _PyOnceFlag_CallOnce(&interp->dict_state.watcher_setup_once,
                         _setup_optimizer_watchers, NULL);

    _Py_uop_abstractcontext_init(ctx, dependencies);
    _Py_UOpsAbstractFrame *frame = _Py_uop_frame_new(ctx, (PyCodeObject *)func->func_code, NULL, 0);
    if (frame == NULL) {
        return 0;
    }
    frame->func = func;
    ctx->curr_frame_depth++;
    ctx->frame = frame;
    _Py_uop_sym_set_stack_depth(ctx, curr_stacklen, frame->stack_pointer);

    _PyUOpInstruction *this_instr = NULL;
    JitOptRef *stack_pointer = ctx->frame->stack_pointer;

    for (int i = 0; i < trace_len; i++) {
        this_instr = &trace[i];
        if (ctx->done) {
            // Don't do any more optimization, but
            // we still need to reach a terminator for corrctness.
            *(ctx->out_buffer.next++) = *this_instr;
            if (is_terminator_uop(this_instr)) {
                break;
            }
            continue;
        }

        int oparg = this_instr->oparg;
        opcode = this_instr->opcode;

        if (!CURRENT_FRAME_IS_INIT_SHIM()) {
            stack_pointer = ctx->frame->stack_pointer;
        }

        DUMP_UOP(ctx, "abs", (int)(this_instr - trace), this_instr, stack_pointer);

        _PyUOpInstruction *out_ptr = ctx->out_buffer.next;

        switch (opcode) {

#include "optimizer_cases.c.h"

            default:
                DPRINTF(1, "\nUnknown opcode in abstract interpreter\n");
                Py_UNREACHABLE();
        }
        // If no ADD_OP was called during this iteration, copy the original instruction
        if (ctx->out_buffer.next == out_ptr) {
            *(ctx->out_buffer.next++) = *this_instr;
        }
        assert(ctx->frame != NULL);
        DUMP_UOPS(ctx, "out", out_ptr, stack_pointer);
        if (!CURRENT_FRAME_IS_INIT_SHIM() && !ctx->done) {
            DPRINTF(3, " stack_level %d\n", STACK_LEVEL());
            ctx->frame->stack_pointer = stack_pointer;
            assert(STACK_LEVEL() >= 0);
        }
    }
    if (ctx->out_of_space) {
        DPRINTF(3, "\n");
        DPRINTF(1, "Out of space in abstract interpreter\n");
    }
    if (ctx->contradiction) {
        // Attempted to push a "bottom" (contradiction) symbol onto the stack.
        // This means that the abstract interpreter has optimized to trace
        // to an unreachable estate.
        // We *could* generate an _EXIT_TRACE or _FATAL_ERROR here, but hitting
        // bottom usually indicates an optimizer bug, so we are probably better off
        // retrying later.
        DPRINTF(3, "\n");
        DPRINTF(1, "Hit bottom in abstract interpreter\n");
        _Py_uop_abstractcontext_fini(ctx);
        OPT_STAT_INC(optimizer_contradiction);
        return 0;
    }

    /* Either reached the end or cannot optimize further, but there
     * would be no benefit in retrying later */
    _Py_uop_abstractcontext_fini(ctx);
    // Check that the trace ends with a proper terminator
    if (uop_buffer_length(&ctx->out_buffer) > 0) {
        assert(is_terminator_uop(uop_buffer_last(&ctx->out_buffer)));
    }

    return uop_buffer_length(&ctx->out_buffer);

error:
    DPRINTF(3, "\n");
    DPRINTF(1, "Encountered error in abstract interpreter\n");
    if (opcode <= MAX_UOP_ID) {
        OPT_ERROR_IN_OPCODE(opcode);
    }
    _Py_uop_abstractcontext_fini(ctx);

    assert(PyErr_Occurred());
    PyErr_Clear();

    return 0;

}

const uint16_t op_without_push[MAX_UOP_ID + 1] = {
    [_COPY] = _NOP,
    [_LOAD_CONST_INLINE] = _NOP,
    [_LOAD_CONST_INLINE_BORROW] = _NOP,
    [_LOAD_FAST] = _NOP,
    [_LOAD_FAST_BORROW] = _NOP,
    [_LOAD_SMALL_INT] = _NOP,
    [_PUSH_NULL] = _NOP,
};

const bool op_skip[MAX_UOP_ID + 1] = {
    [_NOP] = true,
    [_SET_IP] = true,
};

const uint16_t op_without_pop[MAX_UOP_ID + 1] = {
    [_POP_TOP] = _NOP,
    [_POP_TOP_NOP] = _NOP,
    [_POP_TOP_INT] = _NOP,
    [_POP_TOP_FLOAT] = _NOP,
    [_POP_TOP_UNICODE] = _NOP,
};


static int
remove_unneeded_uops(_PyUOpInstruction *buffer, int buffer_size)
{
    /* Remove _SET_IP and _CHECK_VALIDITY where possible.
     * _SET_IP is needed if the following instruction escapes or
     * could error. _CHECK_VALIDITY is needed if the previous
     * instruction could have escaped. */
    int last_set_ip = -1;
    bool may_have_escaped = true;
    int call_depth = 0;
    int max_checked_recursion_depth = -1;
    for (int pc = 0; pc < buffer_size; pc++) {
        int opcode = buffer[pc].opcode;
        switch (opcode) {
            case _START_EXECUTOR:
                may_have_escaped = false;
                call_depth = 0;
                max_checked_recursion_depth = -1;
                break;
            case _MAKE_WARM:
                last_set_ip = -1;
                may_have_escaped = false;
                break;
            case _SET_IP:
                buffer[pc].opcode = _NOP;
                last_set_ip = pc;
                break;
            case _CHECK_VALIDITY:
                if (may_have_escaped) {
                    may_have_escaped = false;
                }
                else {
                    buffer[pc].opcode = _NOP;
                }
                break;
            case _EXIT_TRACE:
            default:
            {
                if (_PyUop_Flags[opcode] & HAS_RECORDS_VALUE_FLAG) {
                    Py_XDECREF((PyObject *)(uintptr_t)buffer[pc].operand0);
                    buffer[pc].opcode = _NOP;
                    break;
                }
                if (opcode == _CHECK_RECURSION_REMAINING) {
                    if (call_depth >= 0 && call_depth <= max_checked_recursion_depth) {
                        buffer[pc].opcode = _NOP;
                        break;
                    }
                    max_checked_recursion_depth = call_depth;
                }
                else if (opcode == _PUSH_FRAME || opcode == _CREATE_INIT_FRAME) {
                    call_depth++;
                }
                else if (opcode == _RETURN_VALUE ||
                         opcode == _RETURN_GENERATOR ||
                         opcode == _YIELD_VALUE)
                {
                    if (call_depth > 0) {
                        call_depth--;
                    }
                    else {
                        max_checked_recursion_depth = -1;
                    }
                }
                // Cancel out pushes and pops, repeatedly. So:
                //     _LOAD_FAST + _POP_TOP + _POP_TOP + _LOAD_CONST_INLINE_BORROW + _POP_TOP
                // ...becomes:
                //     _NOP + _NOP + _POP_TOP + _NOP + _NOP
                while (op_without_pop[opcode]) {
                    _PyUOpInstruction *last = &buffer[pc - 1];
                    while (op_skip[last->opcode]) {
                        last--;
                    }
                    if (op_without_push[last->opcode] && op_without_pop[opcode]) {
                        last->opcode = op_without_push[last->opcode];
                        opcode = buffer[pc].opcode = op_without_pop[opcode];
                        if (op_without_pop[last->opcode]) {
                            opcode = last->opcode;
                            pc = (int)(last - buffer);
                        }
                    }
                    else {
                        break;
                    }
                }
                /* _PUSH_FRAME doesn't escape or error, but it
                 * does need the IP for the return address */
                bool needs_ip = (opcode == _PUSH_FRAME || opcode == _YIELD_VALUE || opcode == _DYNAMIC_EXIT || opcode == _EXIT_TRACE);
                if (_PyUop_Flags[opcode] & HAS_ESCAPES_FLAG) {
                    needs_ip = true;
                    may_have_escaped = true;
                    if (opcode != _CHECK_PERIODIC &&
                        opcode != _CHECK_PERIODIC_IF_NOT_YIELD_FROM &&
                        opcode != _TIER2_RESUME_CHECK &&
                        opcode != _RETURN_VALUE &&
                        opcode != _POP_TOP)
                    {
                        max_checked_recursion_depth = -1;
                    }
                }
                if (needs_ip && last_set_ip >= 0) {
                    assert(buffer[last_set_ip].opcode == _NOP);
                    buffer[last_set_ip].opcode = _SET_IP;
                    last_set_ip = -1;
                }
                if (opcode == _EXIT_TRACE) {
                    return pc + 1;
                }
                break;
            }
            case _JUMP_TO_TOP:
            case _DYNAMIC_EXIT:
            case _DEOPT:
                return pc + 1;
        }
    }
    Py_UNREACHABLE();
}

static int
hoist_loop_preheader_guards(_PyUOpInstruction *output, int length, int curr_stacklen)
{
    (void)curr_stacklen;
    if (length < 4 ||
        output[0].opcode != _START_EXECUTOR ||
        output[1].opcode != _MAKE_WARM ||
        output[length - 1].opcode != _JUMP_TO_TOP)
    {
        return length;
    }

    int first_idx = -1;
    int non_nop_count = 0;
    int type_guard_count = 0;
    int pop_top_count = 0;
    bool has_iter_check = false;
    bool has_compare_op = false;
    bool has_arith_binop = false;
    bool can_peel_body = true;

    bool written[64] = {0};
    int load_count[64] = {0};
    int store_count[64] = {0};

    for (int i = 2; i < length - 1; i++) {
        uint16_t op = output[i].opcode;
        if (op == _NOP) {
            continue;
        }
        if (first_idx < 0) {
            first_idx = i;
        }
        non_nop_count++;
        switch (op) {
            case _ITER_CHECK_RANGE:
            case _ITER_CHECK_LIST:
            case _ITER_CHECK_TUPLE:
                has_iter_check = true;
                break;
            case _COMPARE_OP_INT:
            case _COMPARE_OP_FLOAT:
                has_compare_op = true;
                break;
            case _GUARD_TOS_INT:
            case _GUARD_NOS_INT:
            case _GUARD_TOS_FLOAT:
            case _GUARD_NOS_FLOAT:
            case _GUARD_TOS_LIST:
            case _GUARD_NOS_LIST:
            case _GUARD_TOS_TUPLE:
            case _GUARD_NOS_TUPLE:
            case _GUARD_TOS_DICT:
            case _GUARD_TOS_UNICODE:
            case _GUARD_NOS_UNICODE:
            case _GUARD_NOS_COMPACT_ASCII:
            case _GUARD_TYPE:
            case _GUARD_TYPE_VERSION:
            case _CHECK_MANAGED_OBJECT_HAS_VALUES:
                type_guard_count++;
                break;
            case _GUARD_BINARY_OP_SUBSCR_TUPLE_INT_BOUNDS:
            case _BINARY_OP_SUBSCR_LIST_INT:
            case _BINARY_OP_SUBSCR_TUPLE_INT:
            case _LOAD_ATTR_INSTANCE_VALUE:
            case _LOAD_ATTR_SLOT:
                break;
            case _LOAD_FAST:
            case _LOAD_FAST_BORROW:
                if (output[i].oparg >= 64) {
                    can_peel_body = false;
                }
                else {
                    load_count[output[i].oparg]++;
                }
                break;
            case _LOAD_FAST_0:
            case _LOAD_FAST_1:
            case _LOAD_FAST_2:
            case _LOAD_FAST_3:
            case _LOAD_FAST_4:
            case _LOAD_FAST_5:
            case _LOAD_FAST_6:
            case _LOAD_FAST_7:
                load_count[op - _LOAD_FAST_0]++;
                break;
            case _LOAD_FAST_BORROW_0:
            case _LOAD_FAST_BORROW_1:
            case _LOAD_FAST_BORROW_2:
            case _LOAD_FAST_BORROW_3:
            case _LOAD_FAST_BORROW_4:
            case _LOAD_FAST_BORROW_5:
            case _LOAD_FAST_BORROW_6:
            case _LOAD_FAST_BORROW_7:
                load_count[op - _LOAD_FAST_BORROW_0]++;
                break;
            case _SWAP_FAST:
                if (output[i].oparg >= 64) {
                    can_peel_body = false;
                }
                else {
                    written[output[i].oparg] = true;
                    store_count[output[i].oparg]++;
                }
                break;
            case _SWAP_FAST_0:
            case _SWAP_FAST_1:
            case _SWAP_FAST_2:
            case _SWAP_FAST_3:
            case _SWAP_FAST_4:
            case _SWAP_FAST_5:
            case _SWAP_FAST_6:
            case _SWAP_FAST_7:
                written[op - _SWAP_FAST_0] = true;
                store_count[op - _SWAP_FAST_0]++;
                break;
            case _BINARY_OP_ADD_INT:
            case _BINARY_OP_ADD_INT_INPLACE:
            case _BINARY_OP_ADD_INT_INPLACE_RIGHT:
            case _BINARY_OP_SUBTRACT_INT:
            case _BINARY_OP_SUBTRACT_INT_INPLACE:
            case _BINARY_OP_SUBTRACT_INT_INPLACE_RIGHT:
            case _BINARY_OP_MULTIPLY_INT:
            case _BINARY_OP_MULTIPLY_INT_INPLACE:
            case _BINARY_OP_MULTIPLY_INT_INPLACE_RIGHT:
            case _BINARY_OP_ADD_FLOAT:
            case _BINARY_OP_ADD_FLOAT_INPLACE:
            case _BINARY_OP_ADD_FLOAT_INPLACE_RIGHT:
            case _BINARY_OP_SUBTRACT_FLOAT:
            case _BINARY_OP_SUBTRACT_FLOAT_INPLACE:
            case _BINARY_OP_SUBTRACT_FLOAT_INPLACE_RIGHT:
            case _BINARY_OP_MULTIPLY_FLOAT:
            case _BINARY_OP_MULTIPLY_FLOAT_INPLACE:
            case _BINARY_OP_MULTIPLY_FLOAT_INPLACE_RIGHT:
            case _BINARY_OP_TRUEDIV_FLOAT:
            case _BINARY_OP_TRUEDIV_FLOAT_INPLACE:
            case _BINARY_OP_TRUEDIV_FLOAT_INPLACE_RIGHT:
                has_arith_binop = true;
                break;
            case _SET_IP:
            case _CHECK_PERIODIC:
            case _CHECK_PERIODIC_IF_NOT_YIELD_FROM:
            case _CHECK_VALIDITY:
            case _TIER2_RESUME_CHECK:
            case _GUARD_NOT_EXHAUSTED_RANGE:
            case _GUARD_NOT_EXHAUSTED_LIST:
            case _GUARD_NOT_EXHAUSTED_TUPLE:
            case _ITER_NEXT_RANGE:
            case _ITER_NEXT_LIST:
            case _ITER_NEXT_LIST_TIER_TWO:
            case _ITER_NEXT_TUPLE:
            case _LOAD_SMALL_INT:
            case _LOAD_CONST_INLINE:
            case _LOAD_CONST_INLINE_BORROW:
            case _UNARY_NEGATIVE_FLOAT_INPLACE:
                break;
            case _POP_TOP:
                pop_top_count++;
                break;
            case _POP_TOP_NOP:
            case _POP_TOP_INT:
            case _POP_TOP_FLOAT:
            case _POP_TOP_UNICODE:
            case _GUARD_IS_TRUE_POP:
            case _GUARD_IS_FALSE_POP:
            case _GUARD_BIT_IS_SET_POP:
            case _GUARD_BIT_IS_UNSET_POP:
                break;
            default:
                can_peel_body = false;
                break;
        }
    }

    if (first_idx < 0) {
        return length;
    }

    /* Try Pre-Header guard peeling for arithmetic, container-indexing,
     * list/tuple-iteration, and attribute-reading loops where loop-invariant
     * or inductive local guards can be hoisted to Iteration 1. */
    if (can_peel_body &&
        has_arith_binop &&
        type_guard_count > 0 &&
        pop_top_count <= 2 &&
        non_nop_count <= 32 &&
        !(has_iter_check && has_compare_op) &&
        length + non_nop_count <= UOP_MAX_TRACE_LENGTH)
    {
        _PyUOpInstruction body[32];
        int b_len = 0;
        for (int i = 2; i < length - 1; i++) {
            if (output[i].opcode != _NOP) {
                body[b_len++] = output[i];
            }
        }

        enum {
            VAL_UNKNOWN = 0,
            VAL_INT = 1,
            VAL_FLOAT = 2,
            VAL_ITER_ITEM = 3,
        };
        int stk_type[32];
        int stk_var[32];
        bool stk_entry[32];
        int sp = 0;
        bool seen_write[64] = {0};
        int cur_var_type[64] = {0};
        bool all_stores_int[64];
        bool all_stores_float[64];
        for (int v = 0; v < 64; v++) {
            all_stores_int[v] = true;
            all_stores_float[v] = true;
        }

        /* guard_kind[i]:
         *   0 = not a hoistable guard
         *   1 = iterator check (_ITER_CHECK_{RANGE,LIST,TUPLE})
         *   2 = int guard on entry value of local guard_var[i] (or proven int if -2)
         *   3 = float guard on entry value of local guard_var[i] (or proven float if -2)
         *   4 = invariant local guard (container/type/managed-values) on guard_var[i]
         */
        int guard_kind[32] = {0};
        int guard_var[32];
        for (int i = 0; i < 32; i++) {
            guard_var[i] = -1;
        }
        bool body_valid = true;

        for (int i = 0; i < b_len && body_valid; i++) {
            _PyUOpInstruction *inst = &body[i];
            uint16_t op = inst->opcode;
            switch (op) {
                case _ITER_CHECK_RANGE:
                case _ITER_CHECK_LIST:
                case _ITER_CHECK_TUPLE:
                    guard_kind[i] = 1;
                    break;
                case _SET_IP:
                case _CHECK_PERIODIC:
                case _CHECK_PERIODIC_IF_NOT_YIELD_FROM:
                case _CHECK_VALIDITY:
                case _TIER2_RESUME_CHECK:
                case _GUARD_NOT_EXHAUSTED_RANGE:
                case _GUARD_NOT_EXHAUSTED_LIST:
                case _GUARD_NOT_EXHAUSTED_TUPLE:
                case _GUARD_BINARY_OP_SUBSCR_TUPLE_INT_BOUNDS:
                    break;
                case _ITER_NEXT_RANGE:
                    if (sp >= 32) { body_valid = false; break; }
                    stk_type[sp] = VAL_INT;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    sp++;
                    break;
                case _ITER_NEXT_LIST:
                case _ITER_NEXT_LIST_TIER_TWO:
                case _ITER_NEXT_TUPLE:
                    if (sp >= 32) { body_valid = false; break; }
                    stk_type[sp] = VAL_ITER_ITEM;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    sp++;
                    break;
                case _LOAD_FAST:
                case _LOAD_FAST_BORROW:
                case _LOAD_FAST_0:
                case _LOAD_FAST_1:
                case _LOAD_FAST_2:
                case _LOAD_FAST_3:
                case _LOAD_FAST_4:
                case _LOAD_FAST_5:
                case _LOAD_FAST_6:
                case _LOAD_FAST_7:
                case _LOAD_FAST_BORROW_0:
                case _LOAD_FAST_BORROW_1:
                case _LOAD_FAST_BORROW_2:
                case _LOAD_FAST_BORROW_3:
                case _LOAD_FAST_BORROW_4:
                case _LOAD_FAST_BORROW_5:
                case _LOAD_FAST_BORROW_6:
                case _LOAD_FAST_BORROW_7: {
                    if (sp >= 32) { body_valid = false; break; }
                    int v;
                    if (op == _LOAD_FAST || op == _LOAD_FAST_BORROW) {
                        v = (int)inst->oparg;
                    }
                    else if (op >= _LOAD_FAST_0 && op <= _LOAD_FAST_7) {
                        v = (int)(op - _LOAD_FAST_0);
                    }
                    else {
                        v = (int)(op - _LOAD_FAST_BORROW_0);
                    }
                    stk_type[sp] = cur_var_type[v];
                    stk_var[sp] = v;
                    stk_entry[sp] = !seen_write[v];
                    sp++;
                    break;
                }
                case _LOAD_SMALL_INT:
                    if (sp >= 32) { body_valid = false; break; }
                    stk_type[sp] = VAL_INT;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    sp++;
                    break;
                case _LOAD_CONST_INLINE:
                case _LOAD_CONST_INLINE_BORROW: {
                    if (sp >= 32) { body_valid = false; break; }
                    PyObject *c = (PyObject *)inst->operand0;
                    if (c != NULL && _PyLong_CheckExactAndCompact(c)) {
                        stk_type[sp] = VAL_INT;
                    }
                    else if (c != NULL && PyFloat_CheckExact(c)) {
                        stk_type[sp] = VAL_FLOAT;
                    }
                    else {
                        stk_type[sp] = VAL_UNKNOWN;
                    }
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    sp++;
                    break;
                }
                case _GUARD_TOS_INT:
                case _GUARD_NOS_INT: {
                    int idx = (op == _GUARD_TOS_INT) ? (sp - 1) : (sp - 2);
                    if (idx < 0) { body_valid = false; break; }
                    if (stk_type[idx] == VAL_INT) {
                        guard_kind[i] = 2;
                        guard_var[i] = -2;
                    }
                    else if (stk_entry[idx] && stk_var[idx] >= 0) {
                        guard_kind[i] = 2;
                        guard_var[i] = stk_var[idx];
                    }
                    stk_type[idx] = VAL_INT;
                    if (stk_var[idx] >= 0) {
                        int v = stk_var[idx];
                        cur_var_type[v] = VAL_INT;
                        for (int s = 0; s < sp; s++) {
                            if (stk_var[s] == v) {
                                stk_type[s] = VAL_INT;
                            }
                        }
                    }
                    break;
                }
                case _GUARD_TOS_FLOAT:
                case _GUARD_NOS_FLOAT: {
                    int idx = (op == _GUARD_TOS_FLOAT) ? (sp - 1) : (sp - 2);
                    if (idx < 0) { body_valid = false; break; }
                    if (stk_type[idx] == VAL_FLOAT) {
                        guard_kind[i] = 3;
                        guard_var[i] = -2;
                    }
                    else if (stk_entry[idx] && stk_var[idx] >= 0) {
                        guard_kind[i] = 3;
                        guard_var[i] = stk_var[idx];
                    }
                    stk_type[idx] = VAL_FLOAT;
                    if (stk_var[idx] >= 0) {
                        int v = stk_var[idx];
                        cur_var_type[v] = VAL_FLOAT;
                        for (int s = 0; s < sp; s++) {
                            if (stk_var[s] == v) {
                                stk_type[s] = VAL_FLOAT;
                            }
                        }
                    }
                    break;
                }
                case _GUARD_TOS_LIST:
                case _GUARD_TOS_TUPLE:
                case _GUARD_TOS_DICT:
                case _GUARD_TOS_UNICODE:
                case _GUARD_TYPE:
                case _GUARD_TYPE_VERSION:
                case _CHECK_MANAGED_OBJECT_HAS_VALUES:
                    if (sp < 1) { body_valid = false; break; }
                    if (stk_entry[sp - 1] && stk_var[sp - 1] >= 0 && !written[stk_var[sp - 1]]) {
                        guard_kind[i] = 4;
                        guard_var[i] = stk_var[sp - 1];
                    }
                    break;
                case _GUARD_NOS_LIST:
                case _GUARD_NOS_TUPLE:
                case _GUARD_NOS_UNICODE:
                case _GUARD_NOS_COMPACT_ASCII:
                    if (sp < 2) { body_valid = false; break; }
                    if (stk_entry[sp - 2] && stk_var[sp - 2] >= 0 && !written[stk_var[sp - 2]]) {
                        guard_kind[i] = 4;
                        guard_var[i] = stk_var[sp - 2];
                    }
                    break;
                case _BINARY_OP_SUBSCR_LIST_INT:
                case _BINARY_OP_SUBSCR_TUPLE_INT:
                    if (sp < 2 || sp >= 32) { body_valid = false; break; }
                    stk_type[sp - 2] = VAL_UNKNOWN;
                    stk_var[sp - 2] = -1;
                    stk_entry[sp - 2] = false;
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_type[sp] = VAL_UNKNOWN;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    sp++;
                    break;
                case _LOAD_ATTR_INSTANCE_VALUE:
                case _LOAD_ATTR_SLOT:
                    if (sp < 1 || sp >= 32) { body_valid = false; break; }
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_type[sp] = VAL_UNKNOWN;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    sp++;
                    break;
                case _BINARY_OP_ADD_INT:
                case _BINARY_OP_ADD_INT_INPLACE:
                case _BINARY_OP_ADD_INT_INPLACE_RIGHT:
                case _BINARY_OP_SUBTRACT_INT:
                case _BINARY_OP_SUBTRACT_INT_INPLACE:
                case _BINARY_OP_SUBTRACT_INT_INPLACE_RIGHT:
                case _BINARY_OP_MULTIPLY_INT:
                case _BINARY_OP_MULTIPLY_INT_INPLACE:
                case _BINARY_OP_MULTIPLY_INT_INPLACE_RIGHT:
                    if (sp < 2 || sp >= 32) { body_valid = false; break; }
                    stk_type[sp - 2] = VAL_INT;
                    stk_var[sp - 2] = -1;
                    stk_entry[sp - 2] = false;
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_type[sp] = VAL_UNKNOWN;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    sp++;
                    break;
                case _BINARY_OP_ADD_FLOAT:
                case _BINARY_OP_ADD_FLOAT_INPLACE:
                case _BINARY_OP_ADD_FLOAT_INPLACE_RIGHT:
                case _BINARY_OP_SUBTRACT_FLOAT:
                case _BINARY_OP_SUBTRACT_FLOAT_INPLACE:
                case _BINARY_OP_SUBTRACT_FLOAT_INPLACE_RIGHT:
                case _BINARY_OP_MULTIPLY_FLOAT:
                case _BINARY_OP_MULTIPLY_FLOAT_INPLACE:
                case _BINARY_OP_MULTIPLY_FLOAT_INPLACE_RIGHT:
                case _BINARY_OP_TRUEDIV_FLOAT:
                case _BINARY_OP_TRUEDIV_FLOAT_INPLACE:
                case _BINARY_OP_TRUEDIV_FLOAT_INPLACE_RIGHT:
                    if (sp < 2 || sp >= 32) { body_valid = false; break; }
                    stk_type[sp - 2] = VAL_FLOAT;
                    stk_var[sp - 2] = -1;
                    stk_entry[sp - 2] = false;
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_type[sp] = VAL_UNKNOWN;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    sp++;
                    break;
                case _UNARY_NEGATIVE_FLOAT_INPLACE:
                    if (sp < 1 || sp >= 32) { body_valid = false; break; }
                    stk_type[sp - 1] = VAL_FLOAT;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_type[sp] = VAL_UNKNOWN;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    sp++;
                    break;
                case _COMPARE_OP_INT:
                case _COMPARE_OP_FLOAT:
                    if (sp < 2 || sp >= 32) { body_valid = false; break; }
                    stk_type[sp - 2] = VAL_UNKNOWN;
                    stk_var[sp - 2] = -1;
                    stk_entry[sp - 2] = false;
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_type[sp] = VAL_UNKNOWN;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    sp++;
                    break;
                case _POP_TOP:
                case _POP_TOP_NOP:
                case _POP_TOP_INT:
                case _POP_TOP_FLOAT:
                case _POP_TOP_UNICODE:
                case _GUARD_IS_TRUE_POP:
                case _GUARD_IS_FALSE_POP:
                case _GUARD_BIT_IS_SET_POP:
                case _GUARD_BIT_IS_UNSET_POP:
                    if (sp < 1) { body_valid = false; break; }
                    sp--;
                    break;
                case _SWAP_FAST:
                case _SWAP_FAST_0:
                case _SWAP_FAST_1:
                case _SWAP_FAST_2:
                case _SWAP_FAST_3:
                case _SWAP_FAST_4:
                case _SWAP_FAST_5:
                case _SWAP_FAST_6:
                case _SWAP_FAST_7: {
                    if (sp < 1) { body_valid = false; break; }
                    int v = (op == _SWAP_FAST) ? (int)inst->oparg : (int)(op - _SWAP_FAST_0);
                    int val_type = stk_type[sp - 1];
                    if (val_type != VAL_INT && val_type != VAL_FLOAT && val_type != VAL_ITER_ITEM) {
                        body_valid = false;
                        break;
                    }
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    for (int s = 0; s < sp; s++) {
                        if (stk_var[s] == v) {
                            stk_var[s] = -1;
                            stk_entry[s] = false;
                        }
                    }
                    seen_write[v] = true;
                    cur_var_type[v] = val_type;
                    if (val_type != VAL_INT) {
                        all_stores_int[v] = false;
                    }
                    if (val_type != VAL_FLOAT) {
                        all_stores_float[v] = false;
                    }
                    break;
                }
                default:
                    body_valid = false;
                    break;
            }
        }

        if (sp != 0) {
            body_valid = false;
        }

        bool hoist_inst[32] = {0};
        int hoist_type_guard_count = 0;
        if (body_valid) {
            for (int i = 0; i < b_len; i++) {
                int gk = guard_kind[i];
                int gv = guard_var[i];
                if (gk == 1) {
                    hoist_inst[i] = true;
                }
                else if (gk == 2) {
                    if (gv == -2 || (gv >= 0 && (!written[gv] || all_stores_int[gv]))) {
                        hoist_inst[i] = true;
                        hoist_type_guard_count++;
                    }
                }
                else if (gk == 3) {
                    if (gv == -2 || (gv >= 0 && (!written[gv] || all_stores_float[gv]))) {
                        hoist_inst[i] = true;
                        hoist_type_guard_count++;
                    }
                }
                else if (gk == 4) {
                    if (gv >= 0 && !written[gv]) {
                        hoist_inst[i] = true;
                        hoist_type_guard_count++;
                    }
                }
            }
            /* Ensure every _POP_TOP in body can be specialized to _POP_TOP_INT
             * or _POP_TOP_FLOAT in sbody so sbody has no escaping decrefs. */
            for (int i = 0; i < b_len; i++) {
                if (body[i].opcode == _POP_TOP) {
                    bool pop_specialized = false;
                    if (i > 0) {
                        uint16_t prev_op = body[i - 1].opcode;
                        int v = -1;
                        if (prev_op == _SWAP_FAST) {
                            v = (int)body[i - 1].oparg;
                        }
                        else if (prev_op >= _SWAP_FAST_0 && prev_op <= _SWAP_FAST_7) {
                            v = (int)(prev_op - _SWAP_FAST_0);
                        }
                        if (v >= 0 && v < 64 && written[v]) {
                            bool sbody_int = all_stores_int[v] ||
                                (store_count[v] == 1 && cur_var_type[v] == VAL_INT);
                            bool sbody_float = all_stores_float[v] ||
                                (store_count[v] == 1 && cur_var_type[v] == VAL_FLOAT);
                            if (sbody_int || sbody_float) {
                                pop_specialized = true;
                            }
                        }
                    }
                    if (!pop_specialized) {
                        body_valid = false;
                        break;
                    }
                }
            }
        }

        if (body_valid && hoist_type_guard_count > 0) {
            _PyUOpInstruction make_warm_inst = output[1];
            _PyUOpInstruction jump_inst = output[length - 1];
            int out_idx = 1;
            /* Emit Pre-Header (Iteration 1 with all type and iterator guards) */
            for (int i = 0; i < b_len; i++) {
                uint16_t op = body[i].opcode;
                if (op == _CHECK_PERIODIC ||
                    op == _CHECK_PERIODIC_IF_NOT_YIELD_FROM)
                {
                    continue;
                }
                output[out_idx++] = body[i];
            }
            /* Loop Header */
            output[out_idx++] = make_warm_inst;
            /* Build Steady-State Loop Body with hoisted guards removed */
            _PyUOpInstruction sbody[32];
            int s_len = 0;
            for (int i = 0; i < b_len; i++) {
                if (hoist_inst[i]) {
                    continue;
                }
                sbody[s_len++] = body[i];
            }
            /* Specialize _SWAP_FAST v + _POP_TOP in steady-state loop body:
             * after Iteration 1 (Pre-Header), any local v written in the loop
             * holds a known int or float at the point of _SWAP_FAST. */
            for (int i = 0; i + 1 < s_len; i++) {
                uint16_t sop = sbody[i].opcode;
                int v = -1;
                if (sop == _SWAP_FAST) {
                    v = (int)sbody[i].oparg;
                }
                else if (sop >= _SWAP_FAST_0 && sop <= _SWAP_FAST_7) {
                    v = (int)(sop - _SWAP_FAST_0);
                }
                if (v >= 0 && v < 64 && written[v] &&
                    sbody[i + 1].opcode == _POP_TOP)
                {
                    if (all_stores_int[v] ||
                        (store_count[v] == 1 && cur_var_type[v] == VAL_INT))
                    {
                        sbody[i + 1].opcode = _POP_TOP_INT;
                    }
                    else if (all_stores_float[v] ||
                             (store_count[v] == 1 && cur_var_type[v] == VAL_FLOAT))
                    {
                        sbody[i + 1].opcode = _POP_TOP_FLOAT;
                    }
                }
            }
            /* In-place ownership transfer for int/float accumulators in steady-state
             * loop body: after Iteration 1 creates and stores a uniquely-owned
             * PyLongObject/PyFloatObject into locals[v], use _LOAD_FAST_AND_CLEAR v
             * so _BINARY_OP_{ADD,SUBTRACT,MULTIPLY}_{INT,FLOAT} mutates locals[v]
             * in place on iterations 2..N without allocating or touching the freelist. */
            for (int i = 0; i + 6 < s_len; i++) {
                if (sbody[i].opcode != _LOAD_FAST_BORROW) {
                    continue;
                }
                int v = (int)sbody[i].oparg;
                if (v < 0 || v >= 64 ||
                    !written[v] ||
                    load_count[v] != 1 ||
                    store_count[v] != 1)
                {
                    continue;
                }
                if (all_stores_float[v]) {
                    bool rhs_ok =
                        (sbody[i + 1].opcode == _LOAD_FAST_BORROW &&
                         (int)sbody[i + 1].oparg != v) ||
                        sbody[i + 1].opcode == _LOAD_CONST_INLINE_BORROW;
                    uint16_t binop = sbody[i + 2].opcode;
                    bool binop_ok =
                        binop == _BINARY_OP_ADD_FLOAT ||
                        binop == _BINARY_OP_SUBTRACT_FLOAT ||
                        binop == _BINARY_OP_MULTIPLY_FLOAT;
                    bool pops_ok =
                        (sbody[i + 3].opcode == _POP_TOP_NOP ||
                         sbody[i + 3].opcode == _POP_TOP_FLOAT) &&
                        sbody[i + 4].opcode == _POP_TOP_NOP;
                    if (!rhs_ok || !binop_ok || !pops_ok) {
                        continue;
                    }
                    int k = i + 5;
                    while (k + 3 < s_len &&
                           ((sbody[k].opcode == _LOAD_FAST_BORROW &&
                             (int)sbody[k].oparg != v) ||
                            sbody[k].opcode == _LOAD_CONST_INLINE_BORROW) &&
                           (sbody[k + 1].opcode == _BINARY_OP_ADD_FLOAT_INPLACE ||
                            sbody[k + 1].opcode == _BINARY_OP_SUBTRACT_FLOAT_INPLACE ||
                            sbody[k + 1].opcode == _BINARY_OP_MULTIPLY_FLOAT_INPLACE) &&
                           (sbody[k + 2].opcode == _POP_TOP_NOP ||
                            sbody[k + 2].opcode == _POP_TOP_FLOAT) &&
                           sbody[k + 3].opcode == _POP_TOP_NOP)
                    {
                        k += 4;
                    }
                    if (k + 1 < s_len &&
                        sbody[k].opcode == _SWAP_FAST &&
                        (int)sbody[k].oparg == v &&
                        sbody[k + 1].opcode == _POP_TOP_FLOAT)
                    {
                        sbody[i].opcode = _LOAD_FAST_AND_CLEAR;
                        sbody[i + 4].opcode = _POP_TOP_FLOAT;
                        sbody[k + 1].opcode = _POP_TOP_NOP;
                    }
                }
                else if (all_stores_int[v]) {
                    bool rhs_ok =
                        (sbody[i + 1].opcode == _LOAD_FAST_BORROW &&
                         (int)sbody[i + 1].oparg != v) ||
                        sbody[i + 1].opcode == _LOAD_CONST_INLINE_BORROW ||
                        sbody[i + 1].opcode == _LOAD_SMALL_INT;
                    uint16_t binop = sbody[i + 2].opcode;
                    bool binop_ok =
                        binop == _BINARY_OP_ADD_INT ||
                        binop == _BINARY_OP_SUBTRACT_INT ||
                        binop == _BINARY_OP_MULTIPLY_INT;
                    bool pops_ok =
                        (sbody[i + 3].opcode == _POP_TOP_NOP ||
                         sbody[i + 3].opcode == _POP_TOP_INT) &&
                        sbody[i + 4].opcode == _POP_TOP_NOP;
                    if (!rhs_ok || !binop_ok || !pops_ok) {
                        continue;
                    }
                    int k = i + 5;
                    while (k + 3 < s_len &&
                           ((sbody[k].opcode == _LOAD_FAST_BORROW &&
                             (int)sbody[k].oparg != v) ||
                            sbody[k].opcode == _LOAD_CONST_INLINE_BORROW ||
                            sbody[k].opcode == _LOAD_SMALL_INT) &&
                           (sbody[k + 1].opcode == _BINARY_OP_ADD_INT_INPLACE ||
                            sbody[k + 1].opcode == _BINARY_OP_SUBTRACT_INT_INPLACE ||
                            sbody[k + 1].opcode == _BINARY_OP_MULTIPLY_INT_INPLACE) &&
                           (sbody[k + 2].opcode == _POP_TOP_NOP ||
                            sbody[k + 2].opcode == _POP_TOP_INT) &&
                           (sbody[k + 3].opcode == _POP_TOP_NOP ||
                            sbody[k + 3].opcode == _POP_TOP_INT))
                    {
                        k += 4;
                    }
                    if (k + 1 < s_len &&
                        sbody[k].opcode == _SWAP_FAST &&
                        (int)sbody[k].oparg == v &&
                        sbody[k + 1].opcode == _POP_TOP_INT)
                    {
                        sbody[i].opcode = _LOAD_FAST_AND_CLEAR;
                        sbody[i + 4].opcode = _POP_TOP_INT;
                        sbody[k + 1].opcode = _POP_TOP_NOP;
                    }
                }
            }
            for (int i = 0; i < s_len; i++) {
                output[out_idx++] = sbody[i];
            }
            output[out_idx++] = jump_inst;
            return remove_unneeded_uops(output, out_idx);
        }
    }

    /* Fallback: Hoist leading _ITER_CHECK_{RANGE,LIST,TUPLE} (and depth-0
     * _CHECK_RECURSION_REMAINING across balanced inlined calls) before
     * _MAKE_WARM without duplicating the loop body. */
    int lead_set_ip_idx = -1;
    int lead_check_val_idx = -1;
    int iter_check_idx = -1;
    for (int i = 2; i < length - 1; i++) {
        uint16_t op = output[i].opcode;
        if (op == _NOP ||
            op == _CHECK_PERIODIC ||
            op == _CHECK_PERIODIC_IF_NOT_YIELD_FROM ||
            op == _TIER2_RESUME_CHECK)
        {
            continue;
        }
        if (op == _SET_IP) {
            lead_set_ip_idx = i;
            continue;
        }
        if (op == _CHECK_VALIDITY) {
            lead_check_val_idx = i;
            continue;
        }
        if (op == _ITER_CHECK_RANGE ||
            op == _ITER_CHECK_LIST ||
            op == _ITER_CHECK_TUPLE)
        {
            iter_check_idx = i;
        }
        break;
    }

    if (iter_check_idx >= 0 && length + 3 <= UOP_MAX_TRACE_LENGTH) {
        uint16_t first_op = output[iter_check_idx].opcode;
        bool safe_to_hoist_iter = true;
        int inline_depth = 0;
        int first_recursion_check_idx = -1;
        bool can_hoist_recursion_check = true;
        for (int i = iter_check_idx + 1; i < length - 1; i++) {
            uint16_t op = output[i].opcode;
            if (op == _NOP) {
                continue;
            }
            if (op == _PUSH_FRAME || op == _CREATE_INIT_FRAME) {
                inline_depth++;
                continue;
            }
            if (op == _RETURN_VALUE) {
                inline_depth--;
                if (inline_depth < 0) {
                    safe_to_hoist_iter = false;
                    break;
                }
                continue;
            }
            if (op == _RETURN_GENERATOR ||
                op == _YIELD_VALUE ||
                op == _DYNAMIC_EXIT)
            {
                safe_to_hoist_iter = false;
                break;
            }
            if (op == _CHECK_RECURSION_REMAINING) {
                if (inline_depth == 0) {
                    if (first_recursion_check_idx < 0) {
                        first_recursion_check_idx = i;
                    }
                }
                else {
                    can_hoist_recursion_check = false;
                }
            }
            if ((_PyUop_Flags[op] & HAS_ESCAPES_FLAG) &&
                op != _CHECK_PERIODIC &&
                op != _CHECK_PERIODIC_IF_NOT_YIELD_FROM &&
                op != _TIER2_RESUME_CHECK &&
                op != _POP_TOP &&
                op != _RETURN_VALUE)
            {
                can_hoist_recursion_check = false;
            }
#ifdef Py_GIL_DISABLED
            if (first_op == _ITER_CHECK_LIST &&
                (_PyUop_Flags[op] & HAS_ESCAPES_FLAG))
            {
                safe_to_hoist_iter = false;
                break;
            }
#endif
        }
        if (inline_depth != 0) {
            safe_to_hoist_iter = false;
        }
        if (safe_to_hoist_iter) {
            _PyUOpInstruction temp[UOP_MAX_TRACE_LENGTH];
            int out_idx = 0;
            temp[out_idx++] = output[0]; /* _START_EXECUTOR */
            if (lead_set_ip_idx >= 0) {
                temp[out_idx++] = output[lead_set_ip_idx];
            }
            if (lead_check_val_idx >= 0) {
                temp[out_idx++] = output[lead_check_val_idx];
            }
            temp[out_idx++] = output[iter_check_idx];
            if (can_hoist_recursion_check && first_recursion_check_idx >= 0) {
                _PyUOpInstruction rec_inst = output[first_recursion_check_idx];
                rec_inst.target = output[iter_check_idx].target;
                temp[out_idx++] = rec_inst;
            }
            temp[out_idx++] = output[1]; /* _MAKE_WARM */
            for (int i = 2; i < length; i++) {
                if (i == iter_check_idx || output[i].opcode == _NOP) {
                    continue;
                }
                temp[out_idx++] = output[i];
            }
            memcpy(output, temp, out_idx * sizeof(_PyUOpInstruction));
            return remove_unneeded_uops(output, out_idx);
        }
    }

    return length;
}

//  0 - failure, no error raised, just fall back to Tier 1
// -1 - failure, and raise error
//  > 0 - length of optimized trace
int
_Py_uop_analyze_and_optimize(
    _PyThreadStateImpl *tstate,
    _PyUOpInstruction *buffer,
    int length,
    int curr_stacklen,
    _PyUOpInstruction *output,
    _PyBloomFilter *dependencies
)
{
    OPT_STAT_INC(optimizer_attempts);

    length = optimize_uops(
        tstate, buffer, length, curr_stacklen, output, dependencies);

    if (length == 0) {
        return length;
    }

    assert(length > 0);

    length = remove_unneeded_uops(output, length);
    assert(length > 0);

    length = hoist_loop_preheader_guards(output, length, curr_stacklen);

    OPT_STAT_INC(optimizer_successes);
    return length;
}

#endif /* _Py_TIER2 */
