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
#define sym_set_inline_values _Py_uop_sym_set_inline_values
#define sym_has_inline_values _Py_uop_sym_has_inline_values
#define sym_clear_inline_values _Py_uop_sym_clear_inline_values
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

    if (((uintptr_t)frame->code->co_extra & 0x1) != 0 &&
        trace_len > 0)
    {
        PyCodeObject *co = frame->code;
        int nlocals = co->co_nlocals < 64 ? co->co_nlocals : 64;
        int writes[64] = {0};
        int const_init[64];
        for (int v = 0; v < 64; v++) {
            const_init[v] = -1;
        }
        _Py_CODEUNIT *code = _PyCode_CODE(co);
        Py_ssize_t code_len = Py_SIZE(co);
        bool in_entry_bb = true;
        int prev_const = -1;
        for (Py_ssize_t pc = 0; pc < code_len; ) {
            uint8_t raw_op = code[pc].op.code;
            uint8_t op = _PyOpcode_Deopt[raw_op];
            int arg = code[pc].op.arg;
            int caches = _PyOpcode_Caches[op];
            if (op == EXTENDED_ARG) {
                pc += 1;
                continue;
            }
            if (op == LOAD_CONST) {
                prev_const = in_entry_bb ? arg : -1;
            }
            else if (op == STORE_FAST) {
                if (arg < 64) {
                    writes[arg]++;
                    if (in_entry_bb && prev_const >= 0 && writes[arg] == 1) {
                        const_init[arg] = prev_const;
                    }
                }
                prev_const = -1;
            }
            else if (op == STORE_FAST_LOAD_FAST) {
                int st_arg = arg >> 4;
                if (st_arg < 64) {
                    writes[st_arg]++;
                    if (in_entry_bb && prev_const >= 0 && writes[st_arg] == 1) {
                        const_init[st_arg] = prev_const;
                    }
                }
                prev_const = -1;
            }
            else if (op == STORE_FAST_STORE_FAST) {
                int st1 = arg >> 4;
                int st2 = arg & 0xf;
                if (st1 < 64) {
                    writes[st1]++;
                }
                if (st2 < 64) {
                    writes[st2]++;
                }
                prev_const = -1;
            }
            else if (op == DELETE_FAST) {
                if (arg < 64) {
                    writes[arg] += 2;
                }
                prev_const = -1;
            }
            else {
                if (op == FOR_ITER ||
                    op == JUMP_BACKWARD || op == JUMP_BACKWARD_NO_INTERRUPT ||
                    op == JUMP_FORWARD ||
                    op == POP_JUMP_IF_FALSE || op == POP_JUMP_IF_TRUE ||
                    op == POP_JUMP_IF_NONE || op == POP_JUMP_IF_NOT_NONE ||
                    op == RETURN_VALUE || op == RETURN_GENERATOR || op == YIELD_VALUE)
                {
                    in_entry_bb = false;
                }
                prev_const = -1;
            }
            pc += 1 + caches;
        }
        PyObject *consts = co->co_consts;
        for (int v = co->co_argcount; v < nlocals; v++) {
            if (writes[v] == 1 && const_init[v] >= 0 &&
                const_init[v] < PyTuple_GET_SIZE(consts))
            {
                PyObject *cval = PyTuple_GET_ITEM(consts, const_init[v]);
                if (PyFloat_CheckExact(cval)) {
                    frame->locals[v] = _Py_uop_sym_new_type(ctx, &PyFloat_Type);
                }
                else if (_PyLong_CheckExactAndCompact(cval)) {
                    frame->locals[v] = _Py_uop_sym_new_compact_int(ctx);
                }
            }
        }
    }

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
                else if (opcode == _TIER2_RESUME_CHECK &&
                         call_depth > 0 &&
                         (buffer[pc].oparg & 0x80) != 0)
                {
                    buffer[pc].opcode = _NOP;
                    break;
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
                bool store_attr_no_escape =
                    (opcode == _STORE_ATTR_SLOT &&
                     ((buffer[pc].operand0 >> 13) != 0 || buffer[pc].oparg == 255)) ||
                    (opcode == _STORE_ATTR_INSTANCE_VALUE && buffer[pc].oparg == 255);
                if ((_PyUop_Flags[opcode] & HAS_ESCAPES_FLAG) && !store_attr_no_escape) {
                    needs_ip = true;
                    may_have_escaped = true;
                    if (opcode != _CHECK_PERIODIC &&
                        opcode != _CHECK_PERIODIC_IF_NOT_YIELD_FROM &&
                        opcode != _TIER2_RESUME_CHECK &&
                        opcode != _RETURN_VALUE &&
                        opcode != _POP_TOP &&
                        opcode != _MAKE_HEAP_SAFE &&
                        opcode != _STORE_ATTR_SLOT &&
                        opcode != _STORE_ATTR_INSTANCE_VALUE &&
                        opcode != _BINARY_OP_TRUEDIV_FLOAT &&
                        opcode != _BINARY_OP_TRUEDIV_FLOAT_INPLACE &&
                        opcode != _BINARY_OP_TRUEDIV_FLOAT_INPLACE_RIGHT)
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

static void
mark_inplace_float_attr_updates(_PyUOpInstruction *buffer, int length)
{
    #define NEXT_ACT_IDX(idx) \
        ({ int _i = (idx); \
           while (_i < length && \
                  (buffer[_i].opcode == _NOP || \
                   buffer[_i].opcode == _SET_IP || \
                   buffer[_i].opcode == _CHECK_VALIDITY)) { \
               _i++; \
           } \
           _i; })
    for (int i = 0; i < length; i++) {
        uint16_t op = buffer[i].opcode;
        int v = -1;
        if (op == _LOAD_FAST || op == _LOAD_FAST_BORROW) {
            v = (int)buffer[i].oparg;
        }
        else if (op >= _LOAD_FAST_0 && op <= _LOAD_FAST_7) {
            v = (int)(op - _LOAD_FAST_0);
        }
        else if (op >= _LOAD_FAST_BORROW_0 && op <= _LOAD_FAST_BORROW_7) {
            v = (int)(op - _LOAD_FAST_BORROW_0);
        }
        if (v < 0) {
            continue;
        }
        int j = NEXT_ACT_IDX(i + 1);
        while (j < length &&
               (buffer[j].opcode == _GUARD_TYPE_VERSION ||
                buffer[j].opcode == _GUARD_TYPE ||
                buffer[j].opcode == _CHECK_FUNCTION_VERSION_INLINE ||
                buffer[j].opcode == _CHECK_MANAGED_OBJECT_HAS_VALUES ||
                buffer[j].opcode == _GUARD_DORV_NO_DICT))
        {
            j = NEXT_ACT_IDX(j + 1);
        }
        if (j >= length ||
            (buffer[j].opcode != _LOAD_ATTR_SLOT &&
             buffer[j].opcode != _LOAD_ATTR_INSTANCE_VALUE))
        {
            continue;
        }
        bool is_slot = (buffer[j].opcode == _LOAD_ATTR_SLOT);
        uint64_t offset = is_slot ? (buffer[j].operand0 & 0x1FFF) : buffer[j].operand0;
        int j_pop = NEXT_ACT_IDX(j + 1);
        if (j_pop >= length || buffer[j_pop].opcode != _POP_TOP_NOP) {
            continue;
        }
        int r = NEXT_ACT_IDX(j_pop + 1);
        if (r >= length) {
            continue;
        }
        uint16_t rop = buffer[r].opcode;
        int rv = -1;
        if (rop == _LOAD_FAST || rop == _LOAD_FAST_BORROW) {
            rv = (int)buffer[r].oparg;
        }
        else if (rop >= _LOAD_FAST_0 && rop <= _LOAD_FAST_7) {
            rv = (int)(rop - _LOAD_FAST_0);
        }
        else if (rop >= _LOAD_FAST_BORROW_0 && rop <= _LOAD_FAST_BORROW_7) {
            rv = (int)(rop - _LOAD_FAST_BORROW_0);
        }
        bool rhs_ok = (rop == _LOAD_CONST_INLINE_BORROW ||
                       rop == _LOAD_CONST_INLINE ||
                       (rv >= 0 && rv != v));
        int b = NEXT_ACT_IDX(r + 1);
        int rhs_slot_idx = -1;
        if (rv >= 0) {
            int r2 = b;
            while (r2 < length &&
                   (buffer[r2].opcode == _GUARD_TYPE_VERSION ||
                    buffer[r2].opcode == _GUARD_TYPE ||
                    buffer[r2].opcode == _CHECK_FUNCTION_VERSION_INLINE ||
                    buffer[r2].opcode == _CHECK_MANAGED_OBJECT_HAS_VALUES ||
                    buffer[r2].opcode == _GUARD_DORV_NO_DICT))
            {
                r2 = NEXT_ACT_IDX(r2 + 1);
            }
            if (r2 < length &&
                (buffer[r2].opcode == _LOAD_ATTR_SLOT ||
                 buffer[r2].opcode == _LOAD_ATTR_INSTANCE_VALUE))
            {
                bool r_slot = (buffer[r2].opcode == _LOAD_ATTR_SLOT);
                uint64_t r_off = r_slot ? (buffer[r2].operand0 & 0x1FFF) : buffer[r2].operand0;
                int r_pop = NEXT_ACT_IDX(r2 + 1);
                if (r_pop < length && buffer[r_pop].opcode == _POP_TOP_NOP &&
                    (rv != v || r_off != offset || r_slot != is_slot))
                {
                    rhs_ok = true;
                    if (r_slot && (buffer[r2].operand0 >> 13) == 2) {
                        rhs_slot_idx = r2;
                    }
                    b = NEXT_ACT_IDX(r_pop + 1);
                }
                else {
                    rhs_ok = false;
                }
            }
        }
        if (!rhs_ok) {
            continue;
        }
        while (b < length &&
               (buffer[b].opcode == _GUARD_TOS_FLOAT ||
                buffer[b].opcode == _GUARD_NOS_FLOAT))
        {
            b = NEXT_ACT_IDX(b + 1);
        }
        if (b >= length ||
            (buffer[b].opcode != _BINARY_OP_ADD_FLOAT &&
             buffer[b].opcode != _BINARY_OP_SUBTRACT_FLOAT &&
             buffer[b].opcode != _BINARY_OP_MULTIPLY_FLOAT))
        {
            continue;
        }
        int b_pop1 = NEXT_ACT_IDX(b + 1);
        int b_pop2 = NEXT_ACT_IDX(b_pop1 + 1);
        if (b_pop2 >= length ||
            (buffer[b_pop1].opcode != _POP_TOP_NOP &&
             buffer[b_pop1].opcode != _POP_TOP_FLOAT) ||
            buffer[b_pop2].opcode != _POP_TOP_FLOAT)
        {
            continue;
        }
        int s = NEXT_ACT_IDX(b_pop2 + 1);
        if (s >= length) {
            continue;
        }
        uint16_t sop = buffer[s].opcode;
        int sv = -1;
        if (sop == _LOAD_FAST || sop == _LOAD_FAST_BORROW) {
            sv = (int)buffer[s].oparg;
        }
        else if (sop >= _LOAD_FAST_0 && sop <= _LOAD_FAST_7) {
            sv = (int)(sop - _LOAD_FAST_0);
        }
        else if (sop >= _LOAD_FAST_BORROW_0 && sop <= _LOAD_FAST_BORROW_7) {
            sv = (int)(sop - _LOAD_FAST_BORROW_0);
        }
        if (sv != v) {
            continue;
        }
        int s_next = NEXT_ACT_IDX(s + 1);
        int st = s_next;
        while (st < length &&
               (buffer[st].opcode == _LOCK_OBJECT ||
                buffer[st].opcode == _GUARD_TYPE_VERSION ||
                buffer[st].opcode == _GUARD_TYPE_VERSION_LOCKED ||
                buffer[st].opcode == _GUARD_DORV_NO_DICT))
        {
            st = NEXT_ACT_IDX(st + 1);
        }
        uint16_t exp_store = is_slot ? _STORE_ATTR_SLOT : _STORE_ATTR_INSTANCE_VALUE;
        if (st < length && buffer[st].opcode == exp_store) {
            uint64_t st_off = is_slot ? (buffer[st].operand0 & 0x1FFF) : buffer[st].operand0;
            if (st_off == offset) {
                int st_pop = NEXT_ACT_IDX(st + 1);
                bool s_borrowed = (sop == _LOAD_FAST_BORROW ||
                                   (sop >= _LOAD_FAST_BORROW_0 && sop <= _LOAD_FAST_BORROW_7));
                if (is_slot && (buffer[j].operand0 >> 13) == 2 &&
                    s_next == st && s_borrowed &&
                    st_pop < length && buffer[st_pop].opcode == _POP_TOP_NOP)
                {
                    buffer[j].oparg = 255;
                    if (rhs_slot_idx >= 0 && buffer[b_pop1].opcode == _POP_TOP_FLOAT) {
                        buffer[rhs_slot_idx].oparg = 254;
                        buffer[b_pop1].opcode = _POP_TOP_NOP;
                    }
                    if (buffer[b].opcode == _BINARY_OP_ADD_FLOAT) {
                        buffer[b].opcode = _BINARY_OP_ADD_FLOAT_INPLACE;
                    }
                    else if (buffer[b].opcode == _BINARY_OP_SUBTRACT_FLOAT) {
                        buffer[b].opcode = _BINARY_OP_SUBTRACT_FLOAT_INPLACE;
                    }
                    else {
                        buffer[b].opcode = _BINARY_OP_MULTIPLY_FLOAT_INPLACE;
                    }
                    buffer[b_pop2].opcode = _POP_TOP_NOP;
                    buffer[s].opcode = _POP_TOP_NOP;
                    buffer[s].oparg = 0;
                    buffer[s].operand0 = 0;
                    buffer[st].opcode = _NOP;
                    buffer[st_pop].opcode = _NOP;
                }
                else {
                    buffer[b].oparg = 255;
                    buffer[st].oparg = 255;
                }
            }
        }
    }

    /* Second sub-pass: fuse float FMA patterns:
     * (1) Temp accumulator FMA:
     *     acc_op + 2 pops +
     *     load_a + load_b + _BINARY_OP_MULTIPLY_FLOAT + 2 pops +
     *     _BINARY_OP_{ADD,SUBTRACT}_FLOAT_INPLACE + _POP_TOP_FLOAT + _POP_TOP_NOP
     * (2) Attribute accumulator FMA:
     *     self.attr = self.attr +/- (a * b)
     *     where self.attr is loaded on stack, followed by 1 or 2 loads +
     *     _BINARY_OP_MULTIPLY_FLOAT[_INPLACE[_RIGHT]] + 2 pops +
     *     _BINARY_OP_{ADD,SUBTRACT}_FLOAT[_INPLACE_RIGHT] + 2 pops + store to same self.attr.
     */
    for (int i = 0; i < length; i++) {
        uint16_t op = buffer[i].opcode;
        if (op == _BINARY_OP_MULTIPLY_FLOAT || op == _BINARY_OP_MULTIPLY_FLOAT_INPLACE) {
            int p1 = NEXT_ACT_IDX(i + 1);
            int p2 = NEXT_ACT_IDX(p1 + 1);
            int add_idx = NEXT_ACT_IDX(p2 + 1);
            if (add_idx < length &&
                (buffer[p1].opcode == _POP_TOP_FLOAT || buffer[p1].opcode == _POP_TOP_NOP) &&
                (buffer[p2].opcode == _POP_TOP_FLOAT || buffer[p2].opcode == _POP_TOP_NOP) &&
                (buffer[add_idx].opcode == _BINARY_OP_ADD_FLOAT_INPLACE ||
                 buffer[add_idx].opcode == _BINARY_OP_SUBTRACT_FLOAT_INPLACE))
            {
                int ap1 = NEXT_ACT_IDX(add_idx + 1);
                int ap2 = NEXT_ACT_IDX(ap1 + 1);
                if (ap2 < length &&
                    buffer[ap1].opcode == _POP_TOP_FLOAT &&
                    buffer[ap2].opcode == _POP_TOP_NOP)
                {
                    int is_sub = (buffer[add_idx].opcode == _BINARY_OP_SUBTRACT_FLOAT_INPLACE) ? 1 : 0;
                    buffer[i].opcode = _BINARY_OP_FMA_FLOAT_INPLACE;
                    buffer[i].oparg = is_sub;
                    if (op == _BINARY_OP_MULTIPLY_FLOAT_INPLACE) {
                        buffer[p1].opcode = _POP_TOP_FLOAT;
                    }
                    buffer[add_idx].opcode = _NOP;
                    buffer[ap1].opcode = _NOP;
                    buffer[ap2].opcode = _NOP;
                }
            }
        }
    }

    for (int i = 0; i < length; i++) {
        uint16_t op = buffer[i].opcode;
        int v = -1;
        if (op == _LOAD_FAST || op == _LOAD_FAST_BORROW) {
            v = (int)buffer[i].oparg;
        }
        else if (op >= _LOAD_FAST_0 && op <= _LOAD_FAST_7) {
            v = (int)(op - _LOAD_FAST_0);
        }
        else if (op >= _LOAD_FAST_BORROW_0 && op <= _LOAD_FAST_BORROW_7) {
            v = (int)(op - _LOAD_FAST_BORROW_0);
        }
        if (v < 0) {
            continue;
        }
        int j = NEXT_ACT_IDX(i + 1);
        while (j < length &&
               (buffer[j].opcode == _GUARD_TYPE_VERSION ||
                buffer[j].opcode == _GUARD_TYPE ||
                buffer[j].opcode == _CHECK_FUNCTION_VERSION_INLINE ||
                buffer[j].opcode == _CHECK_MANAGED_OBJECT_HAS_VALUES ||
                buffer[j].opcode == _GUARD_DORV_NO_DICT))
        {
            j = NEXT_ACT_IDX(j + 1);
        }
        if (j >= length ||
            (buffer[j].opcode != _LOAD_ATTR_SLOT &&
             buffer[j].opcode != _LOAD_ATTR_INSTANCE_VALUE))
        {
            continue;
        }
        bool is_slot = (buffer[j].opcode == _LOAD_ATTR_SLOT);
        uint64_t offset = is_slot ? (buffer[j].operand0 & 0x1FFF) : buffer[j].operand0;
        int j_pop = NEXT_ACT_IDX(j + 1);
        if (j_pop >= length || buffer[j_pop].opcode != _POP_TOP_NOP) {
            continue;
        }
        /* Scan forward up to 14 instructions for a multiplication followed immediately
         * by _BINARY_OP_{ADD,SUBTRACT}_FLOAT[_INPLACE_RIGHT] and store back to the same (v, offset).
         * Ensure none of the intermediate loads alias (v, offset). */
        int cur = NEXT_ACT_IDX(j_pop + 1);
        int mul_idx = -1;
        bool safe_rhs = true;
        int steps = 0;
        while (cur < length && steps < 14) {
            uint16_t cop = buffer[cur].opcode;
            if (cop == _LOAD_ATTR_SLOT || cop == _LOAD_ATTR_INSTANCE_VALUE) {
                bool c_slot = (cop == _LOAD_ATTR_SLOT);
                uint64_t c_off = c_slot ? (buffer[cur].operand0 & 0x1FFF) : buffer[cur].operand0;
                if (c_slot == is_slot && c_off == offset) {
                    /* Could alias self.attr if loaded from same local v */
                    safe_rhs = false;
                    break;
                }
            }
            if (cop == _BINARY_OP_MULTIPLY_FLOAT ||
                cop == _BINARY_OP_MULTIPLY_FLOAT_INPLACE ||
                cop == _BINARY_OP_MULTIPLY_FLOAT_INPLACE_RIGHT)
            {
                int mp1 = NEXT_ACT_IDX(cur + 1);
                int mp2 = NEXT_ACT_IDX(mp1 + 1);
                int nx = NEXT_ACT_IDX(mp2 + 1);
                while (nx < length &&
                       (buffer[nx].opcode == _GUARD_TOS_FLOAT ||
                        buffer[nx].opcode == _GUARD_NOS_FLOAT))
                {
                    nx = NEXT_ACT_IDX(nx + 1);
                }
                if (nx < length &&
                    (buffer[nx].opcode == _BINARY_OP_ADD_FLOAT ||
                     buffer[nx].opcode == _BINARY_OP_SUBTRACT_FLOAT ||
                     buffer[nx].opcode == _BINARY_OP_ADD_FLOAT_INPLACE_RIGHT ||
                     buffer[nx].opcode == _BINARY_OP_SUBTRACT_FLOAT_INPLACE_RIGHT))
                {
                    mul_idx = cur;
                    break;
                }
            }
            else if (cop == _SWAP_FAST || (cop >= _SWAP_FAST_0 && cop <= _SWAP_FAST_7) ||
                     cop == _STORE_ATTR_SLOT || cop == _STORE_ATTR_INSTANCE_VALUE ||
                     cop == _PUSH_FRAME || cop == _RETURN_VALUE)
            {
                safe_rhs = false;
                break;
            }
            cur = NEXT_ACT_IDX(cur + 1);
            steps++;
        }
        if (!safe_rhs || mul_idx < 0) {
            continue;
        }
        int mp1 = NEXT_ACT_IDX(mul_idx + 1);
        int mp2 = NEXT_ACT_IDX(mp1 + 1);
        int b = NEXT_ACT_IDX(mp2 + 1);
        while (b < length &&
               (buffer[b].opcode == _GUARD_TOS_FLOAT ||
                buffer[b].opcode == _GUARD_NOS_FLOAT))
        {
            buffer[b].opcode = _NOP;
            b = NEXT_ACT_IDX(b + 1);
        }
        if (b >= length ||
            (buffer[b].opcode != _BINARY_OP_ADD_FLOAT &&
             buffer[b].opcode != _BINARY_OP_SUBTRACT_FLOAT &&
             buffer[b].opcode != _BINARY_OP_ADD_FLOAT_INPLACE_RIGHT &&
             buffer[b].opcode != _BINARY_OP_SUBTRACT_FLOAT_INPLACE_RIGHT))
        {
            continue;
        }
        int bp1 = NEXT_ACT_IDX(b + 1);
        int bp2 = NEXT_ACT_IDX(bp1 + 1);
        if (bp2 >= length ||
            (buffer[bp1].opcode != _POP_TOP_FLOAT && buffer[bp1].opcode != _POP_TOP_NOP) ||
            buffer[bp2].opcode != _POP_TOP_FLOAT)
        {
            continue;
        }
        int s = NEXT_ACT_IDX(bp2 + 1);
        if (s >= length) {
            continue;
        }
        uint16_t sop = buffer[s].opcode;
        int sv = -1;
        if (sop == _LOAD_FAST || sop == _LOAD_FAST_BORROW) {
            sv = (int)buffer[s].oparg;
        }
        else if (sop >= _LOAD_FAST_0 && sop <= _LOAD_FAST_7) {
            sv = (int)(sop - _LOAD_FAST_0);
        }
        else if (sop >= _LOAD_FAST_BORROW_0 && sop <= _LOAD_FAST_BORROW_7) {
            sv = (int)(sop - _LOAD_FAST_BORROW_0);
        }
        if (sv != v) {
            continue;
        }
        int s_next = NEXT_ACT_IDX(s + 1);
        int st = s_next;
        while (st < length &&
               (buffer[st].opcode == _LOCK_OBJECT ||
                buffer[st].opcode == _GUARD_TYPE_VERSION ||
                buffer[st].opcode == _GUARD_TYPE_VERSION_LOCKED ||
                buffer[st].opcode == _GUARD_DORV_NO_DICT))
        {
            st = NEXT_ACT_IDX(st + 1);
        }
        uint16_t exp_store = is_slot ? _STORE_ATTR_SLOT : _STORE_ATTR_INSTANCE_VALUE;
        if (st < length && buffer[st].opcode == exp_store) {
            uint64_t st_off = is_slot ? (buffer[st].operand0 & 0x1FFF) : buffer[st].operand0;
            if (st_off == offset) {
                int is_sub = (buffer[b].opcode == _BINARY_OP_SUBTRACT_FLOAT ||
                              buffer[b].opcode == _BINARY_OP_SUBTRACT_FLOAT_INPLACE_RIGHT) ? 1 : 0;
                uint16_t old_mul = buffer[mul_idx].opcode;
                buffer[mul_idx].opcode = _BINARY_OP_FMA_FLOAT_INPLACE;
                if (old_mul == _BINARY_OP_MULTIPLY_FLOAT_INPLACE) {
                    buffer[mp1].opcode = _POP_TOP_FLOAT;
                }
                else if (old_mul == _BINARY_OP_MULTIPLY_FLOAT_INPLACE_RIGHT) {
                    buffer[mp2].opcode = _POP_TOP_FLOAT;
                }
                buffer[b].opcode = _NOP;
                buffer[bp1].opcode = _NOP;
                buffer[bp2].opcode = _NOP;
                int st_pop = NEXT_ACT_IDX(st + 1);
                bool s_borrowed = (sop == _LOAD_FAST_BORROW ||
                                   (sop >= _LOAD_FAST_BORROW_0 && sop <= _LOAD_FAST_BORROW_7));
                if (is_slot && (buffer[j].operand0 >> 13) == 2 &&
                    s_next == st && s_borrowed &&
                    st_pop < length && buffer[st_pop].opcode == _POP_TOP_NOP)
                {
                    buffer[j].oparg = 255;
                    buffer[mul_idx].oparg = 4 | is_sub;
                    buffer[s].opcode = _POP_TOP_NOP;
                    buffer[s].oparg = 0;
                    buffer[s].operand0 = 0;
                    buffer[st].opcode = _NOP;
                    buffer[st_pop].opcode = _NOP;
                }
                else {
                    buffer[mul_idx].oparg = 2 | is_sub;
                    buffer[st].oparg = 255;
                }
            }
        }
    }

    /* Fourth sub-pass: convert remaining read-only @struct int/float slot loads
     * (_LOAD_ATTR_SLOT with oparg == 0 and strict_tag == 1 or 2) that are immediately
     * consumed by a non-escaping float/int binary/compare op or FMA into borrowed slot loads
     * (oparg = 254) and turn their corresponding _POP_TOP_FLOAT/_POP_TOP_INT into _POP_TOP_NOP. */
    for (int i = 0; i < length; i++) {
        uint16_t slot_tag = (uint16_t)(buffer[i].operand0 >> 13);
        if (buffer[i].opcode != _LOAD_ATTR_SLOT ||
            buffer[i].oparg != 0 ||
            (slot_tag != 1 && slot_tag != 2))
        {
            continue;
        }
        int i_pop = NEXT_ACT_IDX(i + 1);
        if (i_pop >= length || buffer[i_pop].opcode != _POP_TOP_NOP) {
            continue;
        }
        int rel_sp = 1;
        int cur = NEXT_ACT_IDX(i_pop + 1);
        int steps = 0;
        while (cur < length && steps < 12 && rel_sp >= 1) {
            uint16_t cop = buffer[cur].opcode;
            if (cop == _LOAD_FAST_BORROW || cop == _LOAD_FAST ||
                (cop >= _LOAD_FAST_BORROW_0 && cop <= _LOAD_FAST_BORROW_7) ||
                (cop >= _LOAD_FAST_0 && cop <= _LOAD_FAST_7) ||
                cop == _LOAD_CONST_INLINE_BORROW || cop == _LOAD_CONST_INLINE)
            {
                rel_sp++;
            }
            else if (cop == _GUARD_TYPE_VERSION || cop == _GUARD_TYPE ||
                     cop == _CHECK_FUNCTION_VERSION_INLINE ||
                     cop == _GUARD_TOS_FLOAT || cop == _GUARD_NOS_FLOAT ||
                     cop == _GUARD_TOS_INT || cop == _GUARD_NOS_INT ||
                     cop == _GUARD_BINARY_OP_EXTEND ||
                     cop == _GUARD_BINARY_OP_EXTEND_LHS ||
                     cop == _GUARD_BINARY_OP_EXTEND_RHS ||
                     cop == _CHECK_MANAGED_OBJECT_HAS_VALUES || cop == _GUARD_DORV_NO_DICT)
            {
                /* stack depth unchanged */
            }
            else if (cop == _LOAD_ATTR_SLOT || cop == _LOAD_ATTR_INSTANCE_VALUE) {
                int c_pop = NEXT_ACT_IDX(cur + 1);
                if (c_pop >= length || buffer[c_pop].opcode != _POP_TOP_NOP) {
                    break;
                }
                cur = c_pop;
            }
            else if (rel_sp == 1 &&
                     (cop == _BINARY_OP_ADD_FLOAT ||
                      cop == _BINARY_OP_SUBTRACT_FLOAT ||
                      cop == _BINARY_OP_MULTIPLY_FLOAT ||
                      cop == _BINARY_OP_TRUEDIV_FLOAT ||
                      cop == _BINARY_OP_ADD_FLOAT_INPLACE ||
                      cop == _BINARY_OP_SUBTRACT_FLOAT_INPLACE ||
                      cop == _BINARY_OP_MULTIPLY_FLOAT_INPLACE ||
                      cop == _BINARY_OP_TRUEDIV_FLOAT_INPLACE ||
                      cop == _BINARY_OP_FMA_FLOAT_INPLACE ||
                      cop == _BINARY_OP_BITWISE_INT ||
                      cop == _BINARY_OP_ARITH_FLOAT_INT ||
                      cop == _BINARY_OP_EXTEND ||
                      cop == _COMPARE_OP_FLOAT ||
                      cop == _COMPARE_OP_INT))
            {
                int p1 = NEXT_ACT_IDX(cur + 1);
                if (p1 < length &&
                    (buffer[p1].opcode == _POP_TOP_FLOAT ||
                     buffer[p1].opcode == _POP_TOP_INT))
                {
                    buffer[i].oparg = 254;
                    buffer[p1].opcode = _POP_TOP_NOP;
                }
                break;
            }
            else if (rel_sp == 2 &&
                     (cop == _BINARY_OP_ADD_FLOAT ||
                      cop == _BINARY_OP_SUBTRACT_FLOAT ||
                      cop == _BINARY_OP_MULTIPLY_FLOAT ||
                      cop == _BINARY_OP_TRUEDIV_FLOAT ||
                      cop == _BINARY_OP_ADD_FLOAT_INPLACE_RIGHT ||
                      cop == _BINARY_OP_SUBTRACT_FLOAT_INPLACE_RIGHT ||
                      cop == _BINARY_OP_MULTIPLY_FLOAT_INPLACE_RIGHT ||
                      cop == _BINARY_OP_TRUEDIV_FLOAT_INPLACE_RIGHT ||
                      cop == _BINARY_OP_FMA_FLOAT_INPLACE ||
                      cop == _BINARY_OP_BITWISE_INT ||
                      cop == _BINARY_OP_ARITH_FLOAT_INT ||
                      cop == _BINARY_OP_EXTEND ||
                      cop == _COMPARE_OP_FLOAT ||
                      cop == _COMPARE_OP_INT))
            {
                int p1 = NEXT_ACT_IDX(cur + 1);
                int p2 = NEXT_ACT_IDX(p1 + 1);
                if (p2 < length &&
                    (buffer[p2].opcode == _POP_TOP_FLOAT ||
                     buffer[p2].opcode == _POP_TOP_INT))
                {
                    buffer[i].oparg = 254;
                    buffer[p2].opcode = _POP_TOP_NOP;
                }
                break;
            }
            else {
                break;
            }
            cur = NEXT_ACT_IDX(cur + 1);
            steps++;
        }
    }

    /* Deduplicate oparg == 255 (ensure-unique float slot loads) on the same
     * (owner_var, offset) once uniqueness is already established earlier in the
     * trace or loop iteration without any intervening strong reference load or store. */
    {
        bool written_in_trace[64] = {0};
        for (int i = 0; i < length; i++) {
            uint16_t op = buffer[i].opcode;
            int wv = -1;
            if (op == _SWAP_FAST || op == _LOAD_FAST_AND_CLEAR) {
                wv = (int)buffer[i].oparg;
            }
            else if (op >= _SWAP_FAST_0 && op <= _SWAP_FAST_7) {
                wv = (int)(op - _SWAP_FAST_0);
            }
            if (wv >= 0 && wv < 64) {
                written_in_trace[wv] = true;
            }
        }
        struct {
            int owner_var;
            uint64_t offset;
            bool active;
        } unique_slots[32];
        int num_unique = 0;
        int prev_owner_v = -1;
        for (int i = 0; i < length; i++) {
            uint16_t op = buffer[i].opcode;
            if (op == _NOP || op == _SET_IP || op == _CHECK_VALIDITY ||
                op == _GUARD_TYPE_VERSION || op == _GUARD_TYPE ||
                op == _CHECK_FUNCTION_VERSION_INLINE ||
                op == _CHECK_MANAGED_OBJECT_HAS_VALUES ||
                op == _GUARD_DORV_NO_DICT)
            {
                continue;
            }
            if (op == _LOAD_FAST || op == _LOAD_FAST_BORROW) {
                prev_owner_v = (int)buffer[i].oparg;
                continue;
            }
            if (op >= _LOAD_FAST_0 && op <= _LOAD_FAST_7) {
                prev_owner_v = (int)(op - _LOAD_FAST_0);
                continue;
            }
            if (op >= _LOAD_FAST_BORROW_0 && op <= _LOAD_FAST_BORROW_7) {
                prev_owner_v = (int)(op - _LOAD_FAST_BORROW_0);
                continue;
            }
            if (op == _MAKE_WARM) {
                for (int u = 0; u < num_unique; u++) {
                    if (!unique_slots[u].active) {
                        continue;
                    }
                    if (unique_slots[u].owner_var >= 0 &&
                        unique_slots[u].owner_var < 64 &&
                        written_in_trace[unique_slots[u].owner_var])
                    {
                        unique_slots[u].active = false;
                        continue;
                    }
                    for (int k = i + 1; k < length; k++) {
                        if (buffer[k].opcode == _STORE_ATTR_SLOT &&
                            (buffer[k].operand0 & 0x1FFF) == unique_slots[u].offset)
                        {
                            unique_slots[u].active = false;
                            break;
                        }
                        if (buffer[k].opcode == _LOAD_ATTR_SLOT &&
                            buffer[k].oparg == 0 &&
                            (buffer[k].operand0 & 0x1FFF) == unique_slots[u].offset)
                        {
                            unique_slots[u].active = false;
                            break;
                        }
                    }
                }
                prev_owner_v = -1;
                continue;
            }
            int wv = -1;
            if (op == _SWAP_FAST || op == _LOAD_FAST_AND_CLEAR) {
                wv = (int)buffer[i].oparg;
            }
            else if (op >= _SWAP_FAST_0 && op <= _SWAP_FAST_7) {
                wv = (int)(op - _SWAP_FAST_0);
            }
            if (wv >= 0) {
                for (int u = 0; u < num_unique; u++) {
                    if (unique_slots[u].owner_var == wv) {
                        unique_slots[u].active = false;
                    }
                }
                prev_owner_v = -1;
                continue;
            }
            if (op == _LOAD_ATTR_SLOT) {
                uint64_t off = buffer[i].operand0 & 0x1FFF;
                if (buffer[i].oparg == 0) {
                    for (int u = 0; u < num_unique; u++) {
                        if (unique_slots[u].offset == off) {
                            unique_slots[u].active = false;
                        }
                    }
                }
                else if (buffer[i].oparg == 255 && prev_owner_v >= 0) {
                    bool already_unique = false;
                    for (int u = 0; u < num_unique; u++) {
                        if (unique_slots[u].active &&
                            unique_slots[u].owner_var == prev_owner_v &&
                            unique_slots[u].offset == off)
                        {
                            already_unique = true;
                            break;
                        }
                    }
                    if (already_unique) {
                        buffer[i].oparg = 254;
                    }
                    else if (num_unique < 32) {
                        unique_slots[num_unique].owner_var = prev_owner_v;
                        unique_slots[num_unique].offset = off;
                        unique_slots[num_unique].active = true;
                        num_unique++;
                    }
                }
                prev_owner_v = -1;
                continue;
            }
            if (op == _STORE_ATTR_SLOT) {
                uint64_t off = buffer[i].operand0 & 0x1FFF;
                for (int u = 0; u < num_unique; u++) {
                    if (unique_slots[u].offset == off) {
                        unique_slots[u].active = false;
                    }
                }
                prev_owner_v = -1;
                continue;
            }
            if (op == _PUSH_FRAME || op == _RETURN_VALUE ||
                ((_PyUop_Flags[op] & HAS_ESCAPES_FLAG) &&
                 op != _CHECK_PERIODIC &&
                 op != _CHECK_PERIODIC_IF_NOT_YIELD_FROM &&
                 op != _TIER2_RESUME_CHECK &&
                 op != _POP_TOP &&
                 op != _POP_TOP_FLOAT &&
                 op != _POP_TOP_INT))
            {
                num_unique = 0;
            }
            prev_owner_v = -1;
        }
    }

    /* Fifth sub-pass: fuse @struct _LOAD_ATTR_SLOT (oparg == 254) + _POP_TOP_NOP
     * into _LOAD_ATTR_SLOT_NO_POP + _NOP so stack effect is (1 -- 1) instead of
     * (1 -- 2), avoiding _SPILL_OR_RELOAD at stack depth 3 and _POP_TOP_NOP,
     * and executing a single inline pointer load in the JIT/interpreter.
     */
    for (int i = 0; i < length; i++) {
        if (buffer[i].opcode != _LOAD_ATTR_SLOT) {
            continue;
        }
        if ((buffer[i].operand0 >> 13) == 0 || buffer[i].oparg != 254) {
            continue;
        }
        int p1 = NEXT_ACT_IDX(i + 1);
        if ( p1 < length && buffer[p1].opcode == _POP_TOP_NOP) {
            buffer[i].opcode = _LOAD_ATTR_SLOT_NO_POP;
            buffer[i].oparg = (uint16_t)(buffer[i].operand0 >> 13);
            buffer[i].operand0 &= 0x1FFF;
            buffer[p1].opcode = _NOP;
        }
    }
    #undef NEXT_ACT_IDX
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
    bool has_comp_append = false;
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
            case _GUARD_NOS_TYPE:
            case _GUARD_NOS_DICT_SUBSCRIPT:
            case _GUARD_TYPE_VERSION:
            case _CHECK_MANAGED_OBJECT_HAS_VALUES:
            case _GUARD_GLOBALS_VERSION:
            case _GUARD_BUILTINS_IS_CANONICAL:
            case _CHECK_FUNCTION_VERSION_INLINE:
            case _CHECK_STACK_SPACE_OPERAND:
            case _CHECK_RECURSION_REMAINING:
                type_guard_count++;
                break;
            case _GUARD_BINARY_OP_SUBSCR_TUPLE_INT_BOUNDS:
            case _GUARD_BINARY_OP_EXTEND:
            case _GUARD_BINARY_OP_EXTEND_LHS:
            case _GUARD_BINARY_OP_EXTEND_RHS:
            case _BINARY_OP_SUBSCR_LIST_INT:
            case _BINARY_OP_SUBSCR_TUPLE_INT:
            case _BINARY_OP_SUBSCR_DICT_KNOWN_HASH:
            case _BINARY_OP_SUBSCR_DICT:
            case _LOAD_ATTR_INSTANCE_VALUE:
            case _LOAD_ATTR_SLOT:
            case _LOAD_ATTR_SLOT_NO_POP:
            case _LOCK_OBJECT:
            case _GUARD_DORV_NO_DICT:
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
            case _BINARY_OP_FMA_FLOAT_INPLACE:
            case _BINARY_OP_BITWISE_INT:
            case _BINARY_OP_ARITH_FLOAT_INT:
            case _BINARY_OP_EXTEND:
                has_arith_binop = true;
                break;
            case _LIST_APPEND:
            case _SET_ADD:
                has_comp_append = true;
                break;
            case _STORE_SUBSCR_LIST_INT:
            case _STORE_ATTR_INSTANCE_VALUE:
            case _STORE_ATTR_SLOT:
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
     * list/tuple-iteration, and attribute-reading/writing loops where
     * loop-invariant or inductive local guards can be hoisted to Iteration 1. */
    if (can_peel_body &&
        (has_arith_binop || has_comp_append) &&
        type_guard_count > 0 &&
        pop_top_count <= 8 &&
        non_nop_count <= 256 &&
        !(has_iter_check && has_compare_op) &&
        length + non_nop_count <= UOP_MAX_TRACE_LENGTH)
    {
        _PyUOpInstruction body[256];
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
        typedef struct {
            int owner_var;
            uint64_t offset;
            bool is_slot;
            int cur_type;
            bool seen_write;
            bool entry_guarded;
            bool all_stores_int;
            bool all_stores_float;
            int load_count;
            int store_count;
        } AttrSlotInfo;
        AttrSlotInfo attr_slots[32];
        int num_attr_slots = 0;
        int attr_owner_var = -1;
        bool attr_tracking_valid = true;

        int stk_type[64];
        int stk_var[64];
        bool stk_entry[64];
        int stk_attr[64];
        bool stk_attr_entry[64];
        int sp = 0;
        bool seen_write[64] = {0};
        bool var_copied[64] = {0};
        int cur_var_type[64] = {0};
        bool all_stores_int[64];
        bool all_stores_float[64];
        for (int v = 0; v < 64; v++) {
            all_stores_int[v] = true;
            all_stores_float[v] = true;
            stk_attr[v] = -1;
            stk_attr_entry[v] = false;
        }

        /* guard_kind[i]:
         *   0 = not a hoistable guard
         *   1 = iterator/version check (_ITER_CHECK_*, _GUARD_GLOBALS_VERSION, etc.)
         *   2 = int guard on entry value of local guard_var[i] (or proven int if -2)
         *   3 = float guard on entry value of local guard_var[i] (or proven float if -2)
         *   4 = invariant local guard (container/type/managed-values) on guard_var[i]
         *   5 = int guard on entry value of attribute slot attr_slots[guard_var[i]]
         *   6 = float guard on entry value of attribute slot attr_slots[guard_var[i]]
         *   7 = _GUARD_DORV_NO_DICT on invariant receiver local guard_var[i]
         */
        int guard_kind[256] = {0};
        int guard_var[256];
        for (int i = 0; i < 256; i++) {
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
                case _GUARD_GLOBALS_VERSION:
                case _GUARD_BUILTINS_IS_CANONICAL:
                case _CHECK_FUNCTION_VERSION_INLINE:
                case _CHECK_STACK_SPACE_OPERAND:
                case _CHECK_RECURSION_REMAINING:
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
                    if (sp >= 64) { body_valid = false; break; }
                    stk_type[sp] = VAL_INT;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    stk_attr[sp] = -1;
                    stk_attr_entry[sp] = false;
                    sp++;
                    break;
                case _ITER_NEXT_LIST:
                case _ITER_NEXT_LIST_TIER_TWO:
                case _ITER_NEXT_TUPLE:
                    if (sp >= 64) { body_valid = false; break; }
                    stk_type[sp] = VAL_ITER_ITEM;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    stk_attr[sp] = -1;
                    stk_attr_entry[sp] = false;
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
                    if (sp >= 64) { body_valid = false; break; }
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
                    stk_attr[sp] = -1;
                    stk_attr_entry[sp] = false;
                    sp++;
                    break;
                }
                case _LOAD_SMALL_INT:
                    if (sp >= 64) { body_valid = false; break; }
                    stk_type[sp] = VAL_INT;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    stk_attr[sp] = -1;
                    stk_attr_entry[sp] = false;
                    sp++;
                    break;
                case _LOAD_CONST_INLINE:
                case _LOAD_CONST_INLINE_BORROW: {
                    if (sp >= 64) { body_valid = false; break; }
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
                    stk_attr[sp] = -1;
                    stk_attr_entry[sp] = false;
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
                    else if (attr_tracking_valid && stk_attr_entry[idx] && stk_attr[idx] >= 0) {
                        guard_kind[i] = 5;
                        guard_var[i] = stk_attr[idx];
                        attr_slots[stk_attr[idx]].entry_guarded = true;
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
                    if (stk_attr[idx] >= 0) {
                        int a = stk_attr[idx];
                        attr_slots[a].cur_type = VAL_INT;
                        for (int s = 0; s < sp; s++) {
                            if (stk_attr[s] == a) {
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
                    else if (attr_tracking_valid && stk_attr_entry[idx] && stk_attr[idx] >= 0) {
                        guard_kind[i] = 6;
                        guard_var[i] = stk_attr[idx];
                        attr_slots[stk_attr[idx]].entry_guarded = true;
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
                    if (stk_attr[idx] >= 0) {
                        int a = stk_attr[idx];
                        attr_slots[a].cur_type = VAL_FLOAT;
                        for (int s = 0; s < sp; s++) {
                            if (stk_attr[s] == a) {
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
                case _GUARD_NOS_TYPE:
                case _GUARD_NOS_DICT_SUBSCRIPT:
                    if (sp < 2) { body_valid = false; break; }
                    if (stk_entry[sp - 2] && stk_var[sp - 2] >= 0 && !written[stk_var[sp - 2]]) {
                        guard_kind[i] = 4;
                        guard_var[i] = stk_var[sp - 2];
                    }
                    break;
                case _LOCK_OBJECT:
                    if (sp < 1) { body_valid = false; break; }
                    break;
                case _GUARD_DORV_NO_DICT:
                    if (sp < 1) { body_valid = false; break; }
                    if (stk_entry[sp - 1] && stk_var[sp - 1] >= 0 && !written[stk_var[sp - 1]]) {
                        guard_kind[i] = 7;
                        guard_var[i] = stk_var[sp - 1];
                    }
                    break;
                case _GUARD_BINARY_OP_EXTEND:
                case _GUARD_BINARY_OP_EXTEND_LHS:
                case _GUARD_BINARY_OP_EXTEND_RHS:
                    if (sp < 2) { body_valid = false; break; }
                    break;
                case _BINARY_OP_SUBSCR_LIST_INT:
                case _BINARY_OP_SUBSCR_TUPLE_INT:
                case _BINARY_OP_SUBSCR_DICT_KNOWN_HASH:
                case _BINARY_OP_SUBSCR_DICT:
                    if (sp < 2 || sp >= 64) { body_valid = false; break; }
                    stk_type[sp - 2] = VAL_UNKNOWN;
                    stk_var[sp - 2] = -1;
                    stk_entry[sp - 2] = false;
                    stk_attr[sp - 2] = -1;
                    stk_attr_entry[sp - 2] = false;
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_attr[sp - 1] = -1;
                    stk_attr_entry[sp - 1] = false;
                    stk_type[sp] = VAL_UNKNOWN;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    stk_attr[sp] = -1;
                    stk_attr_entry[sp] = false;
                    sp++;
                    break;
                case _LOAD_ATTR_INSTANCE_VALUE:
                case _LOAD_ATTR_SLOT: {
                    if (sp < 1 || sp >= 64) { body_valid = false; break; }
                    bool is_slot = (op == _LOAD_ATTR_SLOT);
                    uint64_t raw_offset = is_slot ? (inst->operand0 & 0x1FFF) : inst->operand0;
                    int owner_v = stk_var[sp - 1];
                    int a_idx = -1;
                    if (attr_tracking_valid && owner_v >= 0 && !written[owner_v] && stk_entry[sp - 1]) {
                        if (attr_owner_var < 0) {
                            attr_owner_var = owner_v;
                        }
                        else if (attr_owner_var != owner_v && !(is_slot && (inst->operand0 >> 13) != 0)) {
                            attr_tracking_valid = false;
                        }
                        if (attr_tracking_valid) {
                            for (int a = 0; a < num_attr_slots; a++) {
                                if (attr_slots[a].owner_var == owner_v &&
                                    attr_slots[a].offset == raw_offset &&
                                    attr_slots[a].is_slot == is_slot)
                                {
                                    a_idx = a;
                                    break;
                                }
                            }
                            if (a_idx < 0 && num_attr_slots < 32) {
                                uint16_t strict_tag = is_slot ? (uint16_t)(inst->operand0 >> 13) : 0;
                                a_idx = num_attr_slots++;
                                attr_slots[a_idx].owner_var = owner_v;
                                attr_slots[a_idx].offset = raw_offset;
                                attr_slots[a_idx].is_slot = is_slot;
                                attr_slots[a_idx].cur_type = (strict_tag == 1) ? VAL_INT :
                                                             (strict_tag == 2) ? VAL_FLOAT : VAL_UNKNOWN;
                                attr_slots[a_idx].seen_write = false;
                                attr_slots[a_idx].entry_guarded = (strict_tag == 1 || strict_tag == 2);
                                attr_slots[a_idx].all_stores_int = true;
                                attr_slots[a_idx].all_stores_float = true;
                                attr_slots[a_idx].load_count = 0;
                                attr_slots[a_idx].store_count = 0;
                            }
                            else if (a_idx < 0) {
                                attr_tracking_valid = false;
                            }
                        }
                    }
                    else if (!(is_slot && (inst->operand0 >> 13) != 0)) {
                        attr_tracking_valid = false;
                    }
                    if (a_idx >= 0 && attr_tracking_valid) {
                        attr_slots[a_idx].load_count++;
                        stk_type[sp - 1] = attr_slots[a_idx].cur_type;
                        stk_var[sp - 1] = -1;
                        stk_entry[sp - 1] = false;
                        stk_attr[sp - 1] = a_idx;
                        stk_attr_entry[sp - 1] = !attr_slots[a_idx].seen_write;
                    }
                    else {
                        uint16_t strict_tag = is_slot ? (uint16_t)(inst->operand0 >> 13) : 0;
                        stk_type[sp - 1] = (strict_tag == 1) ? VAL_INT :
                                           (strict_tag == 2) ? VAL_FLOAT : VAL_UNKNOWN;
                        stk_var[sp - 1] = -1;
                        stk_entry[sp - 1] = false;
                        stk_attr[sp - 1] = -1;
                        stk_attr_entry[sp - 1] = false;
                    }
                    stk_type[sp] = VAL_UNKNOWN;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    stk_attr[sp] = -1;
                    stk_attr_entry[sp] = false;
                    sp++;
                    break;
                }
                case _LOAD_ATTR_SLOT_NO_POP:
                    if (sp < 1) { body_valid = false; break; }
                    stk_type[sp - 1] = (inst->oparg == 1) ? VAL_INT : VAL_FLOAT;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_attr[sp - 1] = -1;
                    stk_attr_entry[sp - 1] = false;
                    break;
                case _STORE_SUBSCR_LIST_INT:
                    if (sp < 3) { body_valid = false; break; }
                    stk_type[sp - 3] = VAL_UNKNOWN;
                    stk_var[sp - 3] = -1;
                    stk_entry[sp - 3] = false;
                    stk_attr[sp - 3] = -1;
                    stk_attr_entry[sp - 3] = false;
                    stk_type[sp - 2] = VAL_UNKNOWN;
                    stk_var[sp - 2] = -1;
                    stk_entry[sp - 2] = false;
                    stk_attr[sp - 2] = -1;
                    stk_attr_entry[sp - 2] = false;
                    sp--;
                    break;
                case _STORE_ATTR_INSTANCE_VALUE:
                case _STORE_ATTR_SLOT: {
                    if (sp < 2) { body_valid = false; break; }
                    bool is_slot = (op == _STORE_ATTR_SLOT);
                    uint64_t raw_offset = is_slot ? (inst->operand0 & 0x1FFF) : inst->operand0;
                    int owner_v = stk_var[sp - 1];
                    int val_type = stk_type[sp - 2];
                    if (attr_tracking_valid && owner_v >= 0 && !written[owner_v] && stk_entry[sp - 1]) {
                        if (attr_owner_var < 0) {
                            attr_owner_var = owner_v;
                        }
                        else if (attr_owner_var != owner_v && !(is_slot && (inst->operand0 >> 13) != 0)) {
                            attr_tracking_valid = false;
                        }
                        int a_idx = -1;
                        if (attr_tracking_valid) {
                            for (int a = 0; a < num_attr_slots; a++) {
                                if (attr_slots[a].owner_var == owner_v &&
                                    attr_slots[a].offset == raw_offset &&
                                    attr_slots[a].is_slot == is_slot)
                                {
                                    a_idx = a;
                                    break;
                                }
                            }
                            if (a_idx < 0 && num_attr_slots < 32) {
                                uint16_t strict_tag = is_slot ? (uint16_t)(inst->operand0 >> 13) : 0;
                                a_idx = num_attr_slots++;
                                attr_slots[a_idx].owner_var = owner_v;
                                attr_slots[a_idx].offset = raw_offset;
                                attr_slots[a_idx].is_slot = is_slot;
                                attr_slots[a_idx].cur_type = (strict_tag == 1) ? VAL_INT :
                                                             (strict_tag == 2) ? VAL_FLOAT : VAL_UNKNOWN;
                                attr_slots[a_idx].seen_write = false;
                                attr_slots[a_idx].entry_guarded = (strict_tag == 1 || strict_tag == 2);
                                attr_slots[a_idx].all_stores_int = true;
                                attr_slots[a_idx].all_stores_float = true;
                                attr_slots[a_idx].load_count = 0;
                                attr_slots[a_idx].store_count = 0;
                            }
                            else if (a_idx < 0) {
                                attr_tracking_valid = false;
                            }
                        }
                        if (a_idx >= 0 && attr_tracking_valid) {
                            for (int s = 0; s < sp; s++) {
                                if (stk_attr[s] == a_idx) {
                                    stk_attr[s] = -1;
                                    stk_attr_entry[s] = false;
                                }
                            }
                            attr_slots[a_idx].seen_write = true;
                            attr_slots[a_idx].store_count++;
                            attr_slots[a_idx].cur_type = val_type;
                            if (val_type != VAL_INT) {
                                attr_slots[a_idx].all_stores_int = false;
                            }
                            if (val_type != VAL_FLOAT) {
                                attr_slots[a_idx].all_stores_float = false;
                            }
                        }
                    }
                    else if (!(is_slot && (inst->operand0 >> 13) != 0 && (val_type == VAL_INT || val_type == VAL_FLOAT))) {
                        attr_tracking_valid = false;
                    }
                    stk_type[sp - 2] = VAL_UNKNOWN;
                    stk_var[sp - 2] = -1;
                    stk_entry[sp - 2] = false;
                    stk_attr[sp - 2] = -1;
                    stk_attr_entry[sp - 2] = false;
                    sp--;
                    break;
                }
                case _BINARY_OP_EXTEND: {
                    if (sp < 2 || sp >= 64) { body_valid = false; break; }
                    _PyBinaryOpSpecializationDescr *d =
                        (_PyBinaryOpSpecializationDescr *)inst->operand0;
                    int res_type = VAL_UNKNOWN;
                    if (d != NULL && d->result_type == &PyFloat_Type) {
                        res_type = VAL_FLOAT;
                    }
                    else if (d != NULL && d->result_type == &PyLong_Type &&
                             stk_type[sp - 2] == VAL_INT && stk_type[sp - 1] == VAL_INT &&
                             (inst->oparg == NB_AND || inst->oparg == NB_OR ||
                              inst->oparg == NB_XOR || inst->oparg == NB_RSHIFT ||
                              inst->oparg == NB_FLOOR_DIVIDE || inst->oparg == NB_REMAINDER ||
                              inst->oparg == NB_INPLACE_AND || inst->oparg == NB_INPLACE_OR ||
                              inst->oparg == NB_INPLACE_XOR || inst->oparg == NB_INPLACE_RSHIFT ||
                              inst->oparg == NB_INPLACE_FLOOR_DIVIDE ||
                              inst->oparg == NB_INPLACE_REMAINDER))
                    {
                        res_type = VAL_INT;
                    }
                    stk_type[sp - 2] = res_type;
                    stk_var[sp - 2] = -1;
                    stk_entry[sp - 2] = false;
                    stk_attr[sp - 2] = -1;
                    stk_attr_entry[sp - 2] = false;
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_attr[sp - 1] = -1;
                    stk_attr_entry[sp - 1] = false;
                    stk_type[sp] = VAL_UNKNOWN;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    stk_attr[sp] = -1;
                    stk_attr_entry[sp] = false;
                    sp++;
                    break;
                }
                case _BINARY_OP_BITWISE_INT:
                case _BINARY_OP_ADD_INT:
                case _BINARY_OP_ADD_INT_INPLACE:
                case _BINARY_OP_ADD_INT_INPLACE_RIGHT:
                case _BINARY_OP_SUBTRACT_INT:
                case _BINARY_OP_SUBTRACT_INT_INPLACE:
                case _BINARY_OP_SUBTRACT_INT_INPLACE_RIGHT:
                case _BINARY_OP_MULTIPLY_INT:
                case _BINARY_OP_MULTIPLY_INT_INPLACE:
                case _BINARY_OP_MULTIPLY_INT_INPLACE_RIGHT:
                    if (sp < 2 || sp >= 64) { body_valid = false; break; }
                    if (stk_attr_entry[sp - 2] && stk_attr[sp - 2] >= 0) {
                        attr_slots[stk_attr[sp - 2]].entry_guarded = true;
                        attr_slots[stk_attr[sp - 2]].cur_type = VAL_INT;
                    }
                    if (stk_attr_entry[sp - 1] && stk_attr[sp - 1] >= 0) {
                        attr_slots[stk_attr[sp - 1]].entry_guarded = true;
                        attr_slots[stk_attr[sp - 1]].cur_type = VAL_INT;
                    }
                    stk_type[sp - 2] = VAL_INT;
                    stk_var[sp - 2] = -1;
                    stk_entry[sp - 2] = false;
                    stk_attr[sp - 2] = -1;
                    stk_attr_entry[sp - 2] = false;
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_attr[sp - 1] = -1;
                    stk_attr_entry[sp - 1] = false;
                    stk_type[sp] = VAL_UNKNOWN;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    stk_attr[sp] = -1;
                    stk_attr_entry[sp] = false;
                    sp++;
                    break;
                case _BINARY_OP_ARITH_FLOAT_INT:
                    if (sp < 2 || sp >= 64) { body_valid = false; break; }
                    stk_type[sp - 2] = VAL_FLOAT;
                    stk_var[sp - 2] = -1;
                    stk_entry[sp - 2] = false;
                    stk_attr[sp - 2] = -1;
                    stk_attr_entry[sp - 2] = false;
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_attr[sp - 1] = -1;
                    stk_attr_entry[sp - 1] = false;
                    stk_type[sp] = VAL_UNKNOWN;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    stk_attr[sp] = -1;
                    stk_attr_entry[sp] = false;
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
                    if (sp < 2 || sp >= 64) { body_valid = false; break; }
                    if (stk_attr_entry[sp - 2] && stk_attr[sp - 2] >= 0) {
                        attr_slots[stk_attr[sp - 2]].entry_guarded = true;
                        attr_slots[stk_attr[sp - 2]].cur_type = VAL_FLOAT;
                    }
                    if (stk_attr_entry[sp - 1] && stk_attr[sp - 1] >= 0) {
                        attr_slots[stk_attr[sp - 1]].entry_guarded = true;
                        attr_slots[stk_attr[sp - 1]].cur_type = VAL_FLOAT;
                    }
                    stk_type[sp - 2] = VAL_FLOAT;
                    stk_var[sp - 2] = -1;
                    stk_entry[sp - 2] = false;
                    stk_attr[sp - 2] = -1;
                    stk_attr_entry[sp - 2] = false;
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_attr[sp - 1] = -1;
                    stk_attr_entry[sp - 1] = false;
                    stk_type[sp] = VAL_UNKNOWN;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    stk_attr[sp] = -1;
                    stk_attr_entry[sp] = false;
                    sp++;
                    break;
                case _BINARY_OP_FMA_FLOAT_INPLACE:
                    if (sp < 3 || sp >= 64) { body_valid = false; break; }
                    stk_type[sp - 3] = VAL_FLOAT;
                    stk_var[sp - 3] = -1;
                    stk_entry[sp - 3] = false;
                    stk_attr[sp - 3] = -1;
                    stk_attr_entry[sp - 3] = false;
                    stk_type[sp - 2] = VAL_UNKNOWN;
                    stk_var[sp - 2] = -1;
                    stk_entry[sp - 2] = false;
                    stk_attr[sp - 2] = -1;
                    stk_attr_entry[sp - 2] = false;
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_attr[sp - 1] = -1;
                    stk_attr_entry[sp - 1] = false;
                    break;
                case _UNARY_NEGATIVE_FLOAT_INPLACE:
                    if (sp < 1 || sp >= 64) { body_valid = false; break; }
                    stk_type[sp - 1] = VAL_FLOAT;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_attr[sp - 1] = -1;
                    stk_attr_entry[sp - 1] = false;
                    break;
                case _COMPARE_OP_INT:
                case _COMPARE_OP_FLOAT:
                    if (sp < 2 || sp >= 64) { body_valid = false; break; }
                    stk_type[sp - 2] = VAL_UNKNOWN;
                    stk_var[sp - 2] = -1;
                    stk_entry[sp - 2] = false;
                    stk_attr[sp - 2] = -1;
                    stk_attr_entry[sp - 2] = false;
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_attr[sp - 1] = -1;
                    stk_attr_entry[sp - 1] = false;
                    stk_type[sp] = VAL_UNKNOWN;
                    stk_var[sp] = -1;
                    stk_entry[sp] = false;
                    stk_attr[sp] = -1;
                    stk_attr_entry[sp] = false;
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
                case _LIST_APPEND:
                case _SET_ADD:
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
                    if (stk_var[sp - 1] >= 0) {
                        var_copied[stk_var[sp - 1]] = true;
                    }
                    stk_type[sp - 1] = VAL_UNKNOWN;
                    stk_var[sp - 1] = -1;
                    stk_entry[sp - 1] = false;
                    stk_attr[sp - 1] = -1;
                    stk_attr_entry[sp - 1] = false;
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

        bool all_attr_stores_safe = attr_tracking_valid;
        bool all_attr_stores_entry_guarded = attr_tracking_valid;
        for (int a = 0; a < num_attr_slots; a++) {
            if (attr_slots[a].seen_write) {
                if (!attr_slots[a].all_stores_int && !attr_slots[a].all_stores_float) {
                    all_attr_stores_safe = false;
                    all_attr_stores_entry_guarded = false;
                }
                if (!attr_slots[a].entry_guarded) {
                    all_attr_stores_entry_guarded = false;
                }
            }
        }

        bool hoist_inst[256] = {0};
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
                else if (gk == 5 && all_attr_stores_safe) {
                    if (gv >= 0 && gv < num_attr_slots &&
                        (!attr_slots[gv].seen_write || attr_slots[gv].all_stores_int))
                    {
                        hoist_inst[i] = true;
                        hoist_type_guard_count++;
                    }
                }
                else if (gk == 6 && all_attr_stores_safe) {
                    if (gv >= 0 && gv < num_attr_slots &&
                        (!attr_slots[gv].seen_write || attr_slots[gv].all_stores_float))
                    {
                        hoist_inst[i] = true;
                        hoist_type_guard_count++;
                    }
                }
                else if (gk == 7 && all_attr_stores_safe) {
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

        if (body_valid && (hoist_type_guard_count > 0 || (has_iter_check && pop_top_count > 0))) {
            _PyUOpInstruction make_warm_inst = output[1];
            _PyUOpInstruction jump_inst = output[length - 1];
            int out_idx = 1;
            bool seen_pre_dorv_guard = false;
            /* Emit Pre-Header (Iteration 1 with all type and iterator guards) */
            for (int i = 0; i < b_len; i++) {
                uint16_t op = body[i].opcode;
                if (op == _CHECK_PERIODIC ||
                    op == _CHECK_PERIODIC_IF_NOT_YIELD_FROM)
                {
                    continue;
                }
                if (op == _GUARD_DORV_NO_DICT &&
                    guard_kind[i] == 7 &&
                    all_attr_stores_entry_guarded)
                {
                    if (seen_pre_dorv_guard) {
                        continue;
                    }
                    seen_pre_dorv_guard = true;
                }
                output[out_idx++] = body[i];
            }
            /* Loop Header */
            output[out_idx++] = make_warm_inst;
            /* Build Steady-State Loop Body with hoisted guards removed */
            _PyUOpInstruction sbody[256];
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
            /* In-place ownership transfer / mutation for int/float accumulators
             * in steady-state loop body (skipping _NOP and _SET_IP instructions):
             * - For float accumulators: keep _LOAD_FAST_BORROW v, convert
             *   _BINARY_OP_{ADD,SUBTRACT,MULTIPLY}_FLOAT to _FLOAT_INPLACE, and
             *   replace _SWAP_FAST v + _POP_TOP_FLOAT with _POP_TOP_NOP + _NOP so
             *   locals[v]->ob_fval is mutated directly in place with zero refcount
             *   ops, zero local stores, and zero _SET_IP instructions.
             * - For int accumulators: use _LOAD_FAST_AND_CLEAR v so
             *   _BINARY_OP_{ADD,SUBTRACT,MULTIPLY}_INT mutates locals[v] in place. */
            #define NEXT_SBODY_IDX(idx) \
                ({ int _idx = (idx); \
                   while (_idx < s_len && \
                          (sbody[_idx].opcode == _NOP || \
                           sbody[_idx].opcode == _SET_IP || \
                           sbody[_idx].opcode == _CHECK_VALIDITY)) { \
                       _idx++; \
                   } \
                   _idx; })
            for (int i = 0; i < s_len; i++) {
                uint16_t lop = sbody[i].opcode;
                int v = -1;
                if (lop == _LOAD_FAST_BORROW) {
                    v = (int)sbody[i].oparg;
                }
                else if (lop >= _LOAD_FAST_BORROW_0 && lop <= _LOAD_FAST_BORROW_7) {
                    v = (int)(lop - _LOAD_FAST_BORROW_0);
                }
                if (v < 0 || v >= 64 ||
                    !written[v] ||
                    var_copied[v] ||
                    load_count[v] < 1 ||
                    store_count[v] != 1)
                {
                    continue;
                }
                if (load_count[v] > 1 && i > 0) {
                    int prev = i - 1;
                    while (prev >= 0 &&
                           (sbody[prev].opcode == _NOP ||
                            sbody[prev].opcode == _SET_IP ||
                            sbody[prev].opcode == _CHECK_VALIDITY ||
                            sbody[prev].opcode == _GUARD_TYPE_VERSION ||
                            sbody[prev].opcode == _CHECK_MANAGED_OBJECT_HAS_VALUES)) {
                        prev--;
                    }
                    if (prev >= 0) {
                        uint16_t pop = sbody[prev].opcode;
                        if (pop != _POP_TOP_INT &&
                            pop != _POP_TOP_FLOAT &&
                            pop != _POP_TOP_NOP &&
                            pop != _GUARD_NOT_EXHAUSTED_RANGE &&
                            pop != _GUARD_NOT_EXHAUSTED_LIST &&
                            pop != _GUARD_NOT_EXHAUSTED_TUPLE)
                        {
                            continue;
                        }
                    }
                }
                int i_rhs = NEXT_SBODY_IDX(i + 1);
                if (i_rhs >= s_len) {
                    continue;
                }
                uint16_t rop = sbody[i_rhs].opcode;
                int rhs_v = -1;
                if (rop == _LOAD_FAST_BORROW) {
                    rhs_v = (int)sbody[i_rhs].oparg;
                }
                else if (rop >= _LOAD_FAST_BORROW_0 && rop <= _LOAD_FAST_BORROW_7) {
                    rhs_v = (int)(rop - _LOAD_FAST_BORROW_0);
                }
                int i_binop = NEXT_SBODY_IDX(i_rhs + 1);
                if (rhs_v >= 0 && rhs_v != v && i_binop < s_len &&
                    (sbody[i_binop].opcode == _LOAD_ATTR_SLOT ||
                     sbody[i_binop].opcode == _LOAD_ATTR_INSTANCE_VALUE))
                {
                    int i_attr_pop = NEXT_SBODY_IDX(i_binop + 1);
                    if (i_attr_pop < s_len && sbody[i_attr_pop].opcode == _POP_TOP_NOP) {
                        i_binop = NEXT_SBODY_IDX(i_attr_pop + 1);
                    }
                }
                int i_pop1 = NEXT_SBODY_IDX(i_binop + 1);
                int i_pop2 = NEXT_SBODY_IDX(i_pop1 + 1);
                if (i_pop2 >= s_len) {
                    continue;
                }
                if (all_stores_float[v]) {
                    bool rhs_ok = (rhs_v >= 0 && rhs_v != v) ||
                                  rop == _LOAD_CONST_INLINE_BORROW;
                    uint16_t binop = sbody[i_binop].opcode;
                    uint16_t inplace_binop = 0;
                    if (binop == _BINARY_OP_ADD_FLOAT || binop == _BINARY_OP_ADD_FLOAT_INPLACE) {
                        inplace_binop = _BINARY_OP_ADD_FLOAT_INPLACE;
                    }
                    else if (binop == _BINARY_OP_SUBTRACT_FLOAT || binop == _BINARY_OP_SUBTRACT_FLOAT_INPLACE) {
                        inplace_binop = _BINARY_OP_SUBTRACT_FLOAT_INPLACE;
                    }
                    else if (binop == _BINARY_OP_MULTIPLY_FLOAT || binop == _BINARY_OP_MULTIPLY_FLOAT_INPLACE) {
                        inplace_binop = _BINARY_OP_MULTIPLY_FLOAT_INPLACE;
                    }
                    bool pops_ok =
                        (sbody[i_pop1].opcode == _POP_TOP_NOP ||
                         sbody[i_pop1].opcode == _POP_TOP_FLOAT) &&
                        (sbody[i_pop2].opcode == _POP_TOP_NOP ||
                         sbody[i_pop2].opcode == _POP_TOP_FLOAT);
                    if (!rhs_ok || inplace_binop == 0 || !pops_ok) {
                        continue;
                    }
                    int k = NEXT_SBODY_IDX(i_pop2 + 1);
                    while (k < s_len) {
                        uint16_t krop = sbody[k].opcode;
                        int krhs_v = -1;
                        if (krop == _LOAD_FAST_BORROW) {
                            krhs_v = (int)sbody[k].oparg;
                        }
                        else if (krop >= _LOAD_FAST_BORROW_0 && krop <= _LOAD_FAST_BORROW_7) {
                            krhs_v = (int)(krop - _LOAD_FAST_BORROW_0);
                        }
                        int k_binop = NEXT_SBODY_IDX(k + 1);
                        if (krhs_v >= 0 && krhs_v != v && k_binop < s_len &&
                            (sbody[k_binop].opcode == _LOAD_ATTR_SLOT ||
                             sbody[k_binop].opcode == _LOAD_ATTR_INSTANCE_VALUE))
                        {
                            int k_attr_pop = NEXT_SBODY_IDX(k_binop + 1);
                            if (k_attr_pop < s_len && sbody[k_attr_pop].opcode == _POP_TOP_NOP) {
                                k_binop = NEXT_SBODY_IDX(k_attr_pop + 1);
                            }
                        }
                        int k_pop1 = NEXT_SBODY_IDX(k_binop + 1);
                        int k_pop2 = NEXT_SBODY_IDX(k_pop1 + 1);
                        if (k_pop2 >= s_len) {
                            break;
                        }
                        if (((krhs_v >= 0 && krhs_v != v) ||
                             krop == _LOAD_CONST_INLINE_BORROW) &&
                            (sbody[k_binop].opcode == _BINARY_OP_ADD_FLOAT_INPLACE ||
                             sbody[k_binop].opcode == _BINARY_OP_SUBTRACT_FLOAT_INPLACE ||
                             sbody[k_binop].opcode == _BINARY_OP_MULTIPLY_FLOAT_INPLACE ||
                             sbody[k_binop].opcode == _BINARY_OP_ADD_FLOAT ||
                             sbody[k_binop].opcode == _BINARY_OP_SUBTRACT_FLOAT ||
                             sbody[k_binop].opcode == _BINARY_OP_MULTIPLY_FLOAT) &&
                            (sbody[k_pop1].opcode == _POP_TOP_NOP ||
                             sbody[k_pop1].opcode == _POP_TOP_FLOAT) &&
                            (sbody[k_pop2].opcode == _POP_TOP_NOP ||
                             sbody[k_pop2].opcode == _POP_TOP_FLOAT))
                        {
                            k = NEXT_SBODY_IDX(k_pop2 + 1);
                        }
                        else {
                            break;
                        }
                    }
                    int k_pop = NEXT_SBODY_IDX(k + 1);
                    if (k_pop < s_len) {
                        uint16_t swop = sbody[k].opcode;
                        int sw_v = (swop == _SWAP_FAST) ? (int)sbody[k].oparg :
                                   (swop >= _SWAP_FAST_0 && swop <= _SWAP_FAST_7) ? (int)(swop - _SWAP_FAST_0) : -1;
                        if (sw_v == v && sbody[k_pop].opcode == _POP_TOP_FLOAT) {
                            sbody[i_binop].opcode = inplace_binop;
                            sbody[i_pop2].opcode = _POP_TOP_NOP;
                            int m = NEXT_SBODY_IDX(i_pop2 + 1);
                            while (m < k) {
                                int m_binop = NEXT_SBODY_IDX(m + 1);
                                if (m_binop < k &&
                                    (sbody[m_binop].opcode == _LOAD_ATTR_SLOT ||
                                     sbody[m_binop].opcode == _LOAD_ATTR_INSTANCE_VALUE))
                                {
                                    int m_attr_pop = NEXT_SBODY_IDX(m_binop + 1);
                                    m_binop = NEXT_SBODY_IDX(m_attr_pop + 1);
                                }
                                int m_pop1 = NEXT_SBODY_IDX(m_binop + 1);
                                int m_pop2 = NEXT_SBODY_IDX(m_pop1 + 1);
                                if (sbody[m_binop].opcode == _BINARY_OP_ADD_FLOAT) {
                                    sbody[m_binop].opcode = _BINARY_OP_ADD_FLOAT_INPLACE;
                                }
                                else if (sbody[m_binop].opcode == _BINARY_OP_SUBTRACT_FLOAT) {
                                    sbody[m_binop].opcode = _BINARY_OP_SUBTRACT_FLOAT_INPLACE;
                                }
                                else if (sbody[m_binop].opcode == _BINARY_OP_MULTIPLY_FLOAT) {
                                    sbody[m_binop].opcode = _BINARY_OP_MULTIPLY_FLOAT_INPLACE;
                                }
                                sbody[m_pop2].opcode = _POP_TOP_NOP;
                                m = NEXT_SBODY_IDX(m_pop2 + 1);
                            }
                            sbody[k].opcode = _POP_TOP_NOP;
                            sbody[k].oparg = 0;
                            sbody[k].operand0 = 0;
                            sbody[k_pop].opcode = _NOP;
                        }
                    }
                }
                else if (all_stores_int[v]) {
                    bool rhs_ok = (rhs_v >= 0 && rhs_v != v) ||
                                   rop == _LOAD_CONST_INLINE_BORROW ||
                                   rop == _LOAD_SMALL_INT;
                    uint16_t binop = sbody[i_binop].opcode;
                    bool binop_ok =
                        binop == _BINARY_OP_ADD_INT ||
                        binop == _BINARY_OP_SUBTRACT_INT ||
                        binop == _BINARY_OP_MULTIPLY_INT;
                    bool pops_ok =
                        (sbody[i_pop1].opcode == _POP_TOP_NOP ||
                         sbody[i_pop1].opcode == _POP_TOP_INT) &&
                        sbody[i_pop2].opcode == _POP_TOP_NOP;
                    if (!rhs_ok || !binop_ok || !pops_ok) {
                        continue;
                    }
                    int k = NEXT_SBODY_IDX(i_pop2 + 1);
                    while (k < s_len) {
                        int k_binop = NEXT_SBODY_IDX(k + 1);
                        int k_pop1 = NEXT_SBODY_IDX(k_binop + 1);
                        int k_pop2 = NEXT_SBODY_IDX(k_pop1 + 1);
                        if (k_pop2 >= s_len) {
                            break;
                        }
                        uint16_t krop = sbody[k].opcode;
                        int krhs_v = -1;
                        if (krop == _LOAD_FAST_BORROW) {
                            krhs_v = (int)sbody[k].oparg;
                        }
                        else if (krop >= _LOAD_FAST_BORROW_0 && krop <= _LOAD_FAST_BORROW_7) {
                            krhs_v = (int)(krop - _LOAD_FAST_BORROW_0);
                        }
                        if (((krhs_v >= 0 && krhs_v != v) ||
                             krop == _LOAD_CONST_INLINE_BORROW ||
                             krop == _LOAD_SMALL_INT) &&
                            (sbody[k_binop].opcode == _BINARY_OP_ADD_INT_INPLACE ||
                             sbody[k_binop].opcode == _BINARY_OP_SUBTRACT_INT_INPLACE ||
                             sbody[k_binop].opcode == _BINARY_OP_MULTIPLY_INT_INPLACE) &&
                            (sbody[k_pop1].opcode == _POP_TOP_NOP ||
                             sbody[k_pop1].opcode == _POP_TOP_INT) &&
                            (sbody[k_pop2].opcode == _POP_TOP_NOP ||
                             sbody[k_pop2].opcode == _POP_TOP_INT))
                        {
                            k = NEXT_SBODY_IDX(k_pop2 + 1);
                        }
                        else {
                            break;
                        }
                    }
                    int k_pop = NEXT_SBODY_IDX(k + 1);
                    if (k_pop < s_len) {
                        uint16_t swop = sbody[k].opcode;
                        int sw_v = (swop == _SWAP_FAST) ? (int)sbody[k].oparg :
                                   (swop >= _SWAP_FAST_0 && swop <= _SWAP_FAST_7) ? (int)(swop - _SWAP_FAST_0) : -1;
                        if (sw_v == v && sbody[k_pop].opcode == _POP_TOP_INT) {
                            sbody[i].opcode = _LOAD_FAST_AND_CLEAR;
                            sbody[i].oparg = v;
                            sbody[i_pop2].opcode = _POP_TOP_INT;
                            sbody[k_pop].opcode = _POP_TOP_NOP;
                        }
                    }
                }
            }
            /* In-place float attribute accumulator in steady-state loop body:
             * after Iteration 1 (Pre-Header) executes _BINARY_OP_{ADD,SUBTRACT,MULTIPLY}_FLOAT
             * and stores the newly allocated, uniquely-owned PyFloatObject into
             * (owner_var, offset), iterations 2..N can mutate that PyFloatObject's
             * ob_fval in place via _BINARY_OP_*_FLOAT_INPLACE and drop the redundant
             * _LOCK_OBJECT + _STORE_ATTR_* store-back. */
            if (all_attr_stores_safe && num_attr_slots > 0) {
                for (int i = 0; i < s_len; i++) {
                    uint16_t lop = sbody[i].opcode;
                    int v = -1;
                    if (lop == _LOAD_FAST || lop == _LOAD_FAST_BORROW) {
                        v = (int)sbody[i].oparg;
                    }
                    else if (lop >= _LOAD_FAST_0 && lop <= _LOAD_FAST_7) {
                        v = (int)(lop - _LOAD_FAST_0);
                    }
                    else if (lop >= _LOAD_FAST_BORROW_0 && lop <= _LOAD_FAST_BORROW_7) {
                        v = (int)(lop - _LOAD_FAST_BORROW_0);
                    }
                    if (v < 0 || v >= 64 || written[v]) {
                        continue;
                    }
                    int i_attr = NEXT_SBODY_IDX(i + 1);
                    int i_pop0 = NEXT_SBODY_IDX(i_attr + 1);
                    int i_rhs = NEXT_SBODY_IDX(i_pop0 + 1);
                    if (i_rhs >= s_len) {
                        continue;
                    }
                    uint16_t aop = sbody[i_attr].opcode;
                    if (aop != _LOAD_ATTR_INSTANCE_VALUE && aop != _LOAD_ATTR_SLOT) {
                        continue;
                    }
                    bool is_slot = (aop == _LOAD_ATTR_SLOT);
                    uint64_t offset = is_slot ? (sbody[i_attr].operand0 & 0x1FFF) : sbody[i_attr].operand0;
                    int a_idx = -1;
                    for (int a = 0; a < num_attr_slots; a++) {
                        if (attr_slots[a].owner_var == v &&
                            attr_slots[a].offset == offset &&
                            attr_slots[a].is_slot == is_slot)
                        {
                            a_idx = a;
                            break;
                        }
                    }
                    if (a_idx < 0 ||
                        !attr_slots[a_idx].all_stores_float ||
                        !attr_slots[a_idx].entry_guarded ||
                        attr_slots[a_idx].load_count < 1 ||
                        attr_slots[a_idx].store_count != 1 ||
                        sbody[i_pop0].opcode != _POP_TOP_NOP)
                    {
                        continue;
                    }
                    if (attr_slots[a_idx].load_count > 1 && i > 0) {
                        int prev = i - 1;
                        while (prev >= 0 &&
                               (sbody[prev].opcode == _NOP ||
                                sbody[prev].opcode == _SET_IP ||
                                sbody[prev].opcode == _CHECK_VALIDITY ||
                                sbody[prev].opcode == _GUARD_TYPE_VERSION ||
                                sbody[prev].opcode == _CHECK_MANAGED_OBJECT_HAS_VALUES)) {
                            prev--;
                        }
                        if (prev >= 0) {
                            uint16_t pop = sbody[prev].opcode;
                            if (pop != _POP_TOP_INT &&
                                pop != _POP_TOP_FLOAT &&
                                pop != _POP_TOP_NOP &&
                                pop != _GUARD_NOT_EXHAUSTED_RANGE &&
                                pop != _GUARD_NOT_EXHAUSTED_LIST &&
                                pop != _GUARD_NOT_EXHAUSTED_TUPLE)
                            {
                                continue;
                            }
                        }
                    }
                    uint16_t rop = sbody[i_rhs].opcode;
                    int rhs_v = -1;
                    if (rop == _LOAD_FAST_BORROW) {
                        rhs_v = (int)sbody[i_rhs].oparg;
                    }
                    else if (rop >= _LOAD_FAST_BORROW_0 && rop <= _LOAD_FAST_BORROW_7) {
                        rhs_v = (int)(rop - _LOAD_FAST_BORROW_0);
                    }
                    bool rhs_ok = (rop == _LOAD_CONST_INLINE_BORROW) ||
                                  (rhs_v >= 0 && rhs_v != v);
                    int i_binop = NEXT_SBODY_IDX(i_rhs + 1);
                    if (rhs_v >= 0 && !written[rhs_v] && i_binop < s_len &&
                        (sbody[i_binop].opcode == _LOAD_ATTR_SLOT ||
                         sbody[i_binop].opcode == _LOAD_ATTR_INSTANCE_VALUE))
                    {
                        bool rhs_slot = (sbody[i_binop].opcode == _LOAD_ATTR_SLOT);
                        uint64_t rhs_off = rhs_slot ? (sbody[i_binop].operand0 & 0x1FFF) : sbody[i_binop].operand0;
                        int i_rhs_pop = NEXT_SBODY_IDX(i_binop + 1);
                        if (i_rhs_pop < s_len && sbody[i_rhs_pop].opcode == _POP_TOP_NOP &&
                            (rhs_v != v || rhs_off != offset || rhs_slot != is_slot))
                        {
                            rhs_ok = true;
                            i_binop = NEXT_SBODY_IDX(i_rhs_pop + 1);
                        }
                        else {
                            rhs_ok = false;
                        }
                    }
                    int i_pop1 = NEXT_SBODY_IDX(i_binop + 1);
                    int i_pop2 = NEXT_SBODY_IDX(i_pop1 + 1);
                    if (i_pop2 >= s_len) {
                        continue;
                    }
                    uint16_t binop = sbody[i_binop].opcode;
                    uint16_t inplace_binop = 0;
                    if (binop == _BINARY_OP_ADD_FLOAT) {
                        inplace_binop = _BINARY_OP_ADD_FLOAT_INPLACE;
                    }
                    else if (binop == _BINARY_OP_SUBTRACT_FLOAT) {
                        inplace_binop = _BINARY_OP_SUBTRACT_FLOAT_INPLACE;
                    }
                    else if (binop == _BINARY_OP_MULTIPLY_FLOAT) {
                        inplace_binop = _BINARY_OP_MULTIPLY_FLOAT_INPLACE;
                    }
                    bool pops_ok =
                        (sbody[i_pop1].opcode == _POP_TOP_NOP ||
                         sbody[i_pop1].opcode == _POP_TOP_FLOAT) &&
                        sbody[i_pop2].opcode == _POP_TOP_FLOAT;
                    if (!rhs_ok || inplace_binop == 0 || !pops_ok) {
                        continue;
                    }
                    int k = NEXT_SBODY_IDX(i_pop2 + 1);
                    while (k < s_len) {
                        uint16_t krop = sbody[k].opcode;
                        int krhs_v = -1;
                        if (krop == _LOAD_FAST_BORROW) {
                            krhs_v = (int)sbody[k].oparg;
                        }
                        else if (krop >= _LOAD_FAST_BORROW_0 && krop <= _LOAD_FAST_BORROW_7) {
                            krhs_v = (int)(krop - _LOAD_FAST_BORROW_0);
                        }
                        bool krhs_ok = (krop == _LOAD_CONST_INLINE_BORROW) ||
                                       (krhs_v >= 0 && krhs_v != v);
                        int k_binop = NEXT_SBODY_IDX(k + 1);
                        if (krhs_v >= 0 && !written[krhs_v] && k_binop < s_len &&
                            (sbody[k_binop].opcode == _LOAD_ATTR_SLOT ||
                             sbody[k_binop].opcode == _LOAD_ATTR_INSTANCE_VALUE))
                        {
                            bool k_slot = (sbody[k_binop].opcode == _LOAD_ATTR_SLOT);
                            uint64_t k_off = k_slot ? (sbody[k_binop].operand0 & 0x1FFF) : sbody[k_binop].operand0;
                            int k_rhs_pop = NEXT_SBODY_IDX(k_binop + 1);
                            if (k_rhs_pop < s_len && sbody[k_rhs_pop].opcode == _POP_TOP_NOP &&
                                (krhs_v != v || k_off != offset || k_slot != is_slot))
                            {
                                krhs_ok = true;
                                k_binop = NEXT_SBODY_IDX(k_rhs_pop + 1);
                            }
                            else {
                                krhs_ok = false;
                            }
                        }
                        int k_pop1 = NEXT_SBODY_IDX(k_binop + 1);
                        int k_pop2 = NEXT_SBODY_IDX(k_pop1 + 1);
                        if (k_pop2 >= s_len) {
                            break;
                        }
                        if (krhs_ok &&
                            (sbody[k_binop].opcode == _BINARY_OP_ADD_FLOAT_INPLACE ||
                             sbody[k_binop].opcode == _BINARY_OP_SUBTRACT_FLOAT_INPLACE ||
                             sbody[k_binop].opcode == _BINARY_OP_MULTIPLY_FLOAT_INPLACE) &&
                            (sbody[k_pop1].opcode == _POP_TOP_NOP ||
                             sbody[k_pop1].opcode == _POP_TOP_FLOAT) &&
                            sbody[k_pop2].opcode == _POP_TOP_NOP)
                        {
                            k = NEXT_SBODY_IDX(k_pop2 + 1);
                        }
                        else {
                            break;
                        }
                    }
                    if (k >= s_len) {
                        continue;
                    }
                    uint16_t slop = sbody[k].opcode;
                    int sv = -1;
                    if (slop == _LOAD_FAST || slop == _LOAD_FAST_BORROW) {
                        sv = (int)sbody[k].oparg;
                    }
                    else if (slop >= _LOAD_FAST_0 && slop <= _LOAD_FAST_7) {
                        sv = (int)(slop - _LOAD_FAST_0);
                    }
                    else if (slop >= _LOAD_FAST_BORROW_0 && slop <= _LOAD_FAST_BORROW_7) {
                        sv = (int)(slop - _LOAD_FAST_BORROW_0);
                    }
                    if (sv != v) {
                        continue;
                    }
                    int m = NEXT_SBODY_IDX(k + 1);
                    if (!is_slot && m < s_len && sbody[m].opcode == _LOCK_OBJECT) {
                        m = NEXT_SBODY_IDX(m + 1);
                    }
                    uint16_t expected_store = is_slot ? _STORE_ATTR_SLOT : _STORE_ATTR_INSTANCE_VALUE;
                    int m_pop = NEXT_SBODY_IDX(m + 1);
                    if (m_pop < s_len &&
                        sbody[m].opcode == expected_store &&
                        (is_slot ? (sbody[m].operand0 & 0x1FFF) : sbody[m].operand0) == offset &&
                        sbody[m_pop].opcode == _POP_TOP_NOP)
                    {
                        sbody[i_binop].opcode = inplace_binop;
                        sbody[i_pop2].opcode = _POP_TOP_NOP;
                        if (is_slot && (sbody[i_attr].operand0 >> 13) == 2) {
                            sbody[i_attr].oparg = 254;
                            sbody[k].opcode = _POP_TOP_NOP;
                        }
                        else {
                            sbody[k].opcode = _POP_TOP_FLOAT;
                        }
                        sbody[k].oparg = 0;
                        sbody[k].operand0 = 0;
                        for (int c = k + 1; c <= m_pop; c++) {
                            sbody[c].opcode = _NOP;
                        }
                    }
                }
            }
            #undef NEXT_SBODY_IDX
            for (int i = 0; i < s_len; i++) {
                if (sbody[i].opcode != _NOP) {
                    output[out_idx++] = sbody[i];
                }
            }
            output[out_idx++] = jump_inst;
            return remove_unneeded_uops(output, out_idx);
        }
    }

    /* Fallback: Hoist leading _ITER_CHECK_{RANGE,LIST,TUPLE} (and depth-0
     * _CHECK_STACK_SPACE_OPERAND / _CHECK_RECURSION_REMAINING across balanced
     * inlined calls) before _MAKE_WARM without duplicating the loop body. */
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
        bool safe_to_hoist_iter = true;
        int inline_depth = 0;
        int first_recursion_check_idx = -1;
        bool hoist_stack_check[UOP_MAX_TRACE_LENGTH] = {0};
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
            if (op == _CHECK_STACK_SPACE_OPERAND ||
                op == _GUARD_GLOBALS_VERSION ||
                op == _GUARD_BUILTINS_IS_CANONICAL ||
                op == _CHECK_FUNCTION_VERSION_INLINE ||
                op == _CHECK_PEP_523)
            {
                if (inline_depth == 0) {
                    hoist_stack_check[i] = true;
                }
                else if (op == _CHECK_STACK_SPACE_OPERAND) {
                    can_hoist_recursion_check = false;
                }
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
                op != _RETURN_VALUE &&
                op != _MAKE_HEAP_SAFE &&
                op != _STORE_ATTR_SLOT &&
                op != _STORE_ATTR_INSTANCE_VALUE &&
                op != _BINARY_OP_TRUEDIV_FLOAT &&
                op != _BINARY_OP_TRUEDIV_FLOAT_INPLACE &&
                op != _BINARY_OP_TRUEDIV_FLOAT_INPLACE_RIGHT)
            {
                can_hoist_recursion_check = false;
            }
        }
        if (inline_depth != 0) {
            safe_to_hoist_iter = false;
        }
        if (safe_to_hoist_iter) {
            /* If hoisting _ITER_CHECK_RANGE, check if the loop induction variable's
             * initial _POP_TOP right after _ITER_NEXT_RANGE can be specialized to
             * _POP_TOP_INT by guarding localsplus[idx_var] in the preheader. */
            if (output[iter_check_idx].opcode == _ITER_CHECK_RANGE &&
                output[iter_check_idx].oparg == 0)
            {
                int scan = iter_check_idx + 1;
                while (scan < length - 1 &&
                       (output[scan].opcode == _NOP ||
                        output[scan].opcode == _SET_IP ||
                        output[scan].opcode == _CHECK_VALIDITY ||
                        output[scan].opcode == _GUARD_NOT_EXHAUSTED_RANGE))
                {
                    scan++;
                }
                if (scan + 2 < length - 1 && output[scan].opcode == _ITER_NEXT_RANGE) {
                    int sw_idx = scan + 1;
                    while (sw_idx < length - 1 &&
                           (output[sw_idx].opcode == _NOP ||
                            output[sw_idx].opcode == _SET_IP ||
                            output[sw_idx].opcode == _CHECK_VALIDITY))
                    {
                        sw_idx++;
                    }
                    int pop_idx = sw_idx + 1;
                    while (pop_idx < length - 1 && output[pop_idx].opcode == _NOP) {
                        pop_idx++;
                    }
                    if (pop_idx < length - 1 && output[pop_idx].opcode == _POP_TOP) {
                        int sv = -1;
                        uint16_t sop = output[sw_idx].opcode;
                        if (sop == _SWAP_FAST) {
                            sv = (int)output[sw_idx].oparg;
                        }
                        else if (sop >= _SWAP_FAST_0 && sop <= _SWAP_FAST_7) {
                            sv = (int)(sop - _SWAP_FAST_0);
                        }
                        if (sv >= 0 && sv < 64) {
                            bool other_writes = false;
                            int d = 0;
                            for (int k = 2; k < length - 1; k++) {
                                uint16_t kop = output[k].opcode;
                                if (kop == _PUSH_FRAME || kop == _CREATE_INIT_FRAME) {
                                    d++;
                                }
                                else if (kop == _RETURN_VALUE) {
                                    d--;
                                }
                                else if (d == 0 && k != sw_idx) {
                                    if ((kop == _SWAP_FAST && (int)output[k].oparg == sv) ||
                                        (kop >= _SWAP_FAST_0 && kop <= _SWAP_FAST_7 &&
                                         (int)(kop - _SWAP_FAST_0) == sv) ||
                                        (kop == _LOAD_FAST_AND_CLEAR && (int)output[k].oparg == sv))
                                    {
                                        other_writes = true;
                                        break;
                                    }
                                }
                            }
                            if (!other_writes) {
                                output[iter_check_idx].oparg = (uint16_t)(0x8000 | sv);
                                output[pop_idx].opcode = _POP_TOP_INT;
                            }
                        }
                    }
                }
            }
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
            if (can_hoist_recursion_check) {
                for (int i = iter_check_idx + 1; i < length - 1; i++) {
                    if (hoist_stack_check[i]) {
                        _PyUOpInstruction stk_inst = output[i];
                        stk_inst.target = output[iter_check_idx].target;
                        temp[out_idx++] = stk_inst;
                    }
                }
                if (first_recursion_check_idx >= 0) {
                    _PyUOpInstruction rec_inst = output[first_recursion_check_idx];
                    rec_inst.target = output[iter_check_idx].target;
                    temp[out_idx++] = rec_inst;
                }
            }
            temp[out_idx++] = output[1]; /* _MAKE_WARM */
            for (int i = 2; i < length; i++) {
                if (i == iter_check_idx || output[i].opcode == _NOP ||
                    (can_hoist_recursion_check && hoist_stack_check[i]))
                {
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

static void
inline_leaf_strict_calls(_PyUOpInstruction *buffer, int length)
{
    for (int i = 0; i < length; i++) {
        if (buffer[i].opcode != _INIT_CALL_PY_EXACT_ARGS) {
            continue;
        }
        int nargs = (int)buffer[i].oparg;
        if (nargs < 0 || nargs > 7) {
            continue;
        }
        int s_idx = i + 1;
        while (s_idx < length &&
               (buffer[s_idx].opcode == _NOP ||
                buffer[s_idx].opcode == _SET_IP ||
                buffer[s_idx].opcode == _CHECK_VALIDITY))
        {
            s_idx++;
        }
        if (s_idx >= length || buffer[s_idx].opcode != _SAVE_RETURN_OFFSET) {
            continue;
        }
        int p_idx = s_idx + 1;
        while (p_idx < length &&
               (buffer[p_idx].opcode == _NOP ||
                buffer[p_idx].opcode == _SET_IP ||
                buffer[p_idx].opcode == _CHECK_VALIDITY))
        {
            p_idx++;
        }
        if (p_idx >= length || buffer[p_idx].opcode != _PUSH_FRAME) {
            continue;
        }
        uint32_t call_target = buffer[p_idx].target;

        int arg_idx[8];
        int check_stk_idx = -1;
        int check_rec_idx = -1;
        int scan = i - 1;
        bool args_ok = true;
        for (int a = nargs; a >= 1; a--) {
            while (scan >= 0 &&
                   (buffer[scan].opcode == _NOP ||
                    buffer[scan].opcode == _SET_IP ||
                    buffer[scan].opcode == _CHECK_VALIDITY ||
                    buffer[scan].opcode == _CHECK_FUNCTION_VERSION_INLINE ||
                    buffer[scan].opcode == _CHECK_STACK_SPACE_OPERAND ||
                    buffer[scan].opcode == _CHECK_RECURSION_REMAINING))
            {
                if (buffer[scan].opcode == _CHECK_STACK_SPACE_OPERAND) {
                    check_stk_idx = scan;
                }
                else if (buffer[scan].opcode == _CHECK_RECURSION_REMAINING) {
                    check_rec_idx = scan;
                }
                scan--;
            }
            if (scan < 0) {
                args_ok = false;
                break;
            }
            uint16_t aop = buffer[scan].opcode;
            if (aop == _LOAD_FAST_BORROW ||
                (aop >= _LOAD_FAST_BORROW_0 && aop <= _LOAD_FAST_BORROW_7) ||
                aop == _LOAD_CONST_INLINE_BORROW ||
                aop == _LOAD_CONST_INLINE ||
                aop == _LOAD_SMALL_INT)
            {
                arg_idx[a] = scan;
                scan--;
            }
            else {
                args_ok = false;
                break;
            }
        }
        if (!args_ok) {
            continue;
        }
        while (scan >= 0 &&
               (buffer[scan].opcode == _NOP ||
                buffer[scan].opcode == _SET_IP ||
                buffer[scan].opcode == _CHECK_VALIDITY ||
                buffer[scan].opcode == _CHECK_FUNCTION_VERSION_INLINE ||
                buffer[scan].opcode == _CHECK_STACK_SPACE_OPERAND ||
                buffer[scan].opcode == _CHECK_RECURSION_REMAINING))
        {
            if (buffer[scan].opcode == _CHECK_STACK_SPACE_OPERAND) {
                check_stk_idx = scan;
            }
            else if (buffer[scan].opcode == _CHECK_RECURSION_REMAINING) {
                check_rec_idx = scan;
            }
            scan--;
        }
        if (scan < 0 || buffer[scan].opcode != _SWAP || buffer[scan].oparg != 2) {
            continue;
        }
        int swap_idx = scan--;
        while (scan >= 0 &&
               (buffer[scan].opcode == _NOP ||
                buffer[scan].opcode == _SET_IP ||
                buffer[scan].opcode == _CHECK_VALIDITY ||
                buffer[scan].opcode == _CHECK_FUNCTION_VERSION_INLINE ||
                buffer[scan].opcode == _CHECK_STACK_SPACE_OPERAND ||
                buffer[scan].opcode == _CHECK_RECURSION_REMAINING))
        {
            if (buffer[scan].opcode == _CHECK_STACK_SPACE_OPERAND) {
                check_stk_idx = scan;
            }
            else if (buffer[scan].opcode == _CHECK_RECURSION_REMAINING) {
                check_rec_idx = scan;
            }
            scan--;
        }
        if (scan < 0 ||
            (buffer[scan].opcode != _LOAD_CONST_INLINE_BORROW &&
             buffer[scan].opcode != _LOAD_CONST_INLINE))
        {
            continue;
        }
        int callable_idx = scan--;
        while (scan >= 0 &&
               (buffer[scan].opcode == _NOP ||
                buffer[scan].opcode == _SET_IP ||
                buffer[scan].opcode == _CHECK_VALIDITY ||
                buffer[scan].opcode == _CHECK_FUNCTION_VERSION_INLINE ||
                buffer[scan].opcode == _CHECK_STACK_SPACE_OPERAND ||
                buffer[scan].opcode == _CHECK_RECURSION_REMAINING ||
                buffer[scan].opcode == _GUARD_TYPE_VERSION))
        {
            if (buffer[scan].opcode == _CHECK_STACK_SPACE_OPERAND) {
                check_stk_idx = scan;
            }
            else if (buffer[scan].opcode == _CHECK_RECURSION_REMAINING) {
                check_rec_idx = scan;
            }
            scan--;
        }
        if (scan < 0) {
            continue;
        }
        uint16_t self_op = buffer[scan].opcode;
        if (self_op != _LOAD_FAST_BORROW &&
            !(self_op >= _LOAD_FAST_BORROW_0 && self_op <= _LOAD_FAST_BORROW_7))
        {
            continue;
        }
        int self_idx = scan;

        int ret_idx = -1;
        int first_callee_idx = -1;
        bool callee_ok = true;
        for (int k = p_idx + 1; k < length; k++) {
            uint16_t kop = buffer[k].opcode;
            if (kop == _NOP || kop == _SET_IP || kop == _CHECK_VALIDITY ||
                kop == _CHECK_RECURSION_REMAINING || kop == _TIER2_RESUME_CHECK)
            {
                continue;
            }
            if (kop == _RETURN_VALUE) {
                ret_idx = k;
                break;
            }
            if (first_callee_idx < 0) {
                if (kop == _LOAD_FAST_BORROW_0 ||
                    (kop == _LOAD_FAST_BORROW && buffer[k].oparg == 0))
                {
                    first_callee_idx = k;
                    continue;
                }
                callee_ok = false;
                break;
            }
            if (kop == _LOAD_FAST_BORROW ||
                (kop >= _LOAD_FAST_BORROW_0 && kop <= _LOAD_FAST_BORROW_7))
            {
                int v = (kop == _LOAD_FAST_BORROW) ? (int)buffer[k].oparg :
                        (int)(kop - _LOAD_FAST_BORROW_0);
                if (v < 0 || v > nargs) {
                    callee_ok = false;
                    break;
                }
            }
            else if (kop == _LOAD_CONST_INLINE_BORROW ||
                     kop == _LOAD_CONST_INLINE ||
                     kop == _LOAD_SMALL_INT ||
                     (kop == _LOAD_ATTR_SLOT && (buffer[k].operand0 >> 13) != 0) ||
                     kop == _LOAD_ATTR_SLOT_NO_POP ||
                     (kop == _STORE_ATTR_SLOT && ((buffer[k].operand0 >> 13) != 0 || buffer[k].oparg == 255)) ||
                     kop == _BINARY_OP_ADD_FLOAT ||
                     kop == _BINARY_OP_ADD_FLOAT_INPLACE ||
                     kop == _BINARY_OP_ADD_FLOAT_INPLACE_RIGHT ||
                     kop == _BINARY_OP_SUBTRACT_FLOAT ||
                     kop == _BINARY_OP_SUBTRACT_FLOAT_INPLACE ||
                     kop == _BINARY_OP_SUBTRACT_FLOAT_INPLACE_RIGHT ||
                     kop == _BINARY_OP_MULTIPLY_FLOAT ||
                     kop == _BINARY_OP_MULTIPLY_FLOAT_INPLACE ||
                     kop == _BINARY_OP_MULTIPLY_FLOAT_INPLACE_RIGHT ||
                     kop == _BINARY_OP_TRUEDIV_FLOAT ||
                     kop == _BINARY_OP_TRUEDIV_FLOAT_INPLACE ||
                     kop == _BINARY_OP_TRUEDIV_FLOAT_INPLACE_RIGHT ||
                     kop == _BINARY_OP_FMA_FLOAT_INPLACE ||
                     kop == _BINARY_OP_BITWISE_INT ||
                     kop == _BINARY_OP_ARITH_FLOAT_INT ||
                     kop == _BINARY_OP_ADD_INT ||
                     kop == _BINARY_OP_ADD_INT_INPLACE ||
                     kop == _BINARY_OP_ADD_INT_INPLACE_RIGHT ||
                     kop == _BINARY_OP_SUBTRACT_INT ||
                     kop == _BINARY_OP_SUBTRACT_INT_INPLACE ||
                     kop == _BINARY_OP_SUBTRACT_INT_INPLACE_RIGHT ||
                     kop == _BINARY_OP_MULTIPLY_INT ||
                     kop == _BINARY_OP_MULTIPLY_INT_INPLACE ||
                     kop == _BINARY_OP_MULTIPLY_INT_INPLACE_RIGHT ||
                     kop == _UNARY_NEGATIVE_FLOAT_INPLACE ||
                     kop == _POP_TOP_NOP ||
                     kop == _POP_TOP_FLOAT ||
                     kop == _POP_TOP_INT)
            {
                /* Allowed non-escaping, guard-free instruction */
            }
            else {
                callee_ok = false;
                break;
            }
        }
        if (!callee_ok || ret_idx < 0 || first_callee_idx < 0) {
            continue;
        }

        _PyUOpInstruction saved_arg[8];
        saved_arg[0] = buffer[self_idx];
        for (int a = 1; a <= nargs; a++) {
            saved_arg[a] = buffer[arg_idx[a]];
            buffer[arg_idx[a]].opcode = _NOP;
        }
        buffer[callable_idx].opcode = _NOP;
        buffer[swap_idx].opcode = _NOP;
        if (check_stk_idx >= 0) {
            buffer[check_stk_idx].opcode = _NOP;
        }
        if (check_rec_idx >= 0) {
            buffer[check_rec_idx].opcode = _NOP;
        }
        buffer[i].opcode = _NOP;
        buffer[s_idx].opcode = _NOP;
        buffer[p_idx].opcode = _NOP;
        buffer[first_callee_idx].opcode = _NOP;
        buffer[ret_idx].opcode = _NOP;

        for (int k = p_idx + 1; k < ret_idx; k++) {
            uint16_t kop = buffer[k].opcode;
            if (kop == _CHECK_RECURSION_REMAINING || kop == _SET_IP) {
                buffer[k].opcode = _NOP;
                continue;
            }
            if (k == first_callee_idx) {
                continue;
            }
            if (kop == _LOAD_FAST_BORROW ||
                (kop >= _LOAD_FAST_BORROW_0 && kop <= _LOAD_FAST_BORROW_7))
            {
                int v = (kop == _LOAD_FAST_BORROW) ? (int)buffer[k].oparg :
                        (int)(kop - _LOAD_FAST_BORROW_0);
                buffer[k] = saved_arg[v];
            }
            buffer[k].target = call_target;
        }
        i = ret_idx;
    }
}

static void
eliminate_non_escaping_struct_allocs(_PyUOpInstruction *buffer, int length)
{
    for (int i = 0; i < length; i++) {
        if (buffer[i].opcode != _ALLOCATE_OBJECT) {
            continue;
        }
        int init_idx = i + 1;
        while (init_idx < length &&
               (buffer[init_idx].opcode == _NOP ||
                buffer[init_idx].opcode == _SET_IP ||
                buffer[init_idx].opcode == _CHECK_VALIDITY))
        {
            init_idx++;
        }
        if (init_idx >= length || buffer[init_idx].opcode != _CREATE_INIT_FRAME) {
            continue;
        }
        int nargs = (int)buffer[init_idx].oparg;
        if (nargs < 1 || nargs > 7) {
            continue;
        }
        int p_idx = init_idx + 1;
        while (p_idx < length &&
               (buffer[p_idx].opcode == _NOP ||
                buffer[p_idx].opcode == _SET_IP ||
                buffer[p_idx].opcode == _CHECK_VALIDITY))
        {
            p_idx++;
        }
        if (p_idx >= length || buffer[p_idx].opcode != _PUSH_FRAME) {
            continue;
        }

        /* Parse __init__ body: must be a sequence of
         * _LOAD_FAST_BORROW_<arg> + _LOAD_FAST_BORROW_0 (self) +
         * _STORE_ATTR_SLOT (strict slot) + _POP_TOP_NOP
         * followed by _LOAD_CONST_INLINE_BORROW(None) + _RETURN_VALUE +
         * _POP_TOP_NOP + _MAKE_HEAP_SAFE + _RETURN_VALUE. */
        uint64_t arg_slot_offset[8] = {0};
        int k = p_idx + 1;
        bool init_ok = true;
        int stores_seen = 0;
        int ret_idx = -1;
        while (k < length) {
            uint16_t kop = buffer[k].opcode;
            if (kop == _NOP || kop == _SET_IP || kop == _CHECK_VALIDITY ||
                kop == _CHECK_RECURSION_REMAINING || kop == _TIER2_RESUME_CHECK)
            {
                k++;
                continue;
            }
            int av = -1;
            if (kop == _LOAD_FAST_BORROW) {
                av = (int)buffer[k].oparg;
            }
            else if (kop >= _LOAD_FAST_BORROW_1 && kop <= _LOAD_FAST_BORROW_7) {
                av = (int)(kop - _LOAD_FAST_BORROW_0);
            }
            if (av >= 1 && av <= nargs && arg_slot_offset[av] == 0) {
                int k1 = k + 1;
                while (k1 < length &&
                       (buffer[k1].opcode == _NOP ||
                        buffer[k1].opcode == _SET_IP ||
                        buffer[k1].opcode == _CHECK_VALIDITY))
                {
                    k1++;
                }
                if (k1 >= length ||
                    (buffer[k1].opcode != _LOAD_FAST_BORROW_0 &&
                     !(buffer[k1].opcode == _LOAD_FAST_BORROW && buffer[k1].oparg == 0)))
                {
                    init_ok = false;
                    break;
                }
                int k2 = k1 + 1;
                while (k2 < length &&
                       (buffer[k2].opcode == _NOP ||
                        buffer[k2].opcode == _SET_IP ||
                        buffer[k2].opcode == _CHECK_VALIDITY))
                {
                    k2++;
                }
                if (k2 >= length || buffer[k2].opcode != _STORE_ATTR_SLOT ||
                    ((buffer[k2].operand0 >> 13) == 0 && buffer[k2].oparg != 255))
                {
                    init_ok = false;
                    break;
                }
                int k3 = k2 + 1;
                while (k3 < length &&
                       (buffer[k3].opcode == _NOP ||
                        buffer[k3].opcode == _SET_IP ||
                        buffer[k3].opcode == _CHECK_VALIDITY))
                {
                    k3++;
                }
                if (k3 >= length || buffer[k3].opcode != _POP_TOP_NOP) {
                    init_ok = false;
                    break;
                }
                arg_slot_offset[av] = buffer[k2].operand0 & 0x1FFF;
                stores_seen++;
                k = k3 + 1;
                continue;
            }
            if (kop == _LOAD_CONST_INLINE_BORROW || kop == _LOAD_CONST_INLINE) {
                int k1 = k + 1;
                while (k1 < length &&
                       (buffer[k1].opcode == _NOP ||
                        buffer[k1].opcode == _SET_IP ||
                        buffer[k1].opcode == _CHECK_VALIDITY))
                {
                    k1++;
                }
                int k2 = k1 + 1;
                while (k2 < length &&
                       (buffer[k2].opcode == _NOP ||
                        buffer[k2].opcode == _SET_IP ||
                        buffer[k2].opcode == _CHECK_VALIDITY))
                {
                    k2++;
                }
                int k3 = k2 + 1;
                while (k3 < length &&
                       (buffer[k3].opcode == _NOP ||
                        buffer[k3].opcode == _SET_IP ||
                        buffer[k3].opcode == _CHECK_VALIDITY))
                {
                    k3++;
                }
                int k4 = k3 + 1;
                while (k4 < length &&
                       (buffer[k4].opcode == _NOP ||
                        buffer[k4].opcode == _SET_IP ||
                        buffer[k4].opcode == _CHECK_VALIDITY))
                {
                    k4++;
                }
                if (k4 < length &&
                    buffer[k1].opcode == _RETURN_VALUE &&
                    buffer[k2].opcode == _POP_TOP_NOP &&
                    buffer[k3].opcode == _MAKE_HEAP_SAFE &&
                    buffer[k4].opcode == _RETURN_VALUE)
                {
                    ret_idx = k4;
                    break;
                }
            }
            init_ok = false;
            break;
        }
        if (!init_ok || ret_idx < 0 || stores_seen != nargs) {
            continue;
        }

        /* Must be immediately stored into a local variable via _SWAP_FAST + _POP_TOP */
        int sw_idx = ret_idx + 1;
        while (sw_idx < length &&
               (buffer[sw_idx].opcode == _NOP ||
                buffer[sw_idx].opcode == _SET_IP ||
                buffer[sw_idx].opcode == _CHECK_VALIDITY))
        {
            sw_idx++;
        }
        if (sw_idx >= length) {
            continue;
        }
        int item_var = -1;
        if (buffer[sw_idx].opcode == _SWAP_FAST) {
            item_var = (int)buffer[sw_idx].oparg;
        }
        else if (buffer[sw_idx].opcode >= _SWAP_FAST_0 &&
                 buffer[sw_idx].opcode <= _SWAP_FAST_7)
        {
            item_var = (int)(buffer[sw_idx].opcode - _SWAP_FAST_0);
        }
        if (item_var < 0) {
            continue;
        }
        int sw_pop = sw_idx + 1;
        while (sw_pop < length &&
               (buffer[sw_pop].opcode == _NOP || buffer[sw_pop].opcode == _SET_IP))
        {
            sw_pop++;
        }
        if (sw_pop >= length ||
            (buffer[sw_pop].opcode != _POP_TOP &&
             buffer[sw_pop].opcode != _POP_TOP_NOP))
        {
            continue;
        }

        /* Scan backwards from i (_ALLOCATE_OBJECT) to find _PUSH_NULL, callable load,
         * and the argument expressions for 1..nargs.
         * We support (nargs - 1) single-uop loads for arguments 1..(nargs - 1), and
         * a pure multi-uop (or single-uop) expression for argument nargs. */
        int push_null_idx = -1;
        int scan = i - 1;
        int depth = 0;
        while (scan >= 0) {
            uint16_t sop = buffer[scan].opcode;
            if (sop == _PUSH_NULL && depth == nargs) {
                push_null_idx = scan;
                break;
            }
            if (sop == _LOAD_FAST_BORROW || sop == _LOAD_FAST ||
                (sop >= _LOAD_FAST_BORROW_0 && sop <= _LOAD_FAST_BORROW_7) ||
                (sop >= _LOAD_FAST_0 && sop <= _LOAD_FAST_7) ||
                sop == _LOAD_CONST_INLINE_BORROW || sop == _LOAD_CONST_INLINE ||
                sop == _LOAD_SMALL_INT)
            {
                depth++;
            }
            else if (sop == _BINARY_OP_ADD_INT || sop == _BINARY_OP_ADD_INT_INPLACE ||
                     sop == _BINARY_OP_ADD_INT_INPLACE_RIGHT ||
                     sop == _BINARY_OP_SUBTRACT_INT || sop == _BINARY_OP_SUBTRACT_INT_INPLACE ||
                     sop == _BINARY_OP_SUBTRACT_INT_INPLACE_RIGHT ||
                     sop == _BINARY_OP_MULTIPLY_INT || sop == _BINARY_OP_MULTIPLY_INT_INPLACE ||
                     sop == _BINARY_OP_MULTIPLY_INT_INPLACE_RIGHT ||
                     sop == _BINARY_OP_BITWISE_INT || sop == _BINARY_OP_EXTEND)
            {
                depth++;
            }
            else if (sop == _POP_TOP_INT || sop == _POP_TOP_FLOAT ||
                     sop == _POP_TOP_NOP || sop == _POP_TOP)
            {
                depth--;
            }
            else if (sop == _NOP || sop == _SET_IP || sop == _CHECK_VALIDITY ||
                     sop == _CHECK_RECURSION_REMAINING ||
                     sop == _CHECK_STACK_SPACE_OPERAND ||
                     sop == _CHECK_FUNCTION_VERSION_INLINE ||
                     sop == _GUARD_TOS_INT || sop == _GUARD_NOS_INT ||
                     sop == _GUARD_TOS_FLOAT || sop == _GUARD_NOS_FLOAT ||
                     sop == _GUARD_BINARY_OP_EXTEND)
            {
                /* stack depth unchanged */
            }
            else {
                break;
            }
            scan--;
        }
        if (push_null_idx <= 0) {
            continue;
        }
        int call_load_idx = push_null_idx - 1;
        while (call_load_idx >= 0 &&
               (buffer[call_load_idx].opcode == _NOP ||
                buffer[call_load_idx].opcode == _SET_IP ||
                buffer[call_load_idx].opcode == _CHECK_VALIDITY ||
                buffer[call_load_idx].opcode == _GUARD_GLOBALS_VERSION))
        {
            call_load_idx--;
        }
        if (call_load_idx < 0 ||
            (buffer[call_load_idx].opcode != _LOAD_CONST_INLINE &&
             buffer[call_load_idx].opcode != _LOAD_CONST_INLINE_BORROW))
        {
            continue;
        }

        /* Collect single-uop arguments 1..(nargs - 1) immediately after _PUSH_NULL */
        int single_arg_idx[8];
        int cur = push_null_idx + 1;
        bool args_valid = true;
        for (int a = 1; a < nargs; a++) {
            while (cur < i &&
                   (buffer[cur].opcode == _NOP ||
                    buffer[cur].opcode == _SET_IP ||
                    buffer[cur].opcode == _CHECK_VALIDITY))
            {
                cur++;
            }
            if (cur >= i) {
                args_valid = false;
                break;
            }
            uint16_t aop = buffer[cur].opcode;
            if (aop == _LOAD_FAST_BORROW ||
                (aop >= _LOAD_FAST_BORROW_0 && aop <= _LOAD_FAST_BORROW_7) ||
                aop == _LOAD_CONST_INLINE_BORROW ||
                aop == _LOAD_CONST_INLINE ||
                aop == _LOAD_SMALL_INT)
            {
                single_arg_idx[a] = cur;
                cur++;
            }
            else {
                args_valid = false;
                break;
            }
        }
        if (!args_valid) {
            continue;
        }
        /* Ensure arg nargs leaves an owned reference on the stack (ends with a binary op) */
        bool last_arg_owned = false;
        int check_rec_idx = -1;
        int check_stk_idx = -1;
        for (int m = cur; m < i; m++) {
            uint16_t mop = buffer[m].opcode;
            if (mop == _BINARY_OP_ADD_INT || mop == _BINARY_OP_ADD_INT_INPLACE ||
                mop == _BINARY_OP_ADD_INT_INPLACE_RIGHT ||
                mop == _BINARY_OP_SUBTRACT_INT || mop == _BINARY_OP_SUBTRACT_INT_INPLACE ||
                mop == _BINARY_OP_SUBTRACT_INT_INPLACE_RIGHT ||
                mop == _BINARY_OP_MULTIPLY_INT || mop == _BINARY_OP_MULTIPLY_INT_INPLACE ||
                mop == _BINARY_OP_MULTIPLY_INT_INPLACE_RIGHT ||
                mop == _BINARY_OP_BITWISE_INT)
            {
                last_arg_owned = true;
            }
            else if (mop == _CHECK_RECURSION_REMAINING) {
                check_rec_idx = m;
            }
            else if (mop == _CHECK_STACK_SPACE_OPERAND) {
                check_stk_idx = m;
            }
        }
        if (!last_arg_owned) {
            continue;
        }

        /* Verify that all subsequent uses of item_var in the trace are
         * _LOAD_FAST_BORROW(item_var) + _LOAD_ATTR_SLOT + _POP_TOP_NOP,
         * and that between sw_pop and the last use of item_var there are no side-exits. */
        int use_load_idx[8];
        int use_slot_idx[8];
        int use_pop_idx[8];
        int use_arg_num[8];
        int num_uses = 0;
        int last_use_idx = -1;
        bool uses_valid = true;
        for (int u = sw_pop + 1; u < length; u++) {
            uint16_t uop = buffer[u].opcode;
            int lv = -1;
            if (uop == _LOAD_FAST || uop == _LOAD_FAST_BORROW ||
                uop == _SWAP_FAST || uop == _LOAD_FAST_AND_CLEAR)
            {
                lv = (int)buffer[u].oparg;
            }
            else if (uop >= _LOAD_FAST_0 && uop <= _LOAD_FAST_7) {
                lv = (int)(uop - _LOAD_FAST_0);
            }
            else if (uop >= _LOAD_FAST_BORROW_0 && uop <= _LOAD_FAST_BORROW_7) {
                lv = (int)(uop - _LOAD_FAST_BORROW_0);
            }
            else if (uop >= _SWAP_FAST_0 && uop <= _SWAP_FAST_7) {
                lv = (int)(uop - _SWAP_FAST_0);
            }
            if (lv != item_var) {
                continue;
            }
            if (uop != _LOAD_FAST_BORROW &&
                !(uop >= _LOAD_FAST_BORROW_0 && uop <= _LOAD_FAST_BORROW_7))
            {
                uses_valid = false;
                break;
            }
            int s1 = u + 1;
            while (s1 < length && (buffer[s1].opcode == _NOP || buffer[s1].opcode == _SET_IP)) {
                s1++;
            }
            if (s1 >= length || buffer[s1].opcode != _LOAD_ATTR_SLOT) {
                uses_valid = false;
                break;
            }
            int s2 = s1 + 1;
            while (s2 < length && (buffer[s2].opcode == _NOP || buffer[s2].opcode == _SET_IP)) {
                s2++;
            }
            if (s2 >= length || buffer[s2].opcode != _POP_TOP_NOP || num_uses >= 8) {
                uses_valid = false;
                break;
            }
            uint64_t off = buffer[s1].operand0 & 0x1FFF;
            int matched_arg = -1;
            for (int a = 1; a <= nargs; a++) {
                if (arg_slot_offset[a] == off) {
                    matched_arg = a;
                    break;
                }
            }
            if (matched_arg < 1) {
                uses_valid = false;
                break;
            }
            use_load_idx[num_uses] = u;
            use_slot_idx[num_uses] = s1;
            use_pop_idx[num_uses] = s2;
            use_arg_num[num_uses] = matched_arg;
            num_uses++;
            last_use_idx = s2;
            u = s2;
        }
        if (!uses_valid || num_uses == 0) {
            continue;
        }
        /* Check no side-exits between sw_pop and last_use_idx */
        bool region_safe = true;
        for (int m = sw_pop + 1; m <= last_use_idx; m++) {
            uint16_t mop = buffer[m].opcode;
            if (mop == _NOP || mop == _SET_IP || mop == _CHECK_VALIDITY ||
                mop == _LOAD_FAST_BORROW ||
                (mop >= _LOAD_FAST_BORROW_0 && mop <= _LOAD_FAST_BORROW_7) ||
                mop == _LOAD_CONST_INLINE_BORROW || mop == _LOAD_CONST_INLINE ||
                mop == _LOAD_SMALL_INT ||
                mop == _LOAD_ATTR_SLOT || mop == _POP_TOP_NOP)
            {
                continue;
            }
            region_safe = false;
            break;
        }
        if (!region_safe) {
            continue;
        }

        /* Save single-uop argument instructions and NOP out allocation + __init__ */
        _PyUOpInstruction saved_arg[8];
        for (int a = 1; a < nargs; a++) {
            saved_arg[a] = buffer[single_arg_idx[a]];
            if (saved_arg[a].opcode == _LOAD_CONST_INLINE) {
                saved_arg[a].opcode = _LOAD_CONST_INLINE_BORROW;
            }
            buffer[single_arg_idx[a]].opcode = _NOP;
        }
        buffer[call_load_idx].opcode = _NOP;
        buffer[push_null_idx].opcode = _NOP;
        if (check_rec_idx >= 0) {
            buffer[check_rec_idx].opcode = _NOP;
        }
        if (check_stk_idx >= 0) {
            buffer[check_stk_idx].opcode = _NOP;
        }
        for (int m = i; m <= ret_idx; m++) {
            buffer[m].opcode = _NOP;
        }

        /* Replace each slot load on item_var with the corresponding scalar value */
        for (int u = 0; u < num_uses; u++) {
            int a = use_arg_num[u];
            if (a < nargs) {
                uint32_t tgt = buffer[use_load_idx[u]].target;
                buffer[use_load_idx[u]] = saved_arg[a];
                buffer[use_load_idx[u]].target = tgt;
            }
            /* If a == nargs, use_load_idx[u] is already _LOAD_FAST_BORROW(item_var)! */
            buffer[use_slot_idx[u]].opcode = _NOP;
            buffer[use_pop_idx[u]].opcode = _NOP;
        }
        i = last_use_idx;
    }
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

    inline_leaf_strict_calls(output, length);
    eliminate_non_escaping_struct_allocs(output, length);
    length = remove_unneeded_uops(output, length);

    bool has_strict_slots = false;
    for (int i = 0; i < length; i++) {
        if ((output[i].opcode == _LOAD_ATTR_SLOT ||
             output[i].opcode == _STORE_ATTR_SLOT) &&
            (output[i].operand0 >> 13) != 0)
        {
            has_strict_slots = true;
            break;
        }
    }
    if (has_strict_slots && length > 180) {
        mark_inplace_float_attr_updates(output, length);
        length = remove_unneeded_uops(output, length);
    }

    length = hoist_loop_preheader_guards(output, length, curr_stacklen);

    mark_inplace_float_attr_updates(output, length);
    length = remove_unneeded_uops(output, length);

    OPT_STAT_INC(optimizer_successes);
    return length;
}

#endif /* _Py_TIER2 */
