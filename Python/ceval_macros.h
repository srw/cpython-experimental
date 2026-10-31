// Macros and other things needed by ceval.c, and bytecodes.c

/* Computed GOTOs, or
       the-optimization-commonly-but-improperly-known-as-"threaded code"
   using gcc's labels-as-values extension
   (http://gcc.gnu.org/onlinedocs/gcc/Labels-as-Values.html).

   The traditional bytecode evaluation loop uses a "switch" statement, which
   decent compilers will optimize as a single indirect branch instruction
   combined with a lookup table of jump addresses. However, since the
   indirect jump instruction is shared by all opcodes, the CPU will have a
   hard time making the right prediction for where to jump next (actually,
   it will be always wrong except in the uncommon case of a sequence of
   several identical opcodes).

   "Threaded code" in contrast, uses an explicit jump table and an explicit
   indirect jump instruction at the end of each opcode. Since the jump
   instruction is at a different address for each opcode, the CPU will make a
   separate prediction for each of these instructions, which is equivalent to
   predicting the second opcode of each opcode pair. These predictions have
   a much better chance to turn out valid, especially in small bytecode loops.

   A mispredicted branch on a modern CPU flushes the whole pipeline and
   can cost several CPU cycles (depending on the pipeline depth),
   and potentially many more instructions (depending on the pipeline width).
   A correctly predicted branch, however, is nearly free.

   At the time of this writing, the "threaded code" version is up to 15-20%
   faster than the normal "switch" version, depending on the compiler and the
   CPU architecture.

   NOTE: care must be taken that the compiler doesn't try to "optimize" the
   indirect jumps by sharing them between all opcodes. Such optimizations
   can be disabled on gcc by using the -fno-gcse flag (or possibly
   -fno-crossjumping).
*/

/* Use macros rather than inline functions, to make it as clear as possible
 * to the C compiler that the tracing check is a simple test then branch.
 * We want to be sure that the compiler knows this before it generates
 * the CFG.
 */

#ifdef WITH_DTRACE
#define OR_DTRACE_LINE | (PyDTrace_LINE_ENABLED() ? 255 : 0)
#else
#define OR_DTRACE_LINE
#endif

#ifdef HAVE_COMPUTED_GOTOS
    #ifndef USE_COMPUTED_GOTOS
    #define USE_COMPUTED_GOTOS 1
    #endif
#else
    #if defined(USE_COMPUTED_GOTOS) && USE_COMPUTED_GOTOS
    #error "Computed gotos are not supported on this compiler."
    #endif
    #undef USE_COMPUTED_GOTOS
    #define USE_COMPUTED_GOTOS 0
#endif

#ifdef Py_STATS
#define INSTRUCTION_STATS(op) \
    do { \
        PyStats *s = _PyStats_GET(); \
        OPCODE_EXE_INC(op); \
        if (s) s->opcode_stats[lastopcode].pair_count[op]++; \
        lastopcode = op; \
    } while (0)
#else
#define INSTRUCTION_STATS(op) ((void)0)
#endif

#ifdef Py_STATS
#   define TAIL_CALL_PARAMS _PyInterpreterFrame *frame, _PyStackRef *stack_pointer, PyThreadState *tstate, _Py_CODEUNIT *next_instr, const void *instruction_funcptr_table, int oparg, int lastopcode
#   define TAIL_CALL_ARGS frame, stack_pointer, tstate, next_instr, instruction_funcptr_table, oparg, lastopcode
#else
#   define TAIL_CALL_PARAMS _PyInterpreterFrame *frame, _PyStackRef *stack_pointer, PyThreadState *tstate, _Py_CODEUNIT *next_instr, const void *instruction_funcptr_table, int oparg
#   define TAIL_CALL_ARGS frame, stack_pointer, tstate, next_instr, instruction_funcptr_table, oparg
#endif

#if _Py_TAIL_CALL_INTERP
#   if defined(__clang__) || defined(__GNUC__)
#       if !_Py__has_attribute(preserve_none) || !_Py__has_attribute(musttail)
#           error "This compiler does not have support for efficient tail calling."
#       endif
#   elif defined(_MSC_VER) && (_MSC_VER < 1950)
#       error "You need at least VS 2026 / PlatformToolset v145 for tail calling."
#   endif
#   if defined(_MSC_VER) && !defined(__clang__)
#      define Py_MUSTTAIL [[msvc::musttail]]
#      define Py_PRESERVE_NONE_CC __preserve_none
#   else
#       define Py_MUSTTAIL __attribute__((musttail))
#       define Py_PRESERVE_NONE_CC __attribute__((preserve_none))
#   endif
    typedef PyObject *(Py_PRESERVE_NONE_CC *py_tail_call_funcptr)(TAIL_CALL_PARAMS);

#   define DISPATCH_TABLE_VAR instruction_funcptr_table
#   define DISPATCH_TABLE instruction_funcptr_handler_table
#   define TRACING_DISPATCH_TABLE instruction_funcptr_tracing_table
#   define TARGET(op) Py_NO_INLINE PyObject *Py_PRESERVE_NONE_CC _TAIL_CALL_##op(TAIL_CALL_PARAMS)

#   define DISPATCH_GOTO() \
        do { \
            Py_MUSTTAIL return (((py_tail_call_funcptr *)instruction_funcptr_table)[opcode])(TAIL_CALL_ARGS); \
        } while (0)
#   define DISPATCH_GOTO_NON_TRACING() \
        do { \
            Py_MUSTTAIL return (((py_tail_call_funcptr *)DISPATCH_TABLE)[opcode])(TAIL_CALL_ARGS); \
        } while (0)
#   define JUMP_TO_LABEL(name) \
        do { \
            Py_MUSTTAIL return (_TAIL_CALL_##name)(TAIL_CALL_ARGS); \
        } while (0)
#   ifdef Py_STATS
#       define JUMP_TO_PREDICTED(name) \
            do { \
                Py_MUSTTAIL return (_TAIL_CALL_##name)(frame, stack_pointer, tstate, this_instr, instruction_funcptr_table, oparg, lastopcode); \
            } while (0)
#   else
#       define JUMP_TO_PREDICTED(name) \
            do { \
                Py_MUSTTAIL return (_TAIL_CALL_##name)(frame, stack_pointer, tstate, this_instr, instruction_funcptr_table, oparg); \
            } while (0)
#   endif
#    define LABEL(name) TARGET(name)
#elif USE_COMPUTED_GOTOS
#  define DISPATCH_TABLE_VAR opcode_targets
#  define DISPATCH_TABLE opcode_targets_table
#  define TRACING_DISPATCH_TABLE opcode_tracing_targets_table
#  define TARGET(op) TARGET_##op:
#  define DISPATCH_GOTO() goto *opcode_targets[opcode]
#  define DISPATCH_GOTO_NON_TRACING() goto *DISPATCH_TABLE[opcode];
#  define JUMP_TO_LABEL(name) goto name;
#  define JUMP_TO_PREDICTED(name) goto PREDICTED_##name;
#  define LABEL(name) name:
#else
#  define TARGET(op) case op: TARGET_##op:
#  define DISPATCH_GOTO() dispatch_code = opcode | tracing_mode ; goto dispatch_opcode
#  define DISPATCH_GOTO_NON_TRACING() dispatch_code = opcode; goto dispatch_opcode
#  define JUMP_TO_LABEL(name) goto name;
#  define JUMP_TO_PREDICTED(name) goto PREDICTED_##name;
#  define LABEL(name) name:
#endif

#if (_Py_TAIL_CALL_INTERP || USE_COMPUTED_GOTOS) && _Py_TIER2
#  define IS_JIT_TRACING() (DISPATCH_TABLE_VAR == TRACING_DISPATCH_TABLE)
#  define ENTER_TRACING() \
    DISPATCH_TABLE_VAR = TRACING_DISPATCH_TABLE;
#  define LEAVE_TRACING() \
    DISPATCH_TABLE_VAR = DISPATCH_TABLE;
#else
#  define IS_JIT_TRACING() (tracing_mode != 0)
#  define ENTER_TRACING() tracing_mode = 255
#  define LEAVE_TRACING() tracing_mode = 0
#endif

#if _Py_TIER2
#define STOP_TRACING() \
    do { \
        if (IS_JIT_TRACING()) { \
            LEAVE_TRACING(); \
            _PyJit_FinalizeTracing(tstate, 0); \
        } \
    } while (0);
#else
#define STOP_TRACING() ((void)(0));
#endif

/* PRE_DISPATCH_GOTO() does lltrace if enabled. Normally a no-op */
#ifdef Py_DEBUG
#define PRE_DISPATCH_GOTO() if (frame->lltrace >= 5) { \
    lltrace_instruction(frame, stack_pointer, next_instr, opcode, oparg); }
#else
#define PRE_DISPATCH_GOTO() ((void)0)
#endif

#ifdef Py_DEBUG
#define LLTRACE_RESUME_FRAME() \
do { \
    _PyFrame_SetStackPointer(frame, stack_pointer); \
    int lltrace = maybe_lltrace_resume_frame(frame, GLOBALS()); \
    stack_pointer = _PyFrame_GetStackPointer(frame); \
    frame->lltrace = lltrace; \
} while (0)
#else
#define LLTRACE_RESUME_FRAME() ((void)0)
#endif

#ifdef Py_GIL_DISABLED
#define QSBR_QUIESCENT_STATE(tstate) _Py_qsbr_quiescent_state(((_PyThreadStateImpl *)tstate)->qsbr)
#else
#define QSBR_QUIESCENT_STATE(tstate)
#endif


/* Do interpreter dispatch accounting for tracing and instrumentation */
#define DISPATCH() \
    { \
        _PyFrame_StackAssertInvalid(frame); \
        NEXTOPARG(); \
        PRE_DISPATCH_GOTO(); \
        DISPATCH_GOTO(); \
    }

#define DISPATCH_NON_TRACING() \
    { \
        _PyFrame_StackAssertInvalid(frame); \
        NEXTOPARG(); \
        PRE_DISPATCH_GOTO(); \
        DISPATCH_GOTO_NON_TRACING(); \
    }

#define DISPATCH_SAME_OPARG() \
    { \
        opcode = next_instr->op.code; \
        PRE_DISPATCH_GOTO(); \
        DISPATCH_GOTO_NON_TRACING(); \
    }

#define DISPATCH_INLINED(NEW_FRAME)                              \
    do {                                                         \
        assert(!IS_PEP523_HOOKED(tstate));                       \
        _PyFrame_SetStackPointer(frame, stack_pointer);          \
        _PyFrame_StackPointerValidate(frame);                    \
        assert((NEW_FRAME)->previous == frame);                  \
        frame = tstate->current_frame = (NEW_FRAME);             \
        CALL_STAT_INC(inlined_py_calls);                         \
        JUMP_TO_LABEL(start_frame);                              \
    } while (0)

/* Tuple access macros */

#ifndef Py_DEBUG
#define GETITEM(v, i) PyTuple_GET_ITEM((v), (i))
#else
static inline PyObject *
GETITEM(PyObject *v, Py_ssize_t i) {
    assert(PyTuple_Check(v));
    assert(i >= 0);
    assert(i < PyTuple_GET_SIZE(v));
    return PyTuple_GET_ITEM(v, i);
}
#endif

/* Code access macros */

/* The integer overflow is checked by an assertion below. */
#define INSTR_OFFSET() ((int)(next_instr - _PyFrame_GetBytecode(frame)))
#define NEXTOPARG()  do { \
        _Py_CODEUNIT word  = {.cache = FT_ATOMIC_LOAD_UINT16_RELAXED(*(uint16_t*)next_instr)}; \
        opcode = word.op.code; \
        oparg = word.op.arg; \
    } while (0)

/* JUMPBY makes the generator identify the instruction as a jump. SKIP_OVER is
 * for advancing to the next instruction, taking into account cache entries
 * and skipped instructions.
 */
#define JUMPBY(x)       (next_instr += (x))
#define SKIP_OVER(x)    (next_instr += (x))

#define STACK_LEVEL()     ((int)(stack_pointer - _PyFrame_Stackbase(frame)))
#define STACK_SIZE()      (_PyFrame_GetCode(frame)->co_stacksize)

#define WITHIN_STACK_BOUNDS() \
   (frame->owner == FRAME_OWNED_BY_INTERPRETER || (STACK_LEVEL() >= 0 && STACK_LEVEL() <= STACK_SIZE()))

#if defined(Py_DEBUG) && !defined(_Py_JIT)
// This allows temporary stack "overflows", provided it's all in the cache at any point of time.
#define ASSERT_WITHIN_STACK_BOUNDS_IGNORING_CACHE(F, L) \
   assert(frame->owner == FRAME_OWNED_BY_INTERPRETER || (STACK_LEVEL() >= 0 && (STACK_LEVEL()) <= STACK_SIZE()))
#else
#define ASSERT_WITHIN_STACK_BOUNDS_IGNORING_CACHE ASSERT_WITHIN_STACK_BOUNDS
#endif

/* Data access macros */
#define FRAME_CO_CONSTS (_PyFrame_GetCode(frame)->co_consts)
#define FRAME_CO_NAMES  (_PyFrame_GetCode(frame)->co_names)

/* Local variable macros */

#define LOCALS_ARRAY    (frame->localsplus)
#define GETLOCAL(i)     (frame->localsplus[i])


#ifdef Py_STATS
#define UPDATE_MISS_STATS(INSTNAME)                              \
    do {                                                         \
        STAT_INC(opcode, miss);                                  \
        STAT_INC((INSTNAME), miss);                              \
        /* The counter is always the first cache entry: */       \
        if (ADAPTIVE_COUNTER_TRIGGERS(next_instr->cache)) {      \
            STAT_INC((INSTNAME), deopt);                         \
        }                                                        \
    } while (0)
#else
#define UPDATE_MISS_STATS(INSTNAME) ((void)0)
#endif


// Try to lock an object in the free threading build, if it's not already
// locked. Use with a DEOPT_IF() to deopt if the object is already locked.
// These are no-ops in the default GIL build. The general pattern is:
//
// DEOPT_IF(!LOCK_OBJECT(op));
// if (/* condition fails */) {
//     UNLOCK_OBJECT(op);
//     DEOPT_IF(true);
//  }
//  ...
//  UNLOCK_OBJECT(op);
//
// NOTE: The object must be unlocked on every exit code path and you should
// avoid any potentially escaping calls (like PyStackRef_CLOSE) while the
// object is locked.
#ifdef Py_GIL_DISABLED
#  define IS_UNSHARED_LOCAL_OBJECT(op) \
    _PyObject_IsUnsharedLocal(_PyObject_CAST(op))
#  define LOCK_SHARED_OBJECT(op) \
    PyMutex_LockFast(&(_PyObject_CAST(op))->ob_mutex)
#  define UNLOCK_SHARED_OBJECT(op) \
    PyMutex_Unlock(&(_PyObject_CAST(op))->ob_mutex)
#  define LOCK_OBJECT(op) \
    (_PyObject_IsUnsharedLocal(_PyObject_CAST(op)) || \
     PyMutex_LockFast(&(_PyObject_CAST(op))->ob_mutex))
#  define UNLOCK_OBJECT(op) \
    do { \
        PyObject *_op = _PyObject_CAST(op); \
        if (PyMutex_IsLocked(&_op->ob_mutex)) { \
            PyMutex_Unlock(&_op->ob_mutex); \
        } \
    } while (0)
#else
#  define IS_UNSHARED_LOCAL_OBJECT(op) (1)
#  define LOCK_SHARED_OBJECT(op) (1)
#  define UNLOCK_SHARED_OBJECT(op) ((void)0)
#  define LOCK_OBJECT(op) (1)
#  define UNLOCK_OBJECT(op) ((void)0)
#endif

#define GLOBALS() frame->f_globals
#define BUILTINS() frame->f_builtins
#define LOCALS() frame->f_locals
#define CONSTS() _PyFrame_GetCode(frame)->co_consts
#define NAMES() _PyFrame_GetCode(frame)->co_names

#if defined(WITH_DTRACE) && !defined(Py_BUILD_CORE_MODULE)
static void dtrace_function_entry(_PyInterpreterFrame *);
static void dtrace_function_return(_PyInterpreterFrame *);

#define DTRACE_FUNCTION_ENTRY()  \
    if (PyDTrace_FUNCTION_ENTRY_ENABLED()) { \
        dtrace_function_entry(frame); \
    }

#define DTRACE_FUNCTION_RETURN() \
    if (PyDTrace_FUNCTION_RETURN_ENABLED()) { \
        dtrace_function_return(frame); \
    }
#else
#define DTRACE_FUNCTION_ENTRY() ((void)0)
#define DTRACE_FUNCTION_RETURN() ((void)0)
#endif

/* This takes a uint16_t instead of a _Py_BackoffCounter,
 * because it is used directly on the cache entry in generated code,
 * which is always an integral type. */
// Force re-specialization when tracing a side exit to get good side exits.
#define ADAPTIVE_COUNTER_TRIGGERS(COUNTER) \
    backoff_counter_triggers(forge_backoff_counter((COUNTER)))

#ifdef Py_GIL_DISABLED
/* Counters are unreachable when thread-local bytecode is disabled,
 * so there is no need to update them. */
#define ADVANCE_ADAPTIVE_COUNTER(COUNTER) \
    do { \
        _Py_BackoffCounter cnt = (COUNTER); \
        if (!backoff_counter_is_unreachable(cnt)) { \
            (COUNTER) = advance_backoff_counter(cnt); \
        } \
    } while (0);

#define PAUSE_ADAPTIVE_COUNTER(COUNTER) \
    do { \
        _Py_BackoffCounter cnt = (COUNTER); \
        if (!backoff_counter_is_unreachable(cnt)) { \
            (COUNTER) = pause_backoff_counter(cnt); \
        } \
    } while (0);
#else
#define ADVANCE_ADAPTIVE_COUNTER(COUNTER) \
    do { \
        (COUNTER) = advance_backoff_counter((COUNTER)); \
    } while (0);

#define PAUSE_ADAPTIVE_COUNTER(COUNTER) \
    do { \
        (COUNTER) = pause_backoff_counter((COUNTER)); \
    } while (0);
#endif

#ifdef ENABLE_SPECIALIZATION
/* Multiple threads may execute these concurrently if thread-local bytecode is
 * disabled and they all execute the main copy of the bytecode. Specialization
 * is disabled in that case so the value is unused, but the RMW cycle should be
 * free of data races.
 */
#define RECORD_BRANCH_TAKEN(bitset, flag) \
    FT_ATOMIC_STORE_UINT16_RELAXED(       \
        bitset, (FT_ATOMIC_LOAD_UINT16_RELAXED(bitset) << 1) | (flag))
#else
#define RECORD_BRANCH_TAKEN(bitset, flag)
#endif

#define UNBOUNDLOCAL_ERROR_MSG \
    "cannot access local variable '%s' where it is not associated with a value"
#define UNBOUNDFREE_ERROR_MSG \
    "cannot access free variable '%s' where it is not associated with a value" \
    " in enclosing scope"
#define NAME_ERROR_MSG "name '%.200s' is not defined"

// If a trace function sets a new f_lineno and
// *then* raises, we use the destination when searching
// for an exception handler, displaying the traceback, and so on
#define INSTRUMENTED_JUMP(src, dest, event) \
do { \
    _Py_CODEUNIT *_dest = (dest); \
    if (tstate->tracing) {\
        next_instr = _dest; \
    } else { \
        _PyFrame_SetStackPointer(frame, stack_pointer); \
        next_instr = _Py_call_instrumentation_jump(this_instr, tstate, event, frame, src, _dest); \
        stack_pointer = _PyFrame_GetStackPointer(frame); \
        if (next_instr == NULL) { \
            next_instr = _dest + 1; \
            JUMP_TO_LABEL(error); \
        } \
    } \
} while (0);


static inline int _Py_EnterRecursivePy(PyThreadState *tstate) {
    return (tstate->py_recursion_remaining-- <= 0) &&
        _Py_CheckRecursiveCallPy(tstate);
}

static inline void _Py_LeaveRecursiveCallPy(PyThreadState *tstate)  {
    tstate->py_recursion_remaining++;
}

/* Implementation of "macros" that modify the instruction pointer,
 * stack pointer, or frame pointer.
 * These need to treated differently by tier 1 and 2.
 * The Tier 1 version is here; Tier 2 is inlined in ceval.c. */

#define LOAD_IP(OFFSET) do { \
        next_instr = frame->instr_ptr + (OFFSET); \
    } while (0)

/* There's no STORE_IP(), it's inlined by the code generator. */

#define LOAD_SP() \
stack_pointer = _PyFrame_GetStackPointer(frame)

#define SAVE_SP() \
_PyFrame_SetStackPointer(frame, stack_pointer)

/* Tier-switching macros. */

#define TIER1_TO_TIER2(EXECUTOR)                        \
do {                                                   \
    OPT_STAT_INC(traces_executed);                     \
    next_instr = _Py_jit_entry((EXECUTOR), frame, stack_pointer, tstate); \
    frame = tstate->current_frame;                     \
    stack_pointer = _PyFrame_GetStackPointer(frame);   \
    int keep_tracing_bit = (uintptr_t)next_instr & 1;   \
    next_instr = (_Py_CODEUNIT *)(((uintptr_t)next_instr) & (~1)); \
    if (next_instr == NULL) {                          \
        /* gh-140104: The exception handler expects frame->instr_ptr
            to after this_instr, not this_instr! */ \
        next_instr = frame->instr_ptr + 1;                 \
        JUMP_TO_LABEL(error);                          \
    }                                                  \
    if (keep_tracing_bit) { \
        assert(uop_buffer_length(&((_PyThreadStateImpl *)tstate)->jit_tracer_state->code_buffer)); \
        ENTER_TRACING(); \
        DISPATCH_NON_TRACING(); \
    } \
    DISPATCH();                                        \
} while (0)

#define TIER2_TO_TIER2(EXECUTOR) \
do {                                                   \
    OPT_STAT_INC(traces_executed);                     \
    current_executor = (EXECUTOR);                     \
    goto tier2_start;                                  \
} while (0)

#define GOTO_TIER_ONE_SETUP \
    tstate->current_executor = NULL;                              \
    OPT_HIST(trace_uop_execution_counter, trace_run_length_hist); \
    _PyFrame_SetStackPointer(frame, stack_pointer);

#define GOTO_TIER_ONE(TARGET) \
    do \
    { \
        GOTO_TIER_ONE_SETUP \
        return (_Py_CODEUNIT *)(TARGET); \
    } while (0)

#define GOTO_TIER_ONE_CONTINUE_TRACING(TARGET) \
    do \
    { \
        GOTO_TIER_ONE_SETUP \
        return (_Py_CODEUNIT *)(((uintptr_t)(TARGET))| 1); \
    } while (0)

#define CURRENT_OPARG()    (next_uop[-1].oparg)
#define CURRENT_OPERAND0_64() (next_uop[-1].operand0)
#define CURRENT_OPERAND1_64() (next_uop[-1].operand1)
#define CURRENT_OPERAND0_32() (next_uop[-1].operand0)
#define CURRENT_OPERAND1_32() (next_uop[-1].operand1)
#define CURRENT_OPERAND0_16() (next_uop[-1].operand0)
#define CURRENT_OPERAND1_16() (next_uop[-1].operand1)
#define CURRENT_TARGET()   (next_uop[-1].target)

#define JUMP_TO_JUMP_TARGET() goto jump_to_jump_target
#define JUMP_TO_ERROR() goto jump_to_error_target

/* Stackref macros */

/* How much scratch space to give stackref to PyObject* conversion. */
#define MAX_STACKREF_SCRATCH 10

#define STACKREFS_TO_PYOBJECTS(ARGS, ARG_COUNT, NAME) \
    /* +1 because vectorcall might use -1 to write self */ \
    PyObject *NAME##_temp[MAX_STACKREF_SCRATCH+1]; \
    PyObject **NAME = _PyObjectArray_FromStackRefArray(ARGS, ARG_COUNT, NAME##_temp);

#define STACKREFS_TO_PYOBJECTS_CLEANUP(NAME) \
    /* +1 because we +1 previously */ \
    _PyObjectArray_Free(NAME - 1, NAME##_temp);

#define CONVERSION_FAILED(NAME) ((NAME) == NULL)

#if defined(Py_DEBUG) && !defined(_Py_JIT)
#define SET_CURRENT_CACHED_VALUES(N) current_cached_values = (N)
#define CHECK_CURRENT_CACHED_VALUES(N) assert(current_cached_values == (N))
#else
#define SET_CURRENT_CACHED_VALUES(N) ((void)0)
#define CHECK_CURRENT_CACHED_VALUES(N) ((void)0)
#endif

#define IS_PEP523_HOOKED(tstate) (tstate->interp->eval_frame != NULL)

static inline Py_ALWAYS_INLINE int
check_periodics(PyThreadState *tstate) {
    _Py_CHECK_EMSCRIPTEN_SIGNALS_PERIODICALLY();
    QSBR_QUIESCENT_STATE(tstate);
    if (_Py_atomic_load_uintptr_relaxed(&tstate->eval_breaker) & _PY_EVAL_EVENTS_MASK) {
        return _Py_HandlePending(tstate);
    }
    return 0;
}

static inline Py_ALWAYS_INLINE int
check_periodics_at_end(PyThreadState *tstate, _PyInterpreterFrame *frame) {
    _Py_CHECK_EMSCRIPTEN_SIGNALS_PERIODICALLY();
    QSBR_QUIESCENT_STATE(tstate);
    if (_Py_atomic_load_uintptr_relaxed(&tstate->eval_breaker) & _PY_EVAL_EVENTS_MASK) {
        // Do not handle pending interrupts if the previous instruction was LOAD_SPECIAL
        // This may also not handle interrupts if a cache looks like LOAD_SPECIAL,
        // but this is benign as we won't skip periodic checks indefinitely.
        if (frame->instr_ptr[-1].op.code == LOAD_SPECIAL) {
            return 0;
        }
        return _Py_HandlePending(tstate);
    }
    return 0;
}

// Mark the generator as executing. Returns true if the state was changed,
// false if it was already executing or finished.
static inline bool
gen_try_set_executing(PyGenObject *gen)
{
#ifdef Py_GIL_DISABLED
    if (!_PyObject_IsUnsharedLocal((PyObject *)gen)) {
        int8_t frame_state = _Py_atomic_load_int8_relaxed(&gen->gi_frame_state);
        while (frame_state < FRAME_SUSPENDED_YIELD_FROM_LOCKED) {
            if (_Py_atomic_compare_exchange_int8(&gen->gi_frame_state,
                                                 &frame_state,
                                                 FRAME_EXECUTING)) {
                return true;
            }
        }
        // NB: We return false for FRAME_SUSPENDED_YIELD_FROM_LOCKED as well.
        // That case is rare enough that we can just handle it in the deopt.
        return false;
    }
#endif
    // Use faster non-atomic modifications in the GIL-enabled build and when
    // the object is unshared thread-local in the free-threaded build.
    if (gen->gi_frame_state < FRAME_EXECUTING) {
        assert(gen->gi_frame_state != FRAME_SUSPENDED_YIELD_FROM_LOCKED);
        gen->gi_frame_state = FRAME_EXECUTING;
        return true;
    }
    return false;
}

// Macro for inplace float binary ops (tier 2 only).
// Mutates the uniquely-referenced (or single-slot attribute) TARGET operand in place.
// TARGET must be either left or right.
static inline int
_PyObject_IsInplaceFloatTarget(PyObject *ob)
{
#if !defined(Py_GIL_DISABLED)
    return Py_REFCNT(ob) <= 2;
#else
    return (_Py_IsOwnedByCurrentThread(ob) &&
            _Py_atomic_load_uint32_relaxed(&ob->ob_ref_local) <= 2 &&
            _Py_atomic_load_ssize_relaxed(&ob->ob_ref_shared) == 0);
#endif
}

#define FLOAT_INPLACE_OP(left, right, TARGET, OP)                        \
    do {                                                                 \
        PyObject *left_o = PyStackRef_AsPyObjectBorrow(left);            \
        PyObject *right_o = PyStackRef_AsPyObjectBorrow(right);          \
        assert(PyFloat_CheckExact(left_o));                              \
        assert(PyFloat_CheckExact(right_o));                             \
        assert(_PyObject_IsInplaceFloatTarget(                           \
            PyStackRef_AsPyObjectBorrow(TARGET)));                       \
        STAT_INC(BINARY_OP, hit);                                        \
        double _dres =                                                   \
            ((PyFloatObject *)left_o)->ob_fval                           \
            OP ((PyFloatObject *)right_o)->ob_fval;                      \
        ((PyFloatObject *)PyStackRef_AsPyObjectBorrow(TARGET))           \
            ->ob_fval = _dres;                                           \
    } while (0)

// Inplace float true division. Sets _divop_err to 1 on zero division.
// Caller must check _divop_err and call ERROR_NO_POP() if set.
#define FLOAT_INPLACE_DIVOP(left, right, TARGET)                         \
    int _divop_err = 0;                                                  \
    do {                                                                 \
        PyObject *left_o = PyStackRef_AsPyObjectBorrow(left);            \
        PyObject *right_o = PyStackRef_AsPyObjectBorrow(right);          \
        assert(PyFloat_CheckExact(left_o));                              \
        assert(PyFloat_CheckExact(right_o));                             \
        assert(_PyObject_IsInplaceFloatTarget(                           \
            PyStackRef_AsPyObjectBorrow(TARGET)));                       \
        STAT_INC(BINARY_OP, hit);                                        \
        double _divisor = ((PyFloatObject *)right_o)->ob_fval;           \
        if (_divisor == 0.0) {                                           \
            PyErr_SetString(PyExc_ZeroDivisionError,                     \
                            "float division by zero");                   \
            _divop_err = 1;                                              \
            break;                                                       \
        }                                                                \
        double _dres = ((PyFloatObject *)left_o)->ob_fval / _divisor;    \
        ((PyFloatObject *)PyStackRef_AsPyObjectBorrow(TARGET))           \
            ->ob_fval = _dres;                                           \
    } while (0)

// Inplace compact int operation. TARGET is expected to be uniquely
// referenced at the optimizer level, but at runtime it may be a
// cached small int singleton. We check _Py_IsImmortal on TARGET
// to decide whether inplace mutation is safe.
//
// After the macro, _int_inplace_res holds the result (may be NULL
// on allocation failure). On success, TARGET was mutated in place
// and _int_inplace_res is a DUP'd reference to it. On fallback
// (small int target, small int result, or overflow), _int_inplace_res
// is from FUNC (_PyCompactLong_Add etc.).
// FUNC is the fallback function (_PyCompactLong_Add etc.)
#if PYLONG_BITS_IN_DIGIT == 30 && defined(__SIZEOF_INT128__)
#define INT_INPLACE_OP(left, right, TARGET, OP, FUNC)                    \
    _PyStackRef _int_inplace_res = FUNC##InPlace(                        \
        &(left), &(right),                                               \
        (PyLongObject *)PyStackRef_AsPyObjectBorrow(left),               \
        (PyLongObject *)PyStackRef_AsPyObjectBorrow(right));
#else
#define INT_INPLACE_OP(left, right, TARGET, OP, FUNC)                    \
    _PyStackRef _int_inplace_res = PyStackRef_NULL;                      \
    do {                                                                 \
        PyObject *target_o = PyStackRef_AsPyObjectBorrow(TARGET);        \
        if (_Py_IsImmortal(target_o)) {                                  \
            break;                                                       \
        }                                                                \
        assert(_PyObject_IsUniquelyReferenced(target_o));                \
        Py_ssize_t left_val = _PyLong_CompactValue(                      \
            (PyLongObject *)PyStackRef_AsPyObjectBorrow(left));          \
        Py_ssize_t right_val = _PyLong_CompactValue(                     \
            (PyLongObject *)PyStackRef_AsPyObjectBorrow(right));         \
        Py_ssize_t result = left_val OP right_val;                       \
        if (!_PY_IS_SMALL_INT(result)                                    \
            && ((twodigits)((stwodigits)result) + PyLong_MASK            \
                < (twodigits)PyLong_MASK + PyLong_BASE))                 \
        {                                                                \
            _PyLong_SetSignAndDigitCount(                                \
                (PyLongObject *)target_o, result < 0 ? -1 : 1, 1);       \
            ((PyLongObject *)target_o)->long_value.ob_digit[0] =         \
                (digit)(result < 0 ? -result : result);                  \
            _int_inplace_res = PyStackRef_DUP(TARGET);                   \
            break;                                                       \
        }                                                                \
    } while (0);                                                         \
    if (PyStackRef_IsNull(_int_inplace_res)) {                           \
        _int_inplace_res = FUNC(                                         \
            (PyLongObject *)PyStackRef_AsPyObjectBorrow(left),           \
            (PyLongObject *)PyStackRef_AsPyObjectBorrow(right));         \
    }
#endif

#define CALL_TP_ITERITEM_NO_ESCAPE(ITER, INDEX) \
    Py_TYPE(ITER)->_tp_iteritem((ITER), (INDEX))

#include "pycore_freelist.h"
#include "pycore_range.h"

static inline Py_ALWAYS_INLINE PyObject *
_PyFloat_FromDouble_Fast(double fval)
{
    PyFloatObject *op = _Py_FREELIST_POP(PyFloatObject, floats);
    if (__builtin_expect(op != NULL, 1)) {
        op->ob_fval = fval;
        return (PyObject *)op;
    }
    return PyFloat_FromDouble(fval);
}

static inline Py_ALWAYS_INLINE void
_PyFloat_ExactDealloc_Fast(PyObject *obj)
{
    assert(PyFloat_CheckExact(obj));
    _Py_FREELIST_FREE(floats, obj, PyObject_Free);
}

static inline Py_ALWAYS_INLINE _PyStackRef
_PyLong_FromLong_RangeIter(_PyRangeIterObject *r, long x)
{
    if (_PY_IS_SMALL_INT(x)) {
        return PyStackRef_FromPyObjectBorrow(
            (PyObject *)&_PyLong_SMALL_INTS[_PY_NSMALLNEGINTS + x]);
    }
    if (__builtin_expect(((twodigits)x + PyLong_MASK) < ((twodigits)PyLong_MASK + PyLong_BASE), 1)) {
        int slot = (int)(r->len & 1);
        PyLongObject *cached = (PyLongObject *)r->cached_item[slot];
        if (__builtin_expect(cached != NULL && _PyObject_IsUniquelyReferenced((PyObject *)cached), 1)) {
            _PyLong_SetSignAndDigitCount(cached, x < 0 ? -1 : 1, 1);
            cached->long_value.ob_digit[0] = (digit)(x < 0 ? -(twodigits)x : (twodigits)x);
            return PyStackRef_FromPyObjectNew((PyObject *)cached);
        }
        PyLongObject *v = _Py_FREELIST_POP(PyLongObject, ints);
        if (__builtin_expect(v != NULL, 1)) {
            _PyLong_SetSignAndDigitCount(v, x < 0 ? -1 : 1, 1);
            v->long_value.ob_digit[0] = (digit)(x < 0 ? -(twodigits)x : (twodigits)x);
            Py_XSETREF(r->cached_item[slot], Py_NewRef((PyObject *)v));
            return PyStackRef_FromPyObjectStealMortal((PyObject *)v);
        }
    }
    PyObject *res = PyLong_FromLong(x);
    if (res == NULL) {
        return PyStackRef_NULL;
    }
    return PyStackRef_FromPyObjectSteal(res);
}
#define _PyLong_FromLong_StackRef(x) _PyLong_FromLong_RangeIter(r, (x))

static inline Py_ALWAYS_INLINE stwodigits
_PyLong_AsSTwoDigits_Fast(const PyLongObject *op)
{
    assert(op->long_value.lv_tag < (3 << _PyLong_NON_SIZE_BITS));
    if (__builtin_expect(_PyLong_IsCompact(op), 1)) {
        return _PyLong_CompactValue(op);
    }
    twodigits mag = ((twodigits)op->long_value.ob_digit[1] << PyLong_SHIFT) |
                    op->long_value.ob_digit[0];
    return (op->long_value.lv_tag & _PyLong_SIGN_MASK) == 0
           ? (stwodigits)mag
           : -(stwodigits)mag;
}

static inline Py_ALWAYS_INLINE _PyStackRef
_PyLong_FromNonNegativeSsize_t_StackRef(Py_ssize_t x)
{
    assert(x >= 0);
    if (x < _PY_NSMALLPOSINTS) {
        return PyStackRef_FromPyObjectBorrow(
            (PyObject *)&_PyLong_SMALL_INTS[_PY_NSMALLNEGINTS + x]);
    }
    if (__builtin_expect(x < PyLong_BASE, 1)) {
        PyLongObject *v = _Py_FREELIST_POP(PyLongObject, ints);
        if (__builtin_expect(v != NULL, 1)) {
            _PyLong_SetSignAndDigitCount(v, 1, 1);
            v->long_value.ob_digit[0] = (digit)x;
            return PyStackRef_FromPyObjectStealMortal((PyObject *)v);
        }
    }
    PyObject *res_o = PyLong_FromSsize_t(x);
    if (res_o == NULL) {
        return PyStackRef_NULL;
    }
    return PyStackRef_FromPyObjectSteal(res_o);
}

static inline Py_ALWAYS_INLINE _PyStackRef
_PyLong_FromSTwoDigitsInPlace2(
    _PyStackRef *pleft, _PyStackRef *pright,
    PyLongObject *a, PyLongObject *b, stwodigits x)
{
    if (_PY_IS_SMALL_INT(x)) {
        return PyStackRef_FromPyObjectBorrow(
            (PyObject *)&_PyLong_SMALL_INTS[_PY_NSMALLNEGINTS + x]);
    }
    twodigits abs_x = x < 0 ? 0U - (twodigits)x : (twodigits)x;
    int sign = x < 0 ? -1 : 1;
    Py_ssize_t ndigits = (abs_x >> PyLong_SHIFT) == 0 ? 1 : 2;
    if (PyStackRef_RefcountOnObject(*pleft) &&
        _PyObject_IsUniquelyReferenced((PyObject *)a))
    {
        _PyLong_SetSignAndDigitCount(a, sign, ndigits);
        a->long_value.ob_digit[0] = (digit)(abs_x & PyLong_MASK);
        if (ndigits == 2) {
            a->long_value.ob_digit[1] = (digit)(abs_x >> PyLong_SHIFT);
        }
        _PyStackRef res = *pleft;
        *pleft = PyStackRef_FromPyObjectBorrow((PyObject *)a);
        return res;
    }
    if (PyStackRef_RefcountOnObject(*pright) &&
        _PyObject_IsUniquelyReferenced((PyObject *)b))
    {
        _PyLong_SetSignAndDigitCount(b, sign, ndigits);
        b->long_value.ob_digit[0] = (digit)(abs_x & PyLong_MASK);
        if (ndigits == 2) {
            b->long_value.ob_digit[1] = (digit)(abs_x >> PyLong_SHIFT);
        }
        _PyStackRef res = *pright;
        *pright = PyStackRef_FromPyObjectBorrow((PyObject *)b);
        return res;
    }
    PyLongObject *v = _Py_FREELIST_POP(PyLongObject, ints);
    if (__builtin_expect(v != NULL, 1)) {
        _PyLong_SetSignAndDigitCount(v, sign, ndigits);
        v->long_value.ob_digit[0] = (digit)(abs_x & PyLong_MASK);
        if (ndigits == 2) {
            v->long_value.ob_digit[1] = (digit)(abs_x >> PyLong_SHIFT);
        }
        return PyStackRef_FromPyObjectStealMortal((PyObject *)v);
    }
    PyObject *res_o = PyLong_FromSsize_t((Py_ssize_t)x);
    if (res_o == NULL) {
        return PyStackRef_NULL;
    }
    return PyStackRef_FromPyObjectSteal(res_o);
}

static inline Py_ALWAYS_INLINE _PyStackRef
_PyCompactLong_FromSTwoDigitsInPlace(
    _PyStackRef *pleft, _PyStackRef *pright,
    PyLongObject *a, PyLongObject *b, stwodigits x,
    _PyStackRef (*fallback)(PyLongObject *, PyLongObject *))
{
    if (_PY_IS_SMALL_INT(x)) {
        return PyStackRef_FromPyObjectBorrow(
            (PyObject *)&_PyLong_SMALL_INTS[_PY_NSMALLNEGINTS + x]);
    }
    if (__builtin_expect(((twodigits)x + PyLong_MASK) < ((twodigits)PyLong_MASK + PyLong_BASE), 1)) {
        if (PyStackRef_RefcountOnObject(*pleft) &&
            _PyObject_IsUniquelyReferenced((PyObject *)a))
        {
            _PyLong_SetSignAndDigitCount(a, x < 0 ? -1 : 1, 1);
            a->long_value.ob_digit[0] = (digit)(x < 0 ? -(twodigits)x : (twodigits)x);
            _PyStackRef res = *pleft;
            *pleft = PyStackRef_FromPyObjectBorrow((PyObject *)a);
            return res;
        }
        if (PyStackRef_RefcountOnObject(*pright) &&
            _PyObject_IsUniquelyReferenced((PyObject *)b))
        {
            _PyLong_SetSignAndDigitCount(b, x < 0 ? -1 : 1, 1);
            b->long_value.ob_digit[0] = (digit)(x < 0 ? -(twodigits)x : (twodigits)x);
            _PyStackRef res = *pright;
            *pright = PyStackRef_FromPyObjectBorrow((PyObject *)b);
            return res;
        }
        PyLongObject *v = _Py_FREELIST_POP(PyLongObject, ints);
        if (__builtin_expect(v != NULL, 1)) {
            _PyLong_SetSignAndDigitCount(v, x < 0 ? -1 : 1, 1);
            v->long_value.ob_digit[0] = (digit)(x < 0 ? -(twodigits)x : (twodigits)x);
            return PyStackRef_FromPyObjectStealMortal((PyObject *)v);
        }
    }
#if PYLONG_BITS_IN_DIGIT == 30 && defined(__SIZEOF_INT128__)
    return _PyLong_FromSTwoDigitsInPlace2(pleft, pright, a, b, x);
#else
    return fallback(a, b);
#endif
}

static inline Py_ALWAYS_INLINE _PyStackRef
_PyCompactLong_AddInPlace(
    _PyStackRef *pleft, _PyStackRef *pright,
    PyLongObject *a, PyLongObject *b)
{
    if (__builtin_expect(_PyLong_BothAreCompact(a, b), 1)) {
        stwodigits x = _PyLong_CompactValue(a) + _PyLong_CompactValue(b);
        return _PyCompactLong_FromSTwoDigitsInPlace(
            pleft, pright, a, b, x, _PyCompactLong_Add);
    }
#if PYLONG_BITS_IN_DIGIT == 30 && defined(__SIZEOF_INT128__)
    stwodigits x = _PyLong_AsSTwoDigits_Fast(a) + _PyLong_AsSTwoDigits_Fast(b);
    const stwodigits max_2digit = ((stwodigits)1 << (2 * PyLong_SHIFT)) - 1;
    if (__builtin_expect(x >= -max_2digit && x <= max_2digit, 1)) {
        return _PyLong_FromSTwoDigitsInPlace2(pleft, pright, a, b, x);
    }
#endif
    return PyStackRef_NULL;
}

static inline Py_ALWAYS_INLINE _PyStackRef
_PyCompactLong_SubtractInPlace(
    _PyStackRef *pleft, _PyStackRef *pright,
    PyLongObject *a, PyLongObject *b)
{
    if (__builtin_expect(_PyLong_BothAreCompact(a, b), 1)) {
        stwodigits x = _PyLong_CompactValue(a) - _PyLong_CompactValue(b);
        return _PyCompactLong_FromSTwoDigitsInPlace(
            pleft, pright, a, b, x, _PyCompactLong_Subtract);
    }
#if PYLONG_BITS_IN_DIGIT == 30 && defined(__SIZEOF_INT128__)
    stwodigits x = _PyLong_AsSTwoDigits_Fast(a) - _PyLong_AsSTwoDigits_Fast(b);
    const stwodigits max_2digit = ((stwodigits)1 << (2 * PyLong_SHIFT)) - 1;
    if (__builtin_expect(x >= -max_2digit && x <= max_2digit, 1)) {
        return _PyLong_FromSTwoDigitsInPlace2(pleft, pright, a, b, x);
    }
#endif
    return PyStackRef_NULL;
}

static inline Py_ALWAYS_INLINE _PyStackRef
_PyCompactLong_MultiplyInPlace(
    _PyStackRef *pleft, _PyStackRef *pright,
    PyLongObject *a, PyLongObject *b)
{
    if (__builtin_expect(_PyLong_BothAreCompact(a, b), 1)) {
        stwodigits x = _PyLong_CompactValue(a) * _PyLong_CompactValue(b);
        return _PyCompactLong_FromSTwoDigitsInPlace(
            pleft, pright, a, b, x, _PyCompactLong_Multiply);
    }
#if PYLONG_BITS_IN_DIGIT == 30 && defined(__SIZEOF_INT128__)
    __int128 prod = (__int128)_PyLong_AsSTwoDigits_Fast(a) *
                    (__int128)_PyLong_AsSTwoDigits_Fast(b);
    const stwodigits max_2digit = ((stwodigits)1 << (2 * PyLong_SHIFT)) - 1;
    if (__builtin_expect(prod >= -max_2digit && prod <= max_2digit, 1)) {
        return _PyLong_FromSTwoDigitsInPlace2(pleft, pright, a, b, (stwodigits)prod);
    }
#endif
    return PyStackRef_NULL;
}

static inline Py_ALWAYS_INLINE int
_PyEval_BinaryOpExtendGuardFast(_PyBinaryOpSpecializationDescr *d,
                                PyObject *left_o, PyObject *right_o)
{
    if (d->guard == NULL) {
        return Py_TYPE(left_o) == d->lhs_type && Py_TYPE(right_o) == d->rhs_type;
    }
    return d->guard(left_o, right_o);
}

static inline Py_ALWAYS_INLINE int
_PyEval_LoadAttrSlotFast(PyObject *owner_o, uint16_t index, int oparg, _PyStackRef *attr_out)
{
    PyObject **addr = (PyObject **)((char *)owner_o + (index & 0x1FFF));
    PyObject *attr_o = FT_ATOMIC_LOAD_PTR_ACQUIRE(*addr);
    if (__builtin_expect(attr_o == NULL, 0)) {
        return 0;
    }
    if (oparg == 254) {
        *attr_out = PyStackRef_FromPyObjectBorrow(attr_o);
        return 1;
    }
    if (oparg == 255) {
        assert(PyFloat_CheckExact(attr_o));
        if (!_PyObject_IsUniquelyReferenced(attr_o)) {
            PyObject *copy = _PyFloat_FromDouble_Fast(((PyFloatObject *)attr_o)->ob_fval);
            if (__builtin_expect(copy == NULL, 0)) {
                PyErr_Clear();
                return 0;
            }
            FT_ATOMIC_STORE_PTR_RELEASE(*addr, copy);
            _Py_DECREF_SPECIALIZED(attr_o, _PyFloat_ExactDealloc_Fast);
            attr_o = copy;
        }
        *attr_out = PyStackRef_FromPyObjectBorrow(attr_o);
        return 1;
    }
#ifdef Py_GIL_DISABLED
    if (IS_UNSHARED_LOCAL_OBJECT(owner_o)) {
        *attr_out = PyStackRef_FromPyObjectNew(attr_o);
        return 1;
    }
    return _Py_TryIncrefCompareStackRef(addr, attr_o, attr_out);
#else
    *attr_out = PyStackRef_FromPyObjectNew(attr_o);
    return 1;
#endif
}

#include "opcode.h"

static inline Py_ALWAYS_INLINE _PyStackRef
_PyCompactLong_BitwiseInPlace(_PyStackRef *pleft, _PyStackRef *pright,
                              PyLongObject *a, PyLongObject *b, int oparg)
{
    if (_PyLong_BothAreCompact(a, b)) {
        if (oparg == NB_AND || oparg == NB_INPLACE_AND) {
            return _PyLong_FromSTwoDigitsInPlace2(
                pleft, pright, a, b,
                (stwodigits)(_PyLong_CompactValue(a) & _PyLong_CompactValue(b)));
        }
        if (oparg == NB_OR || oparg == NB_INPLACE_OR) {
            return _PyLong_FromSTwoDigitsInPlace2(
                pleft, pright, a, b,
                (stwodigits)(_PyLong_CompactValue(a) | _PyLong_CompactValue(b)));
        }
        if (oparg == NB_XOR || oparg == NB_INPLACE_XOR) {
            return _PyLong_FromSTwoDigitsInPlace2(
                pleft, pright, a, b,
                (stwodigits)(_PyLong_CompactValue(a) ^ _PyLong_CompactValue(b)));
        }
        if (oparg == NB_RSHIFT || oparg == NB_INPLACE_RSHIFT) {
            Py_ssize_t rval = _PyLong_CompactValue(b);
            if (__builtin_expect((size_t)rval < PyLong_SHIFT, 1)) {
                return _PyLong_FromSTwoDigitsInPlace2(
                    pleft, pright, a, b,
                    (stwodigits)(_PyLong_CompactValue(a) >> rval));
            }
            return _PyLong_FromSTwoDigitsInPlace2(
                pleft, pright, a, b,
                _PyLong_CompactValue(a) < 0 ? -1 : 0);
        }
        if (oparg == NB_FLOOR_DIVIDE || oparg == NB_INPLACE_FLOOR_DIVIDE) {
            stwodigits lhs_val = (stwodigits)_PyLong_CompactValue(a);
            stwodigits rhs_val = (stwodigits)_PyLong_CompactValue(b);
            stwodigits q = lhs_val / rhs_val;
            stwodigits r = lhs_val % rhs_val;
            if (r != 0 && ((lhs_val ^ rhs_val) < 0)) {
                q--;
            }
            return _PyLong_FromSTwoDigitsInPlace2(pleft, pright, a, b, q);
        }
        if (oparg == NB_REMAINDER || oparg == NB_INPLACE_REMAINDER) {
            stwodigits lhs_val = (stwodigits)_PyLong_CompactValue(a);
            stwodigits rhs_val = (stwodigits)_PyLong_CompactValue(b);
            stwodigits rval = lhs_val % rhs_val;
            if (rval != 0 && ((lhs_val ^ rhs_val) < 0)) {
                rval += rhs_val;
            }
            return _PyLong_FromSTwoDigitsInPlace2(pleft, pright, a, b, rval);
        }
    }
    stwodigits lhs_val = _PyLong_AsSTwoDigits_Fast(a);
    stwodigits rhs_val = _PyLong_AsSTwoDigits_Fast(b);
    if (oparg == NB_AND || oparg == NB_INPLACE_AND) {
        return _PyLong_FromSTwoDigitsInPlace2(pleft, pright, a, b, lhs_val & rhs_val);
    }
    if (oparg == NB_OR || oparg == NB_INPLACE_OR) {
        return _PyLong_FromSTwoDigitsInPlace2(pleft, pright, a, b, lhs_val | rhs_val);
    }
    if (oparg == NB_XOR || oparg == NB_INPLACE_XOR) {
        return _PyLong_FromSTwoDigitsInPlace2(pleft, pright, a, b, lhs_val ^ rhs_val);
    }
    if (oparg == NB_RSHIFT || oparg == NB_INPLACE_RSHIFT) {
        if (__builtin_expect((uint64_t)rhs_val < 2 * PyLong_SHIFT, 1)) {
            return _PyLong_FromSTwoDigitsInPlace2(pleft, pright, a, b, lhs_val >> rhs_val);
        }
        return _PyLong_FromSTwoDigitsInPlace2(pleft, pright, a, b, lhs_val < 0 ? -1 : 0);
    }
    if (oparg == NB_FLOOR_DIVIDE || oparg == NB_INPLACE_FLOOR_DIVIDE) {
        stwodigits q = lhs_val / rhs_val;
        stwodigits r = lhs_val % rhs_val;
        if (r != 0 && ((lhs_val ^ rhs_val) < 0)) {
            q--;
        }
        return _PyLong_FromSTwoDigitsInPlace2(pleft, pright, a, b, q);
    }
    stwodigits rval = lhs_val % rhs_val;
    if (rval != 0 && ((lhs_val ^ rhs_val) < 0)) {
        rval += rhs_val;
    }
    return _PyLong_FromSTwoDigitsInPlace2(pleft, pright, a, b, rval);
}

static inline Py_ALWAYS_INLINE _PyStackRef
_PyFloat_MixedArithInPlace(_PyStackRef *pleft, _PyStackRef *pright,
                           PyObject *left_o, PyObject *right_o, int oparg)
{
    if (PyFloat_CheckExact(left_o)) {
        double lval = ((PyFloatObject *)left_o)->ob_fval;
        double rval = (double)_PyLong_AsSTwoDigits_Fast((PyLongObject *)right_o);
        double dres = (oparg == NB_MULTIPLY || oparg == NB_INPLACE_MULTIPLY) ? (lval * rval) :
                      (oparg == NB_ADD || oparg == NB_INPLACE_ADD) ? (lval + rval) : (lval - rval);
        if (PyStackRef_RefcountOnObject(*pleft) &&
            _PyObject_IsUniquelyReferenced(left_o))
        {
            ((PyFloatObject *)left_o)->ob_fval = dres;
            _PyStackRef res = *pleft;
            *pleft = PyStackRef_FromPyObjectBorrow(left_o);
            return res;
        }
        PyObject *f = _PyFloat_FromDouble_Fast(dres);
        return PyStackRef_FromPyObjectSteal(f);
    }
    else {
        double lval = (double)_PyLong_AsSTwoDigits_Fast((PyLongObject *)left_o);
        double rval = ((PyFloatObject *)right_o)->ob_fval;
        double dres = (oparg == NB_MULTIPLY || oparg == NB_INPLACE_MULTIPLY) ? (lval * rval) :
                      (oparg == NB_ADD || oparg == NB_INPLACE_ADD) ? (lval + rval) : (lval - rval);
        if (PyStackRef_RefcountOnObject(*pright) &&
            _PyObject_IsUniquelyReferenced(right_o))
        {
            ((PyFloatObject *)right_o)->ob_fval = dres;
            _PyStackRef res = *pright;
            *pright = PyStackRef_FromPyObjectBorrow(right_o);
            return res;
        }
        PyObject *f = _PyFloat_FromDouble_Fast(dres);
        return PyStackRef_FromPyObjectSteal(f);
    }
}

static inline Py_ALWAYS_INLINE _PyStackRef
_PyEval_BinaryOpExtendFast(_PyBinaryOpSpecializationDescr *d,
                           _PyStackRef *pleft, _PyStackRef *pright,
                           PyObject *left_o, PyObject *right_o, int oparg)
{
    if (PyLong_CheckExact(left_o) && PyLong_CheckExact(right_o) &&
        (((PyLongObject *)left_o)->long_value.lv_tag < (3 << _PyLong_NON_SIZE_BITS)) &&
        (((PyLongObject *)right_o)->long_value.lv_tag < (3 << _PyLong_NON_SIZE_BITS)))
    {
        PyLongObject *a = (PyLongObject *)left_o;
        PyLongObject *b = (PyLongObject *)right_o;
        if (oparg == NB_AND || oparg == NB_INPLACE_AND ||
            oparg == NB_OR || oparg == NB_INPLACE_OR ||
            oparg == NB_XOR || oparg == NB_INPLACE_XOR)
        {
            return _PyCompactLong_BitwiseInPlace(pleft, pright, a, b, oparg);
        }
        if (oparg == NB_RSHIFT || oparg == NB_INPLACE_RSHIFT) {
            stwodigits rval = _PyLong_AsSTwoDigits_Fast(b);
            if (__builtin_expect(rval >= 0 && rval < 2 * PyLong_SHIFT, 1)) {
                return _PyCompactLong_BitwiseInPlace(pleft, pright, a, b, oparg);
            }
        }
        if ((oparg == NB_REMAINDER || oparg == NB_INPLACE_REMAINDER) &&
            !_PyLong_IsZero(b))
        {
            return _PyCompactLong_BitwiseInPlace(pleft, pright, a, b, oparg);
        }
    }
    else if ((oparg == NB_MULTIPLY || oparg == NB_INPLACE_MULTIPLY) &&
             PyLong_CheckExact(left_o) && _PyLong_IsCompact((PyLongObject *)left_o) &&
             PyFloat_CheckExact(right_o))
    {
        double dres = (double)_PyLong_CompactValue((PyLongObject *)left_o) *
                      ((PyFloatObject *)right_o)->ob_fval;
        if (PyStackRef_RefcountOnObject(*pright) &&
            _PyObject_IsUniquelyReferenced(right_o))
        {
            ((PyFloatObject *)right_o)->ob_fval = dres;
            _PyStackRef res = *pright;
            *pright = PyStackRef_FromPyObjectBorrow(right_o);
            return res;
        }
        PyObject *f = _PyFloat_FromDouble_Fast(dres);
        if (f == NULL) {
            return PyStackRef_NULL;
        }
        return PyStackRef_FromPyObjectSteal(f);
    }
    else if ((oparg == NB_MULTIPLY || oparg == NB_INPLACE_MULTIPLY) &&
             PyFloat_CheckExact(left_o) &&
             PyLong_CheckExact(right_o) && _PyLong_IsCompact((PyLongObject *)right_o))
    {
        double dres = ((PyFloatObject *)left_o)->ob_fval *
                      (double)_PyLong_CompactValue((PyLongObject *)right_o);
        if (PyStackRef_RefcountOnObject(*pleft) &&
            _PyObject_IsUniquelyReferenced(left_o))
        {
            ((PyFloatObject *)left_o)->ob_fval = dres;
            _PyStackRef res = *pleft;
            *pleft = PyStackRef_FromPyObjectBorrow(left_o);
            return res;
        }
        PyObject *f = _PyFloat_FromDouble_Fast(dres);
        if (f == NULL) {
            return PyStackRef_NULL;
        }
        return PyStackRef_FromPyObjectSteal(f);
    }
    PyObject *res_o = d->action(left_o, right_o);
    if (res_o == NULL) {
        return PyStackRef_NULL;
    }
    assert(d->result_type == NULL || Py_TYPE(res_o) == d->result_type);
    assert(!d->result_unique || Py_REFCNT(res_o) == 1 || _Py_IsImmortal(res_o));
    assert(!PyFloat_CheckExact(res_o) || Py_REFCNT(res_o) == 1);
    return PyStackRef_FromPyObjectSteal(res_o);
}

static inline Py_ALWAYS_INLINE _PyStackRef
_PyEval_BinaryOpInPlace(_PyStackRef *plhs, _PyStackRef *prhs,
                        PyObject *lhs_o, PyObject *rhs_o, int oparg)
{
    if ((oparg == NB_TRUE_DIVIDE || oparg == NB_INPLACE_TRUE_DIVIDE) &&
        PyFloat_CheckExact(lhs_o) && PyFloat_CheckExact(rhs_o))
    {
        double rval = ((PyFloatObject *)rhs_o)->ob_fval;
        if (__builtin_expect(rval != 0.0, 1)) {
            double dres = ((PyFloatObject *)lhs_o)->ob_fval / rval;
            if (PyStackRef_RefcountOnObject(*plhs) &&
                _PyObject_IsUniquelyReferenced(lhs_o))
            {
                ((PyFloatObject *)lhs_o)->ob_fval = dres;
                _PyStackRef res = *plhs;
                *plhs = PyStackRef_FromPyObjectBorrow(lhs_o);
                return res;
            }
            if (PyStackRef_RefcountOnObject(*prhs) &&
                _PyObject_IsUniquelyReferenced(rhs_o))
            {
                ((PyFloatObject *)rhs_o)->ob_fval = dres;
                _PyStackRef res = *prhs;
                *prhs = PyStackRef_FromPyObjectBorrow(rhs_o);
                return res;
            }
            PyObject *d = _PyFloat_FromDouble_Fast(dres);
            if (d == NULL) {
                return PyStackRef_NULL;
            }
            return PyStackRef_FromPyObjectSteal(d);
        }
    }
    PyObject *res_o = _PyEval_BinaryOps[oparg](lhs_o, rhs_o);
    if (res_o == NULL) {
        return PyStackRef_NULL;
    }
    return PyStackRef_FromPyObjectSteal(res_o);
}

static inline Py_ALWAYS_INLINE void
_PyEval_FrameClearAndPopFast(PyThreadState *tstate, _PyInterpreterFrame *dying)
{
    if (__builtin_expect(
            dying->owner == FRAME_OWNED_BY_THREAD &&
            FT_ATOMIC_LOAD_PTR_RELAXED(dying->frame_obj) == NULL &&
            dying->f_locals == NULL &&
            (PyObject **)dying != &tstate->datastack_chunk->data[0], 1))
    {
        _PyThreadState_UpdateLastProfiledFrame(tstate, dying, tstate->current_frame);
        assert((PyObject **)dying + _PyFrame_GetCode(dying)->co_framesize ==
               tstate->datastack_top);
        assert(_PyThreadState_GET()->current_frame != dying);
        assert(dying->stackpointer != NULL);
        _PyStackRef *sp = dying->stackpointer;
        _PyStackRef *locals = dying->localsplus;
        dying->stackpointer = locals;
        while (sp > locals) {
            sp--;
            PyStackRef_XCLOSE(*sp);
        }
        PyStackRef_CLOSE(dying->f_funcobj);
        PyStackRef_CLOSE(dying->f_executable);
        assert(tstate->datastack_top >= (PyObject **)dying);
        tstate->datastack_top = (PyObject **)dying;
        return;
    }
    _PyEval_FrameClearAndPop(tstate, dying);
}

static inline Py_ALWAYS_INLINE uint32_t
_PyCode_GetStrictSig(PyCodeObject *co)
{
    uintptr_t extra = (uintptr_t)co->co_extra;
    return (extra & 1) ? (uint32_t)extra : 0;
}

static inline Py_ALWAYS_INLINE int
_PyEval_StrictTagCheck(_PyStackRef ref, uint32_t tag)
{
    PyObject *obj = PyStackRef_AsPyObjectBorrow(ref);
    switch (tag) {
        case 1: return Py_TYPE(obj) != &PyLong_Type;
        case 2: return Py_TYPE(obj) != &PyFloat_Type;
        case 3: return Py_TYPE(obj) != &PyUnicode_Type;
        case 4: return Py_TYPE(obj) != &PyBool_Type;
        case 5: return obj != Py_None;
        default: return 0;
    }
}

static inline const char *
_PyEval_StrictTagName(uint32_t tag)
{
    switch (tag) {
        case 1: return "int";
        case 2: return "float";
        case 3: return "str";
        case 4: return "bool";
        case 5: return "None";
        default: return "object";
    }
}

static inline Py_ALWAYS_INLINE int
_PyEval_StrictParamsCheck(_PyInterpreterFrame *frame)
{
    uint32_t sig = _PyCode_GetStrictSig(_PyFrame_GetCode(frame));
    if (__builtin_expect(sig > 0x1F, 0)) {
        uint32_t param_tags = sig >> 5;
        _PyStackRef *locals = frame->localsplus;
        int i = 0;
        while (param_tags != 0) {
            uint32_t tag = param_tags & 0x7;
            if (tag != 0 && __builtin_expect(_PyEval_StrictTagCheck(locals[i], tag), 0)) {
                return 1;
            }
            param_tags >>= 3;
            i++;
        }
    }
    return 0;
}

static inline void
_PyEval_FormatStrictParamError(PyThreadState *tstate, _PyInterpreterFrame *frame, uint32_t sig)
{
    PyCodeObject *co = _PyFrame_GetCode(frame);
    uint32_t param_tags = sig >> 5;
    _PyStackRef *locals = frame->localsplus;
    int i = 0;
    while (param_tags != 0) {
        uint32_t tag = param_tags & 0x7;
        if (tag != 0 && _PyEval_StrictTagCheck(locals[i], tag)) {
            PyObject *val = PyStackRef_AsPyObjectBorrow(locals[i]);
            PyObject *pname = PyTuple_GET_ITEM(co->co_localsplusnames, i);
            _PyErr_Format(tstate, PyExc_TypeError,
                          "@strict type violation in %U(): argument '%U' expected %s, got %s (%R)",
                          co->co_qualname, pname, _PyEval_StrictTagName(tag),
                          Py_TYPE(val)->tp_name, val);
            return;
        }
        param_tags >>= 3;
        i++;
    }
}

static inline Py_ALWAYS_INLINE int
_PyEval_StrictParamsErrCheck(PyThreadState *tstate, _PyInterpreterFrame *frame)
{
    uint32_t sig = _PyCode_GetStrictSig(_PyFrame_GetCode(frame));
    if (__builtin_expect(sig > 0x1F, 0)) {
        if (__builtin_expect(_PyEval_StrictParamsCheck(frame), 0)) {
            tstate->current_frame = frame;
            _PyEval_FormatStrictParamError(tstate, frame, sig);
            return 1;
        }
    }
    return 0;
}

static inline Py_ALWAYS_INLINE int
_PyEval_StrictReturnCheck(PyThreadState *tstate, _PyInterpreterFrame *frame, _PyStackRef retval)
{
    uint32_t sig = _PyCode_GetStrictSig(_PyFrame_GetCode(frame));
    if (__builtin_expect((sig & 0x1C) != 0, 0)) {
        uint32_t tag = (sig >> 2) & 0x7;
        if (__builtin_expect(_PyEval_StrictTagCheck(retval, tag), 0)) {
            tstate->current_frame = frame;
            PyCodeObject *co = _PyFrame_GetCode(frame);
            PyObject *val = PyStackRef_AsPyObjectBorrow(retval);
            _PyErr_Format(tstate, PyExc_TypeError,
                          "@strict type violation in %U(): return value expected %s, got %s (%R)",
                          co->co_qualname, _PyEval_StrictTagName(tag),
                          Py_TYPE(val)->tp_name, val);
            return 1;
        }
    }
    return 0;
}
