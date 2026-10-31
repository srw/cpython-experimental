# Free-Threaded Tier 2 / JIT, `@strict` / `@struct`, and Auto-Parallel Runtime Architecture

This document describes the architecture, compiler passes, runtime invariants, special cases, and future roadmap of the experimental **Free-Threaded (`--disable-gil`) + Tier 2 Abstract Interpreter + Native AArch64 Copy-and-Patch JIT + Opt-In Sound Gradual Typing (`@strict` / `@struct`) + Auto-Parallel Dispatch** runtime on the `experimental-nogil-jit` branch.

---

## Table of Contents

1. [Design Philosophy & Core Thesis](#1-design-philosophy--core-thesis)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Pillar 1: Zero-Indirection Free-Threaded Runtime (`--disable-gil`)](#3-pillar-1-zero-indirection-free-threaded-runtime---disable-gil)
4. [Pillar 2: Tier 2 Micro-Op Optimizer & Native Copy-and-Patch JIT](#4-pillar-2-tier-2-micro-op-optimizer--native-copy-and-patch-jit)
5. [Pillar 3: Opt-In Sound Gradual Typing (`@strict`, `@struct`, `strict_mode`, `strict_module`)](#5-pillar-3-opt-in-sound-gradual-typing-strict-struct-strict_mode-strict_module)
6. [Pillar 4: C-Level Work-Stealing Auto-Parallel Dispatch Engine](#6-pillar-4-c-level-work-stealing-auto-parallel-dispatch-engine)
7. [Special Cases, Corner Cases, and Deoptimization Invariants](#7-special-cases-corner-cases-and-deoptimization-invariants)
8. [Benchmark Results & Competitive Landscape](#8-benchmark-results--competitive-landscape)
9. [Next Steps & Research Roadmap](#9-next-steps--research-roadmap)
10. [Academic & Systems Publication References](#10-academic--systems-publication-references)

---

## 1. Design Philosophy & Core Thesis

Standard Python performance is bounded by three structural taxes:

1. **Dynamic Dispatch & Guard Tax:** Every bytecode operation (`a + b`, `obj.x`) must verify operand types, check for descriptor overrides, and resolve dunder methods at runtime.
2. **Heap Boxing & Reference-Counting Tax:** Even when types are known, intermediate numbers (`float`, `int`) and short-lived composite objects are heap-allocated, reference-counted, and freed on every loop iteration. Under PEP 703 free-threading (`--disable-gil`), atomic reference-count operations and per-object locking amplify this overhead.
3. **Unsound Type Annotations:** PEP 484 type annotations (`def f(x: int) -> int:`) are erased at runtime and unenforced at call boundaries, meaning a JIT compiler *cannot* trust them to omit type guards.

### Why We Don't Copy PyPy or GraalPy Directly

Tracing JITs like **PyPy** (RPython) and **GraalPy** (Truffle/Graal) rely on **moving generational garbage collectors** where nursery allocation is a pointer bump and escaping virtual objects are materialized lazily during GC or deoptimization.

CPython's C-API contract and PEP 703 free-threaded memory model require **deterministic biased reference counting** (`ob_ref_local` and `ob_ref_shared`), tagged stack references (`_PyStackRef`), and zero-dependency compilation (Copy-and-Patch stencils). Therefore, rather than importing foreign GC abstractions, this runtime adapts the core algorithmic breakthroughs from the last decade of dynamic language research directly to CPython's reference-counted invariants:

- **Ownership-Driven In-Place Box Reuse:** In a biased reference-counting VM, an object owned exclusively by the current thread (`_Py_IsOwnedByCurrentThread(obj) && ob_ref_local == 1 && ob_ref_shared == 0`) *is* a mutable unboxed register storage cell. Mutating its payload in place eliminates both allocation and reference-count traffic.
- **Partial Escape Analysis (PEA) on Linear Micro-Op Traces:** Following Stadler et al. (*PLDI 2014*) and Bolz et al. (*ICOOOLPS 2011*), non-escaping `@struct` instances and tuples allocated inside hot traces have their fields scalar-replaced directly on the abstract interpreter stack.
- **Basic Block Versioning (BBV) via Loop Pre-Header Peeling:** Following Chevalier-Boisvert & Feeley (*OOPSLA 2015*), peeling the first iteration of a loop into a trace pre-header establishes steady-state type and validity proofs for the loop body without repeated per-iteration guards.
- **Opt-In Sound Gradual Typing & Sealed Layout (`@strict` / `@struct`):** Following Cinder's Static Python (*lu et al., 2022*) and Rastogi et al. (*POPL 2015*), enforcing type annotations in C at function and slot boundaries turns user annotations into sound compiler invariants.

---

## 2. High-Level System Architecture

```mermaid
flowchart TD
    subgraph UserCode["Python Source Code"]
        A["Standard Undecorated Python"]
        B["Opt-In @strict / @struct / strict_mode()"]
    end

    subgraph Tier1["Tier 1: Free-Threaded Adaptive Interpreter (Py_GIL_DISABLED)"]
        C["Zero-Indirection Main-Thread TLBC + Worker Cache Inheritance"]
        D["C-Enforced @struct Slot Descriptors (_Py_STRICT_SLOT_*)"]
        E["C-Enforced Function Signatures (_co_strict_sig / _CHECK_FUNCTION_STRICT_ARGS)"]
        F["Unshared-Local Fast Paths (list/dict/set/compact-int/float freelist)"]
    end

    subgraph Tier2["Tier 2: Trace Optimizer (Python/optimizer_analysis.c)"]
        G["1. Abstract Interpretation & Symbolic Type Propagation"]
        H["2. Partial Escape Analysis (eliminate_non_escaping_struct_allocs)"]
        I["3. Loop Pre-Header Guard Hoisting & Peeling (hoist_loop_preheader_guards)"]
        J["4. In-Place Float/Int Chaining & FMA Fusion (mark_inplace_float_*)"]
        K["5. Loop Accumulator Ownership Transfer & _LOAD_ATTR_SLOT_NO_POP"]
        L["6. Multi-Op Kernel Recognition (_ITER_SUM_LIST, _ITER_NBODY_PAIR, etc.)"]
    end

    subgraph NativeJIT["Tier 3: Native AArch64 Copy-and-Patch JIT + Parallel Engine"]
        M["Per-Uop Stencil Cache & Non-Escaping C-Call Lowering"]
        N["Work-Stealing C Dispatch Engine (auto-parallel map/filter/sorted)"]
    end

    A --> C
    B --> D
    B --> E
    C --> F
    D --> G
    E --> G
    F --> G
    G --> H --> I --> J --> K --> L --> M
    F --> N
    M --> N
```

---

## 3. Pillar 1: Zero-Indirection Free-Threaded Runtime (`--disable-gil`)

Under `Py_GIL_DISABLED`, standard CPython pays a single-threaded overhead penalty due to thread-local bytecode (TLBC) indirection, deferred reference counting checks, critical sections on containers, and freelist contention. We eliminate these bottlenecks through five runtime mechanisms:

### 3.1 Zero-Indirection Main-Thread TLBC & Worker Inheritance
- The creating/main thread executes directly out of `co->co_code_adaptive` (`tlbc_index == 0`) without a TLBC table lookup on every frame push.
- When worker threads spawn or steal tasks, they inherit already-specialized adaptive opcodes from the parent thread's TLBC copy rather than warming up from unspecialized `LOAD_ATTR` / `BINARY_OP` instructions.

### 3.2 Unshared-Local Ownership Fast Paths
- Defined in `Include/internal/pycore_object.h` and `Python/ceval_macros.h`:
  - `_Py_IsUnsharedLocal(op)` checks `_Py_IsOwnedByCurrentThread(op) && (op->ob_ref_shared == 0)`.
  - `_Py_IsUniquelyReferencedLocal(op)` checks `_Py_IsOwnedByCurrentThread(op) && (op->ob_ref_local == 1) && (op->ob_ref_shared == 0)`.
- Built-in `list`, `dict`, `set`, and heap-type slot setters check `_Py_IsUnsharedLocal(container)` to bypass `_Py_CRITICAL_SECTION` lock acquisition when a container has never been shared across threads (`ob_ref_shared == 0`).

### 3.3 Unified 60-Bit (`<= 2`-Digit) `PyLong` Freelist & Compact Ints
- Standard CPython only caches 1-digit (`< 2^30`) `PyLong` objects in its freelist. Any integer in `[2^30, 2^60 - 1]` (common in bitboards, hashes, timestamps, and 32-bit bitwise loops) previously hit `_PyObject_Malloc` on every operation.
- `Objects/longobject.c` and `Include/internal/pycore_freelist.h` unify 1- and 2-digit `PyLongObject` allocations into a single `sizeof(PyLongObject) + 2 * sizeof(digit)` slab in the per-thread freelist.
- This allows 1-digit ints to grow into 2-digit ints (and vice versa) **in place** without reallocation.

### 3.4 Deferred Refcounting on `PyCodeObject`
- Enabling `_PyObject_SetDeferredRefcount` on `PyCodeObject` (`Objects/codeobject.c`) eliminates atomic reference count increments/decrements on function code objects during recursive calls and generator/frame creation/teardown.

---

## 4. Pillar 2: Tier 2 Micro-Op Optimizer & Native Copy-and-Patch JIT

When a loop (`JUMP_BACKWARD`) or function entry (`RESUME`) warms up, `_PyOptimizer_Optimize` records a linear trace of micro-ops (`_PyUOpInstruction`) and runs `_Py_uop_analyze_and_optimize` (`Python/optimizer_analysis.c`).

### 4.1 Complete Optimizer Pass Pipeline (`optimize_uops`)

The order of passes in `Python/optimizer_analysis.c` is carefully structured because later passes rely on invariants established by earlier passes, and stack-depth (`sp`) simulations must agree with the exact operand arity of every uop in the trace:

```mermaid
flowchart TD
    P0["0. Pre-Peeling Inlined Frame Cleanup\n(fold_trivial_inlined_calls)"] --> P1
    P1["1. Abstract Interpretation & Type Narrowing\n(optimizer_bytecodes.c -> optimizer_cases.c.h)"] --> P2
    P2["2. Partial Escape Analysis for @struct\n(eliminate_non_escaping_struct_allocs)"] --> P3
    P3["3. Dead Guard & Redundant Check Removal\n(remove_unneeded_uops)"] --> P4
    P4["4. Pre-Peeling In-Place Float Attr Updates (Small Traces)\n(mark_inplace_float_attr_updates when !has_strict_slots || len <= 180)"] --> P5
    P5["5. Loop Pre-Header Peeling & LICM Guard Hoisting\n(hoist_loop_preheader_guards)"] --> P6
    P6["6. Post-Peeling In-Place Float Box Chaining & FMA Fusion\n(mark_inplace_float_allocs + mark_inplace_float_attr_updates)"] --> P7
    P7["7. Loop Accumulator Exclusive Ownership Transfer\n(mark_local_exclusivity + _LOAD_ATTR_SLOT_NO_POP)"] --> P8
    P8["8. Vectorized / Unrolled Loop Kernel Fusion\n(_ITER_SUM_LIST, _ITER_DOT_PRODUCT, _ITER_NBODY_PAIR, etc.)"]
```

### 4.2 Abstract Interpretation & Specialized Lowering (`Python/optimizer_bytecodes.c`)
During abstract interpretation (`_Py_uop_analyze_and_optimize`), each stack reference carries a `JitOptSymbol` tracking:
- Known exact `PyTypeObject *` (`&PyLong_Type`, `&PyFloat_Type`, `&PyUnicode_Type`, or user `@struct` type).
- Compact-int flags (known 1- or 2-digit integer).
- Constant value folding and nullability/truthiness proofs.

Key lowerings added in `optimizer_bytecodes.c`:
- **Float Arithmetic (`+`, `-`, `*`, `/`):** Lowered to `_BINARY_OP_ADD_FLOAT`, `_BINARY_OP_SUBTRACT_FLOAT`, `_BINARY_OP_MULTIPLY_FLOAT`, `_BINARY_OP_TRUEDIV_FLOAT`.
- **Mixed Float-Int Arithmetic (`_BINARY_OP_ARITH_FLOAT_INT`):** When one operand is proven `float` and the other is proven `int` (in either order for `+`, `-`, `*`, `/`), `_BINARY_OP_EXTEND` lowers to `_BINARY_OP_ARITH_FLOAT_INT` (`oparg = 0..7` encoding operation and operand order), calling `_PyFloat_MixedArithInPlace` without `PyNumber_*` C-API dispatch.
- **1- & 2-Digit Compact-Int Bitwise & DivMod (`_BINARY_OP_BITWISE_INT`):** When both operands are proven `int`, `_BINARY_OP_EXTEND` lowers `&`, `|`, `^`, `>>`, `//`, and `%` into `_BINARY_OP_BITWISE_INT`, which extracts up to 2 digits (`60` bits) as `int64_t` and mutates uniquely owned result boxes in place via `_PyCompactLong_BitwiseInPlace`.

### 4.3 Partial Escape Analysis / Scalar Replacement (`eliminate_non_escaping_struct_allocs`)
Located in `Python/optimizer_analysis.c`. When an `@struct` class is instantiated inside a hot loop (`Item(i, float(i) * 1.5, i & 3)`), Tier 2 inlines `__init__`:

```text
# Before PEA:
_ALLOCATE_OBJECT          (Type = Item)
_CREATE_INIT_FRAME        (pushes self + arg0, arg1, arg2)
_PUSH_FRAME
_LOAD_FAST_BORROW 0       (self)
_LOAD_FAST_BORROW 1       (arg0)
_STORE_ATTR_SLOT 0        (self.sku = arg0)
...
_RETURN_VALUE             (pops __init__ frame)
_MAKE_HEAP_SAFE           (publishes self)
_STORE_FAST 5             (item = self)
_LOAD_FAST_BORROW 5       (item)
_LOAD_ATTR_SLOT 1         (item.price)
```

`eliminate_non_escaping_struct_allocs` performs a forward dataflow and escape analysis pass over each `_ALLOCATE_OBJECT` $\rightarrow$ `_CREATE_INIT_FRAME` pair:
1. **Escape Verification:** Tracks `self` across the inlined `__init__` frame and across caller local slots (`_STORE_FAST` / `_LOAD_FAST_BORROW`). If `self` is passed to any external call, stored into a container/attribute, returned, or live at a side-exit after escaping to a caller local, PEA conservatively aborts for that allocation site.
2. **Scalar Replacement:** Maps each `slot_offset` written by `_STORE_ATTR_SLOT` in `__init__` to a fresh temporary local slot (`next_temp_local = max_local_used + 1`) in the caller frame.
3. **Read Forwarding:** Rewrites every downstream `_LOAD_ATTR_SLOT(slot_offset)` on `self` into `_POP_TOP` + `_LOAD_FAST_BORROW(temp_local)` (or `_LOAD_FAST(temp_local)`).
4. **Allocation Elimination:** Replaces `_ALLOCATE_OBJECT`, `_CREATE_INIT_FRAME`, `_PUSH_FRAME`, `_STORE_ATTR_SLOT`, `_RETURN_VALUE`, and `_MAKE_HEAP_SAFE` with `_NOP` / `_POP_TOP`, reducing heap allocations to **zero**.

### 4.4 Loop Pre-Header Peeling & LICM (`hoist_loop_preheader_guards`)
In dynamic loops, the first iteration resolves local slot initialization (`_LOAD_FAST_CHECK`), type checks (`_GUARD_TYPE_VERSION`), and container validity (`_GUARD_LIST_VALID`).
- `hoist_loop_preheader_guards` peels up to `64` loop-body uops into a pre-header before `_JUMP_TO_TOP` and rewrites steady-state `_LOAD_FAST` instructions in the loop body to `_LOAD_FAST_BORROW` whenever the local slot is proven unmutated or reassigned with the same type.
- Redundant `_GUARD_TYPE_VERSION_LOCKED` and `_CHECK_VALIDITY` instructions on loop-invariant receivers (`self`, outer containers) are hoisted out of the steady-state loop.

### 4.5 In-Place Box Reuse, FMA Fusion, and `_LOAD_ATTR_SLOT_NO_POP`
In floating-point kernels (such as `nbody_float`, polynomial evaluation, and physics updates):
1. **Expression-Tree Box Reuse (`mark_inplace_float_allocs`):** Rewrites chains of `_BINARY_OP_ADD_FLOAT`, `_BINARY_OP_SUBTRACT_FLOAT`, `_BINARY_OP_MULTIPLY_FLOAT`, and `_BINARY_OP_TRUEDIV_FLOAT` into `*_INPLACE` variants that mutate the left/right operand's `PyFloatObject` `ob_fval` directly when `ob_ref_local == 1 && ob_ref_shared == 0`.
2. **Fused Multiply-Add (`_BINARY_OP_FMA_FLOAT_INPLACE`):** Fuses `_BINARY_OP_MULTIPLY_FLOAT[_INPLACE]` followed by `_BINARY_OP_ADD_FLOAT[_INPLACE]` or `_BINARY_OP_SUBTRACT_FLOAT[_INPLACE]` into a single 3-operand uop executing hardware `fma(a, b, c)` (`a * b + c` or `c ± a * b`, parameterized by `oparg` mode `0..2`).
3. **In-Place Slot Updates (`_STORE_ATTR_SLOT_FLOAT_INPLACE` & `_LOAD_ATTR_SLOT_NO_POP`):**
   - When an `@struct` float slot is updated (`b.vx = b.vx - dx * b2.mass`), `mark_inplace_float_attr_updates` rewrites `_STORE_ATTR_SLOT` to `_STORE_ATTR_SLOT_FLOAT_INPLACE`, which overwrites the existing `PyFloatObject` inside the instance slot if uniquely owned.
   - When the receiver object on the stack is only consumed by `_LOAD_ATTR_SLOT` followed by `_POP_TOP` or another slot load, `_LOAD_ATTR_SLOT` is rewritten to `_LOAD_ATTR_SLOT_NO_POP` (borrowing the slot reference without incrementing/decrementing the receiver's reference count).

### 4.6 Non-Escaping C Helper Lowering in Copy-and-Patch (`Tools/cases_generator/analyzer.py`)
In CPython's Copy-and-Patch generator, any uop that calls an arbitrary C function is normally marked `escaping`, forcing the code generator to spill `stack_pointer` and reload frame state before and after the call.
- By designing pure register/box-mutating helpers that cannot trigger GC, arbitrary `__del__` finalizers, or Python frame re-entry (`_PyCompactLong_BitwiseInPlace`, `_PyFloat_MixedArithInPlace`, `_PyList_GetItemRefNoLock`, `_PyObject_CheckStrictSlotType`, etc.) and registering them in `NON_ESCAPING_FUNCTIONS` in `Tools/cases_generator/analyzer.py`, the JIT compiles these uops with zero frame-spill overhead.

---

## 5. Pillar 3: Opt-In Sound Gradual Typing (`@strict`, `@struct`, `strict_mode`, `strict_module`)

Implemented across `Lib/strict.py`, `Objects/typeobject.c`, `Python/structmember.c`, `Objects/funcobject.c`, `Python/bytecodes.c`, and `Python/optimizer_bytecodes.c`.

### 5.1 `@struct`: Sealed, Slot-Backed, C-Enforced Classes
Applying `@struct` (or `@strict` to a class with field annotations):
1. **Synthesizes `__slots__`:** Eliminates per-instance `__dict__` allocation and hash-table lookups.
2. **Binds C-Level `_Py_STRICT_SLOT_*` Flags on `PyMemberDef`:**
   - During `type_new` (`Objects/typeobject.c`), `__strict_slot_types__` maps field annotations (`int`, `float`, `str`, `bool`) to bitflags (`_Py_STRICT_SLOT_INT`, `_Py_STRICT_SLOT_FLOAT`, `_Py_STRICT_SLOT_STR`, `_Py_STRICT_SLOT_BOOL`) in `PyMemberDef.flags` (`Include/descrobject.h`).
   - In `PyMember_SetOne` (`Python/structmember.c`) and the Tier 1/Tier 2 `_STORE_ATTR_SLOT` fast paths, `_PyObject_CheckStrictSlotType` validates the stored value in C in a single pointer comparison (`Py_TYPE(v) == &PyLong_Type` and `Py_TYPE(v) != &PyBool_Type` for `int`). Any invalid assignment—even from uninstrumented third-party code—immediately raises `TypeError`.
3. **Seals the Type (`Py_TPFLAGS_IMMUTABLETYPE`):** Prevents monkey-patching methods/descriptors after class creation or mutating `obj.__class__`.
4. **Tier 2 Unchecked Type Seeding:** Because the C slot setter guarantees the slot's runtime type, `_LOAD_ATTR_SLOT` in `Python/optimizer_bytecodes.c` inspects `PyMemberDef.flags` on immutable `@struct` types and immediately marks the loaded symbol as a known `int` (`sym_set_compact_int`), `float`, `str`, or `bool` **without emitting any runtime type guard**.

### 5.2 `@strict`: C-Level Signature Bitmask (`_co_strict_sig`)
When `@strict` decorates a function with primitive parameter/return annotations (`int`, `float`, `str`, `bool`, `None`) and up to 9 positional/keyword parameters:
- `_try_enable_c_strict_sig` (`Lib/strict.py`) encodes the expected types into a 64-bit integer `co->_co_strict_sig` on `PyCodeObject`:
  - Bit `0` (`0x1`): Strict signature active.
  - Bit `1` (`0x2`): Checked parameter local slots are **never mutated** inside the function body (verified by scanning `co_code` for `STORE_FAST`, `DELETE_FAST`, `STORE_FAST_LOAD_FAST`, `STORE_FAST_STORE_FAST`, `SWAP_FAST`).
  - Bits `2..4`: Return type tag (`1=int, 2=float, 3=str, 4=bool, 5=None`).
  - Bits `5 + 3*i .. 7 + 3*i`: Parameter `i` type tag.
- At runtime, `_CHECK_FUNCTION_STRICT_ARGS` validates arguments on function entry in C (zero wrapper frame overhead!).
- In Tier 2 (`optimizer_bytecodes.c`), if bit `0x2` is set on `co->_co_strict_sig`, every `_LOAD_FAST` / `_LOAD_FAST_BORROW` of a checked parameter slot immediately seeds the abstract interpreter with the exact static type with **zero runtime guards**.

### 5.3 Third-Party & Legacy Boundary Enforcement (`strict_mode` & `strict_module`)
- **`with strict_mode(require_annotations=..., check_modules=...):`** Uses PEP 669 `sys.monitoring` (`PY_START` and `PY_RETURN` events) to intercept calls into uninstrumented third-party or legacy functions inside the `with` block, enforcing their annotations (or raising `TypeError` if annotations are missing when `require_annotations=True`).
- **`strict_module(mod, recursive=True)`:** Explicitly wraps functions and classes in an imported module with `@strict`.

---

## 6. Pillar 4: C-Level Work-Stealing Auto-Parallel Dispatch Engine

Located in `Python/sysmodule.c` and `Python/bltinmodule.c`:
- Provides a persistent C-level work-stealing thread pool (`sys._parallel_map`, auto-parallel `map()`, `filter()`, `sorted()`, `itertools.starmap()`).
- **Adaptive Cost-Guided Dispatch Threshold:** Profiles the per-element bytecode cost on the first few items and only dispatches to worker threads when `N * per_item_cost` exceeds the thread wakeup threshold, ensuring zero regression on small inputs (`1T` monotonic fast path).
- **Lock-Free `functools.lru_cache`:** Uses optimistic atomic hits under `Py_GIL_DISABLED` so read-heavy memoized functions scale linearly across threads.

---

## 7. Special Cases, Corner Cases, and Deoptimization Invariants

Working inside a free-threaded, reference-counted C VM with a linear uop trace optimizer introduces several subtle corner cases that must be preserved for soundness:

### 7.1 Biased Reference Counting vs. In-Place Box Mutation
- **The Hazard:** If a `PyFloatObject` or `PyLongObject` is referenced by another local variable, a container, another thread (`ob_ref_shared != 0`), or is an immortal singleton (such as `_PyLong_GetZero()` or small ints `[-5, 256]`), mutating `ob_fval` or `long_value.ob_digit` in place silently corrupts aliased values!
- **The Invariant:** Every in-place helper (`_PyCompactLong_BitwiseInPlace`, `_PyFloat_MixedArithInPlace`, `_BINARY_OP_*_FLOAT_INPLACE`, `_STORE_ATTR_SLOT_FLOAT_INPLACE`) checks:
  ```c
  _Py_IsOwnedByCurrentThread(obj) && (obj->ob_ref_local == 1) && (obj->ob_ref_shared == 0)
  ```
  Note that immortal objects have `ob_ref_local == _Py_IMMORTAL_INITIAL_REFCNT` (`0xFFFFFFFF`), so `ob_ref_local == 1` automatically rejects all immortal singletons and small-int caches in a single comparison.

### 7.2 `bool` Is a Subclass of `int` in Python, But Has Distinct Representation
- In Python, `isinstance(True, int)` is `True`, and `PyBool_Type` is a subtype of `PyLong_Type`.
- However, in `@strict` / `@struct` type checking and compact-int in-place mutation:
  1. `@strict` enforces `type(val) is int` (`Py_TYPE(v) == &PyLong_Type`), rejecting `bool` where `int` is expected.
  2. `Py_True` and `Py_False` are immortal singletons; `Py_TYPE(v) == &PyLong_Type` + `ob_ref_local == 1` guarantees `True`/`False` are never mutated in place.

### 7.3 Stack Depth (`sp`) Tracking Across 3-Operand FMA and Pre-Header Peeling
- `hoist_loop_preheader_guards` simulates the operand stack (`sp`) across the trace to identify which `_LOAD_FAST` instructions correspond to top-level statement boundaries vs. mid-expression operands.
- Standard binary ops pop 2 and push 1 (`sp -= 1`). However:
  - `_BINARY_OP_FMA_FLOAT_INPLACE` pops **3** operands (`a`, `b`, `c`) and pushes **1** (`sp -= 2`).
  - `_UNARY_NEGATIVE_FLOAT_INPLACE` pops **1** and pushes **1** (`sp += 0`).
  - `_LOAD_ATTR_SLOT_NO_POP` peeks the receiver (`sp` unchanged for receiver) and pushes the attribute (`sp += 1`).
- **Critical Rule:** `hoist_loop_preheader_guards` explicitly tracks `_BINARY_OP_FMA_FLOAT_INPLACE` (`sp -= 2`) and `_UNARY_NEGATIVE_FLOAT_INPLACE` (`sp += 0`), AND `_LOAD_ATTR_SLOT_NO_POP` replacement runs *after* `hoist_loop_preheader_guards` (or is skipped before peeling on large `@struct` traces `length > 180`) so `sp` simulation remains exact.

### 7.4 Partial Escape Analysis (`eliminate_non_escaping_struct_allocs`) Safety Conditions
Scalar replacement of `@struct` allocations must abort for a candidate `_ALLOCATE_OBJECT` if any of the following hold:
1. **Unpaired Frame Push/Return:** `_CREATE_INIT_FRAME` does not have a matching `_RETURN_VALUE` + `_MAKE_HEAP_SAFE` within the trace before `_JUMP_TO_TOP` or `_EXIT_TRACE`.
2. **Side-Exits Inside `__init__`:** Any instruction inside the inlined `__init__` body (other than a hoisted/proven-eliminated guard) can `_DEOPT` or `_EXIT_TRACE` after `_PUSH_FRAME`, because reconstructing a half-initialized virtual frame at a side-exit would require materializing the elided object.
3. **Uninitialized or Conditional Slot Reads:** A downstream `_LOAD_ATTR_SLOT(offset)` reads a slot offset that was not unconditionally written by `_STORE_ATTR_SLOT(offset)` in `__init__`.
4. **Object Identity / Escape:** The instance symbol is consumed by any instruction other than `_STORE_FAST`, `_LOAD_FAST_BORROW`, `_LOAD_FAST`, `_POP_TOP`, or `_LOAD_ATTR_SLOT`, or remains live in a caller local variable at `_JUMP_TO_TOP` / `_EXIT_TRACE`.

### 7.5 Python Floor Division (`//`) & Modulo (`%`) vs. C99 Signed Division
- C99 `/` and `%` truncate toward zero (`-7 / 2 == -3`, `-7 % 2 == -1`), whereas Python `//` and `%` floor toward $-\infty$ (`-7 // 2 == -4`, `-7 % 2 == 1`).
- In `_PyCompactLong_BitwiseInPlace` (`Python/ceval_macros.h`), `oparg == 4` (`//`) and `oparg == 5` (`%`) compute C division/remainder first and apply the sign-mismatch correction:
  ```c
  int64_t q = a / b;
  int64_t r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) {
      q -= 1;
      r += b;
  }
  ```
- Furthermore, when `b == 0` (`ZeroDivisionError`) or `b < 0` in `>>` (`ValueError: negative shift count`), `_PyCompactLong_BitwiseInPlace` delegates cleanly to `PyNumber_FloorDivide` / `PyNumber_Remainder` / `PyNumber_Rshift` so exact Python exceptions and tracebacks are preserved.

---

## 8. Benchmark Results & Competitive Landscape

Reproducible via `Tools/benchmarks/bench_strict_duplication.py` on Apple Silicon (M-series, AArch64, `--disable-gil --enable-experimental-jit`):

| Workload | Tier 1 (`PYTHON_JIT=0`) | Undecorated JIT | `@strict` / `@struct` JIT | Speedup vs. Tier 1 | Speedup vs. Undecorated JIT | Static C# (`.NET 10`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dataclass_inst`** | `123.56 ms` | `26.75 ms` | **`4.80 ms`** | **`25.74x`** | **`5.57x`** | `1.42 ms` |
| **`object_attr_churn`** | `135.20 ms` | `48.52 ms` | **`10.17 ms`** | **`13.29x`** | **`4.77x`** | `0.62 ms` |
| **`parallel_4threads` (4T)** | `248.10 ms` | `79.95 ms` | **`19.67 ms`** | **`12.61x`** | **`4.06x`** | `29.27 ms` |
| **`nbody_float`** | `376.82 ms` | `185.01 ms` | **`40.24 ms`** | **`9.36x`** | **`4.60x`** | `5.86 ms` |
| **`attr_loop`** | `74.80 ms` | `35.87 ms` | **`11.23 ms`** | **`6.66x`** | **`3.19x`** | `3.58 ms` |
| **`fib_iter`** | `59.42 ms` | `22.97 ms` | **`15.46 ms`** | **`3.84x`** | **`1.49x`** | `3.57 ms` |
| **`call_overhead`** | `110.27 ms` | `49.70 ms` | **`33.17 ms`** | **`3.32x`** | **`1.50x`** | `4.19 ms` |
| **`mixed_dispatch`** | `94.77 ms` | `34.23 ms` | **`33.03 ms`** | **`2.87x`** | **`1.04x`** | `36.32 ms` |
| **`arith_loop`** | `126.44 ms` | `78.59 ms` | **`51.40 ms`** | **`2.46x`** | **`1.53x`** | `28.59 ms` |
| **`list_comp`** | `111.79 ms` | `59.17 ms` | **`57.40 ms`** | **`1.95x`** | **`1.03x`** | `40.48 ms` |
| **`dict_Comprehension`** | `95.55 ms` | `84.88 ms` | **`85.40 ms`** | **`1.12x`** | `0.99x` | `132.49 ms` |
| **`string_Builder`** | `114.23 ms` | `102.77 ms` | **`102.77 ms`** | **`1.11x`** | `1.00x` | `67.38 ms` |
| **Geometric Mean (11 core)** | `122.90 ms` | `56.71 ms` | **`30.22 ms`** | **`4.06x`** | **`1.88x`** | `12.68 ms` |

---

## 9. Next Steps & Research Roadmap

To close the remaining `2.4x` geometric-mean gap between `@strict`/`@struct` Python (`30.22 ms`) and static AOT/JIT languages like C# `.NET 10` (`12.68 ms`), the next engineering milestones are:

1. **Unboxed C-Value Slots in `@struct` (`_Py_UNBOXED_SLOT_DOUBLE` / `_Py_UNBOXED_SLOT_INT64`):**
   - Currently, `@struct` instances store `PyObject *` pointers in their slots and mutate `PyFloatObject` / `PyLongObject` boxes in place.
   - Storing raw `double` (`8` bytes) and `int64_t` (`8` bytes) directly inside the `@struct` instance memory (like `T_DOUBLE` / `T_LONGLONG` in `PyMemberDef`, boxed lazily only when accessed from untyped Tier 1 code) will eliminate pointer indirection (`b->slot_ptr->ob_fval` $\rightarrow$ `b->raw_fval`) and bring `nbody_float` (`40.24 ms`) and `object_attr_churn` (`10.17 ms`) down to native C/C# speeds (`~5 ms` and `~1 ms`).
2. **True Register-Carried Unboxed Scalars Across Copy-and-Patch Stencils:**
   - Introducing dedicated unboxed `double` and `int64_t` scratch registers in the JIT calling convention (`Tools/jit/template.c`) so `_LOAD_FAST_UNBOXED_FLOAT` / `_BINARY_OP_ADD_UNBOXED_FLOAT` operate solely in hardware FP/SIMD registers (`d0`–`d7`) without writing back to `PyFloatObject->ob_fval` until loop exit or side-exit deoptimization.
3. **Typed Contiguous Homogeneous Lists (`strict.Array[int]` / `strict.Array[float]`):**
   - Adding an opt-in typed backing store to `PyListObject` (similar to PyPy's *Storage Strategies* by Bolz et al., *2013*) where a `list` containing only `int` or `float` stores raw `int64_t[]` or `double[]` in `ob_item`, auto-vectorizing `list_comp` and array kernels with AArch64 NEON SIMD instructions.
4. **Cross-Iteration Partial Escape Analysis (Loop-Carried Virtual `@struct`s):**
   - Extending `eliminate_non_escaping_struct_allocs` so `@struct` instances stored into a loop-carried local variable (`state = State(state.x + 1, state.y * 2)`) are also scalar-replaced across the `_JUMP_TO_TOP` back-edge, materializing the heap object only at `_EXIT_TRACE`.
5. **Deoptimization Materialization Metadata (`_PyExitData` Virtual Frame Reconstruction):**
   - Recording virtualized `@struct` field mappings inside `_PyExitData` so even allocations that contain cold side-exits inside their lifetime can be scalar-replaced along the hot path and lazily allocated only if the side-exit is actually taken.

---

## 10. Academic & Systems Publication References

The compiler passes, memory model optimizations, and sound gradual typing primitives in this runtime draw directly from peer-reviewed literature across dynamic language JITs, reference-counting garbage collection, and type systems:

### Copy-and-Patch & Template JIT Compilation
1. **Xu, H., & Kjolstad, F. (2021).** *Copy-and-Patch Compilation: A fast compilation algorithm for high-level languages and bytecode.* Proceedings of the ACM on Programming Languages, **5**(OOPSLA), Article 136, 1–30. [https://doi.org/10.1145/3485513](https://doi.org/10.1145/3485513) ([arXiv:2011.13127](https://arxiv.org/abs/2011.13127))
   - **Application in this codebase:** Foundation of CPython's stencil compiler (`Python/jit.c`, `Tools/jit/`), extended here with a per-uop stencil cache, free-threaded `Py_GIL_DISABLED` support, and `NON_ESCAPING_FUNCTIONS` call-spill elimination (`Tools/cases_generator/analyzer.py`).
2. **Brandt, B. (2023–2024).** *PEP 744 – JIT Compilation.* Python Enhancement Proposals. [https://peps.python.org/pep-0744/](https://peps.python.org/pep-0744/)

### Partial Escape Analysis & Allocation Removal in Tracing JITs
3. **Stadler, L., Würthinger, T., & Mössenböck, H. (2014).** *Partial Escape Analysis and Scalar Replacement for Java.* In Proceedings of the Annual IEEE/ACM International Symposium on Code Generation and Optimization (**CGO '14**), pp. 165–174. [https://doi.org/10.1145/2544137.2544157](https://doi.org/10.1145/2544137.2544157)
   - **Application in this codebase:** Directly informs `eliminate_non_escaping_struct_allocs` (`Python/optimizer_analysis.c`), which scalar-replaces non-escaping `@struct` instances and forwards field writes from inlined `__init__` frames directly to downstream `_LOAD_ATTR_SLOT` reads (`5.01x` speedup on `dataclass_inst`).
4. **Bolz, C. F., Cuni, A., Fijałkowski, M., Leuschel, M., Pedroni, S., & Rigo, A. (2011).** *Allocation Removal by Partial Evaluation in a Tracing JIT.* In Proceedings of the 20th ACM SIGPLAN Workshop on Partial Evaluation and Program Manipulation (**PEPM '11**), pp. 43–52. [https://doi.org/10.1145/1929501.1929508](https://doi.org/10.1145/1929501.1929508)
   - **Application in this codebase:** Informs tuple constant/variable scalar replacement and linear-trace object virtualization without building a full sea-of-nodes SSA graph.

### Basic Block Versioning, Type Specialization, & Guard Hoisting
5. **Chevalier-Boisvert, M., & Feeley, M. (2015).** *Simple and Effective Type Check Removal through Lazy Basic Block Versioning.* In 29th European Conference on Object-Oriented Programming (**ECOOP 2015**), LIPIcs Vol. 37, pp. 101–123. [https://doi.org/10.4230/LIPIcs.ECOOP.2015.101](https://doi.org/10.4230/LIPIcs.ECOOP.2015.101) ([arXiv:1411.0352](https://arxiv.org/abs/1411.0352))
   - **Application in this codebase:** Informs `hoist_loop_preheader_guards` (`Python/optimizer_analysis.c`), which peels the first loop iteration into a trace pre-header to establish steady-state type and non-nullness versions (`_LOAD_FAST` $\rightarrow$ `_LOAD_FAST_BORROW`), eliminating per-iteration type guards inside the loop body.
6. **Shannon, M., et al. (2021).** *PEP 659 – Specializing Adaptive Interpreter.* Python Enhancement Proposals. [https://peps.python.org/pep-0659/](https://peps.python.org/pep-0659/)
7. **Würthinger, T., Wimmer, C., Wöß, A., Stadler, L., Duboscq, G., Humer, C., Richards, G., Simon, D., & Wolczko, M. (2013).** *One VM to Rule Them All.* In Proceedings of the 2013 ACM International Symposium on New Ideas, New Paradigms, and Reflections on Programming & Software (**Onward! 2013**), pp. 187–204. [https://doi.org/10.1145/2509578.2509581](https://doi.org/10.1145/2509578.2509581)

### Sound Gradual Typing, Sealed Layouts, & Static Python
8. **Lu, K., Gross, C., Maher, M., et al. (Meta Engineering / Instagram, 2021–2023).** *Cinder: Meta's Internal Performance-Oriented Production Version of CPython (Static Python & Shadowcode).* [https://github.com/facebookincubator/cinder](https://github.com/facebookincubator/cinder)
   - **Application in this codebase:** Inspires the `@struct` and `@strict` design (`Lib/strict.py`, `Objects/typeobject.c`, `Python/structmember.c`, `Objects/funcobject.c`), using C-enforced slot descriptors (`_Py_STRICT_SLOT_*`), `Py_TPFLAGS_IMMUTABLETYPE` class sealing, and 64-bit compact function signature bitmasks (`_co_strict_sig`) to eliminate type guards in Tier 2.
9. **Rastogi, A., Swamy, N., Fournet, C., Bierman, G., & Vekris, P. (2015).** *Safe & Efficient Gradual Typing for TypeScript.* In Proceedings of the 42nd Annual ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (**POPL '15**), pp. 167–180. [https://doi.org/10.1145/2676726.2676971](https://doi.org/10.1145/2676726.2676971)
   - **Application in this codebase:** Establishes the theoretical foundation that gradual type annotations only yield compiler speedups when enforced soundly at boundary crossings (`_CHECK_FUNCTION_STRICT_ARGS`, `_PyObject_CheckStrictSlotType`, and `strict_mode()`).
10. **Vitousek, M. M., Swords, C., & Siek, J. G. (2017).** *Big Types in Little Runtime: Open-World Soundness and Collaborative Blame for Gradual Type Systems.* In Proceedings of the 44th ACM SIGPLAN Symposium on Principles of Programming Languages (**POPL '17**), pp. 762–774. [https://doi.org/10.1145/3009837.3009849](https://doi.org/10.1145/3009837.3009849)
    - **Application in this codebase:** Informs the open-world third-party boundary enforcement in `strict_mode()` (via PEP 669 `sys.monitoring`) and `strict_module()`.
11. **Lehtosalo, J., Sullivan, M. J., et al. (2019–2024).** *Mypyc: Compiling Type-Annotated Python to C Extensions.* [https://github.com/mypyc/mypyc](https://github.com/mypyc/mypyc)

### Free-Threading, Biased Reference Counting, & In-Place Reuse (`Perceus`)
12. **Gross, S. (2023).** *PEP 703 – Making the Global Interpreter Lock Optional in CPython.* Python Enhancement Proposals. [https://peps.python.org/pep-0703/](https://peps.python.org/pep-0703/)
13. **Choi, J., Shull, T., & Torrellas, J. (2018).** *Biased Reference Counting: Minimizing Atomic Operations in Garbage Collection.* In Proceedings of the 27th International Conference on Parallel Architectures and Compilation Techniques (**PACT '18**), Article 35, 1–12. [https://doi.org/10.1145/3243176.3243195](https://doi.org/10.1145/3243176.3243195)
    - **Application in this codebase:** Exploited by `_Py_IsUnsharedLocal` and `_Py_IsUniquelyReferencedLocal` (`ob_ref_local == 1 && ob_ref_shared == 0`) to elide per-object critical sections and enable thread-safe in-place box reuse without atomic instructions.
14. **Reinking, A., Xie, N., de Moura, L., & Leijen, D. (2021).** *Perceus: Garbage Free Reference Counting with Reuse.* In Proceedings of the 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementation (**PLDI 2021**), pp. 96–111. [https://doi.org/10.1145/3453483.3454032](https://doi.org/10.1145/3453483.3454032)
    - **Application in this codebase:** Directly inspires our reference-uniqueness in-place mutation passes (`mark_inplace_float_allocs`, `mark_inplace_float_attr_updates`, `mark_local_exclusivity`, `_PyCompactLong_BitwiseInPlace`, `_PyFloat_MixedArithInPlace`), turning uniquely-referenced `PyFloatObject` and `PyLongObject` heap cells into zero-allocation mutable registers.

### Storage Strategies & Domain-Specific Native Compilation
15. **Bolz, C. F., Diekmann, L., & Tratt, L. (2013).** *Storage Strategies for Collections in Dynamically Typed Languages.* In Proceedings of the 2013 ACM SIGPLAN International Conference on Object Oriented Programming Systems Languages & Applications (**OOPSLA '13**), pp. 167–182. [https://doi.org/10.1145/2509136.2509531](https://doi.org/10.1145/2509136.2509531)
16. **Shajii, A., Numanagić, I., Baghdadi, R., Berger, B., & Amarasinghe, S. (2023).** *Codon: A Compiler for High-Performance Pythonic Applications and DSLs.* In Proceedings of the 32nd ACM SIGPLAN International Conference on Compiler Construction (**CC 2023**), pp. 191–202. [https://doi.org/10.1145/3578360.3580275](https://doi.org/10.1145/3578360.3580275)

### Summary Mapping: Literature $\rightarrow$ Codebase Implementation

| Publication / Concept | Primary Codebase Files | Key Symbols / Uops |
| :--- | :--- | :--- |
| **Stadler et al. (CGO '14)** & **Bolz et al. (PEPM '11)**<br>*Partial Escape Analysis & Scalar Replacement* | `Python/optimizer_analysis.c` | `eliminate_non_escaping_struct_allocs` |
| **Reinking et al. (PLDI '21)** & **Choi et al. (PACT '18)**<br>*Reference-Unique In-Place Reuse & Biased RC* | `Python/optimizer_analysis.c`<br>`Python/ceval_macros.h`<br>`Python/bytecodes.c` | `mark_inplace_float_allocs`<br>`mark_inplace_float_attr_updates`<br>`_PyCompactLong_BitwiseInPlace`<br>`_PyFloat_MixedArithInPlace`<br>`_BINARY_OP_FMA_FLOAT_INPLACE` |
| **Chevalier-Boisvert & Feeley (ECOOP '15)**<br>*Lazy Basic Block Versioning / Loop Peeling* | `Python/optimizer_analysis.c` | `hoist_loop_preheader_guards`<br>`mark_local_exclusivity` |
| **Lu et al. (Cinder, 2022)** & **Rastogi et al. (POPL '15)**<br>*Sound Gradual Typing & Sealed Slots* | `Lib/strict.py`<br>`Objects/typeobject.c`<br>`Python/structmember.c`<br>`Objects/funcobject.c`<br>`Python/optimizer_bytecodes.c` | `@struct`, `@strict`<br>`_Py_STRICT_SLOT_*`<br>`_co_strict_sig`<br>`_CHECK_FUNCTION_STRICT_ARGS`<br>`_LOAD_ATTR_SLOT_NO_POP` |
| **Vitousek et al. (POPL '17)**<br>*Open-World Boundary Enforcement* | `Lib/strict.py` | `strict_mode()`, `strict_module()` |
| **Xu & Kjolstad (OOPSLA '21)**<br>*Copy-and-Patch Stencil Compilation* | `Python/jit.c`<br>`Tools/cases_generator/analyzer.py` | `_PyJIT_Compile`<br>`NON_ESCAPING_FUNCTIONS` |

