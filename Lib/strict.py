"""Opt-in strict runtime typing, sealed slot-backed structs, and explicit
third-party boundary enforcement (`@struct`, `@strict`, `strict_module`, `strict_mode`).
"""

import contextlib
import functools
import inspect
import sys
import types

__all__ = [
    "struct",
    "strict",
    "strict_module",
    "strict_mode",
]


def _matches_type(val, expected):
    if expected is inspect.Parameter.empty or expected is object or expected is typing_any():
        return True
    if expected is None or expected is type(None):
        return val is None
    if expected is int:
        return type(val) is int
    if expected is float:
        return type(val) is float
    if expected is str:
        return type(val) is str
    if expected is bool:
        return type(val) is bool
    origin = getattr(expected, "__origin__", None)
    if origin is not None:
        args = getattr(expected, "__args__", ())
        if origin is list:
            if not isinstance(val, list):
                return False
            return True if not args else all(_matches_type(x, args[0]) for x in val)
        if origin is dict:
            if not isinstance(val, dict):
                return False
            return True if len(args) < 2 else all(
                _matches_type(k, args[0]) and _matches_type(v, args[1])
                for k, v in val.items()
            )
        if origin is tuple:
            if not isinstance(val, tuple):
                return False
            if len(args) == 2 and args[1] is Ellipsis:
                return all(_matches_type(x, args[0]) for x in val)
            return len(val) == len(args) and all(
                _matches_type(x, t) for x, t in zip(val, args)
            )
        if not isinstance( origin, type):
            # typing.Union
            return any(_matches_type(val, t) for t in args)
        return isinstance(val, origin)
    if isinstance(expected, types.UnionType):
        return any(_matches_type(val, t) for t in expected.__args__)
    if isinstance(expected, type):
        return isinstance(val, expected)
    return True


def typing_any():
    mod = sys.modules.get("typing")
    return getattr(mod, "Any", None) if mod is not None else None


def _check_value(val, expected, func_name, label):
    if not _matches_type(val, expected):
        exp_name = getattr(expected, "__name__", repr(expected))
        raise TypeError(
            f"@strict type violation in {func_name}(): {label} expected {exp_name}, "
            f"got {type(val).__name__} ({val!r})"
        )


def struct(cls=None, /, *, freeze=True, auto_init=True):
    """Compile a Python class with type annotations into a sealed, slot-backed, C-enforced `@struct`.

    - Synthesizes `__slots__` from class annotations (eliminating per-instance `__dict__`).
    - Sets C-level `_Py_STRICT_SLOT_*` descriptor flags on `PyMemberDef` so any attribute
      store (even from uninstrumented third-party or legacy libraries) is type-checked in C
      and raises `TypeError` on mismatch.
    - Propagates static slot types directly into the Tier 2 / JIT abstract interpreter
      (`_LOAD_ATTR_SLOT`), eliminating runtime type guards in hot loops.
    - Seals the class with `Py_TPFLAGS_IMMUTABLETYPE` (`__strict_freeze__ = True`) so neither
      the class nor `obj.__class__` can be monkey-patched at runtime.
    """
    def wrap(c):
        ann = dict(getattr(c, "__annotations__", {}))
        slots = tuple(ann.keys())
        ns = {}
        for k, v in c.__dict__.items():
            if k in ("__dict__", "__weakref__", "__slots__"):
                continue
            ns[k] = v
        ns["__slots__"] = slots
        ns["__strict_slot_types__"] = ann
        ns["__strict_freeze__"] = bool(freeze)

        if auto_init and "__init__" not in c.__dict__:
            params = ", ".join(slots)
            body_lines = [f"    self.{s} = {s}" for s in slots] or ["    pass"]
            init_src = f"def __init__(self, {params}):\n" + "\n".join(body_lines)
            init_ns = {}
            exec(init_src, {}, init_ns)
            init_fn = init_ns["__init__"]
            init_fn.__annotations__ = {**ann, "return": None}
            ns["__init__"] = init_fn

        ns["__qualname__"] = c.__qualname__
        ns["__module__"] = c.__module__
        new_cls = type(c.__name__, c.__bases__, ns)
        return new_cls

    return wrap if cls is None else wrap(cls)


_PRIMITIVE_STRICT_TAGS = {
    int: 1,
    float: 2,
    str: 3,
    bool: 4,
    None: 5,
    type(None): 5,
}


def _try_enable_c_strict_sig(fn, sig, param_checks, has_ret_check, ret_expected):
    if not isinstance(fn, types.FunctionType):
        return False
    co = fn.__code__
    # Do not use C-level _co_strict_sig on generators/coroutines or functions with cellvars
    if co.co_flags & (0x0020 | 0x0080 | 0x0100 | 0x0200):
        return False
    if co.co_cellvars or fn.__closure__:
        return False
    if fn.__name__ in co.co_names:
        return False
    params = list(sig.parameters.values())
    if len(params) > 9:
        return False
    for p in params:
        if p.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            return False

    ret_tag = 0
    if has_ret_check:
        if ret_expected is inspect.Parameter.empty or ret_expected is object or ret_expected is typing_any():
            ret_tag = 0
        elif ret_expected in _PRIMITIVE_STRICT_TAGS:
            ret_tag = _PRIMITIVE_STRICT_TAGS[ret_expected]
        else:
            return False

    param_map = dict(param_checks)
    strict_sig = 0x1 | (ret_tag << 2)
    checked_indices = set()
    for idx, p in enumerate(params):
        if p.name not in param_map:
            continue
        expected = param_map[p.name]
        if expected is inspect.Parameter.empty or expected is object or expected is typing_any():
            continue
        if expected not in _PRIMITIVE_STRICT_TAGS:
            return False
        tag = _PRIMITIVE_STRICT_TAGS[expected]
        strict_sig |= (tag << (5 + idx * 3))
        checked_indices.add(idx)

    # Check if any strict-checked parameter slot is overwritten in bytecode
    # Opcode numbers in co_code (wordcode: even bytes are opcodes, odd bytes are opargs)
    import opcode as _opcode
    store_ops = {
        _opcode.opmap.get(name)
        for name in ("STORE_FAST", "DELETE_FAST", "STORE_FAST_LOAD_FAST", "STORE_FAST_STORE_FAST", "SWAP_FAST")
    } - {None}
    store_fast_load_fast = _opcode.opmap.get("STORE_FAST_LOAD_FAST", -1)
    store_fast_store_fast = _opcode.opmap.get("STORE_FAST_STORE_FAST", -1)
    raw = co.co_code
    mutated = False
    for i in range(0, len(raw), 2):
        op = raw[i]
        if op in store_ops:
            arg = raw[i + 1]
            if op == store_fast_load_fast:
                if (arg >> 4) in checked_indices:
                    mutated = True
                    break
            elif op == store_fast_store_fast:
                if (arg >> 4) in checked_indices or (arg & 0xF) in checked_indices:
                    mutated = True
                    break
            else:
                if arg in checked_indices:
                    mutated = True
                    break
    if not mutated:
        strict_sig |= 0x2

    fn.__strict_sig__ = strict_sig
    fn.__is_strict_wrapped__ = True
    fn.__strict_unwrapped__ = fn
    return True


def _compile_fast_strict_wrapper(fn, sig, param_checks, has_ret_check, ret_expected):
    if _try_enable_c_strict_sig(fn, sig, param_checks, has_ret_check, ret_expected):
        return fn
    # Only synthesize a direct fast wrapper for functions with simple positional/keyword parameters
    params = list(sig.parameters.values())
    for p in params:
        if p.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            return None

    is_self_recursive = (
        isinstance(fn, types.FunctionType)
        and fn.__name__ in fn.__code__.co_names
        and not fn.__closure__
    )
    target_fn = fn
    if is_self_recursive:
        inner_globals = dict(fn.__globals__)
        target_fn = types.FunctionType(
            fn.__code__,
            inner_globals,
            name=fn.__name__,
            argdefs=fn.__defaults__,
            kwdefaults=fn.__kwdefaults__,
        )
        inner_globals[fn.__name__] = target_fn

    ns = {
        "_target_fn": target_fn,
        "_check_value": _check_value,
        "_int": int,
        "_float": float,
        "_str": str,
        "_bool": bool,
        "_type": type,
        "_isinstance": isinstance,
    }
    param_strs = []
    for idx, p in enumerate(params):
        if p.default is not inspect.Parameter.empty:
            def_name = f"_def_{idx}"
            ns[def_name] = p.default
            param_strs.append(f"{p.name}={def_name}")
        else:
            param_strs.append(p.name)

    lines = [f"def _fast_wrapper({', '.join(param_strs)}):"]
    for idx, (name, expected) in enumerate(param_checks):
        exp_sym = f"_exp_{idx}"
        ns[exp_sym] = expected
        if expected is int:
            lines.append(f"    if _type({name}) is not _int: _check_value({name}, {exp_sym}, {fn.__qualname__!r}, \"argument '{name}'\")")
        elif expected is float:
            lines.append(f"    if _type({name}) is not _float: _check_value({name}, {exp_sym}, {fn.__qualname__!r}, \"argument '{name}'\")")
        elif expected is str:
            lines.append(f"    if _type({name}) is not _str: _check_value({name}, {exp_sym}, {fn.__qualname__!r}, \"argument '{name}'\")")
        elif expected is bool:
            lines.append(f"    if _type({name}) is not _bool: _check_value({name}, {exp_sym}, {fn.__qualname__!r}, \"argument '{name}'\")")
        elif isinstance(expected, type):
            lines.append(f"    if not _isinstance({name}, {exp_sym}): _check_value({name}, {exp_sym}, {fn.__qualname__!r}, \"argument '{name}'\")")
        else:
            lines.append(f"    _check_value({name}, {exp_sym}, {fn.__qualname__!r}, \"argument '{name}'\")")

    call_args = ", ".join(p.name for p in params)
    lines.append(f"    _res = _target_fn({call_args})")
    if has_ret_check:
        ns["_ret_exp"] = ret_expected
        if ret_expected is int:
            lines.append(f"    if _type(_res) is not _int: _check_value(_res, _ret_exp, {fn.__qualname__!r}, 'return value')")
        elif ret_expected is float:
            lines.append(f"    if _type(_res) is not _float: _check_value(_res, _ret_exp, {fn.__qualname__!r}, 'return value')")
        elif ret_expected is str:
            lines.append(f"    if _type(_res) is not _str: _check_value(_res, _ret_exp, {fn.__qualname__!r}, 'return value')")
        elif ret_expected is bool:
            lines.append(f"    if _type(_res) is not _bool: _check_value(_res, _ret_exp, {fn.__qualname__!r}, 'return value')")
        elif ret_expected is None or ret_expected is type(None):
            lines.append(f"    if _res is not None: _check_value(_res, _ret_exp, {fn.__qualname__!r}, 'return value')")
        elif isinstance(ret_expected, type):
            lines.append(f"    if not _isinstance(_res, _ret_exp): _check_value(_res, _ret_exp, {fn.__qualname__!r}, 'return value')")
        else:
            lines.append(f"    _check_value(_res, _ret_exp, {fn.__qualname__!r}, 'return value')")
    lines.append("    return _res")
    exec("\n".join(lines), ns)
    w = functools.wraps(fn)(ns["_fast_wrapper"])
    w.__is_strict_wrapped__ = True
    w.__strict_unwrapped__ = fn
    return w


def _wrap_callable_strict(fn, *, propagate=False, require_annotations=False):
    if getattr(fn, "__is_strict_wrapped__", False):
        return fn
    sig = inspect.signature(fn)
    ann = getattr(fn, "__annotations__", {}) or {}
    if require_annotations:
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            if name not in ann:
                raise TypeError(
                    f"@strict(require_annotations=True): missing type annotation for "
                    f"parameter '{name}' in {fn.__qualname__}()"
                )
        if "return" not in ann:
            raise TypeError(
                f"@strict(require_annotations=True): missing return type annotation "
                f"in {fn.__qualname__}()"
            )

    param_checks = [
        (name, ann[name])
        for name in sig.parameters
        if name in ann and name not in ("self", "cls")
    ]
    has_ret_check = "return" in ann
    ret_expected = ann.get("return")

    if not propagate:
        fast_w = _compile_fast_strict_wrapper(fn, sig, param_checks, has_ret_check, ret_expected)
        if fast_w is not None:
            return fast_w

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if param_checks:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for name, expected in param_checks:
                _check_value(bound.arguments[name], expected, fn.__qualname__, f"argument '{name}'")
        if propagate:
            with strict_mode(require_annotations=require_annotations):
                res = fn(*args, **kwargs)
        else:
            res = fn(*args, **kwargs)
        if has_ret_check:
            _check_value(res, ret_expected, fn.__qualname__, "return value")
        return res

    wrapper.__is_strict_wrapped__ = True
    wrapper.__strict_unwrapped__ = fn
    return wrapper


def strict(obj=None, /, *, propagate=False, require_annotations=False):
    """Enforce type annotations on a function or class, optionally propagating strictness
    into all downstream third-party/legacy calls (`propagate=True`).
    """
    def decorate(target):
        if isinstance(target, type):
            # If the class has field annotations, also compile it as @struct
            ann = getattr(target, "__annotations__", None)
            for attr_name, attr_val in list(target.__dict__.items()):
                if isinstance(attr_val, types.FunctionType):
                     wrapped = _wrap_callable_strict(
                         attr_val,
                         propagate=propagate,
                         require_annotations=require_annotations,
                     )
                     setattr(target, attr_name, wrapped)
            if ann:
                return struct(target, freeze=True, auto_init=True)
            return target
        if callable(target):
            return _wrap_callable_strict(
                target,
                propagate=propagate,
                require_annotations=require_annotations,
            )
        raise TypeError(f"@strict can only be applied to functions or classes, got {type(target).__name__}")

    return decorate if obj is None else decorate(obj)


@contextlib.contextmanager
def strict_mode(*, require_annotations=False, check_modules=None):
    """Explicitly force all downstream calls (including third-party and legacy libraries)
    within this `with` block to inherit strict runtime type checking via `sys.monitoring`.

    - Any function with type annotations called inside `with strict_mode():` has its
      arguments and return value checked at runtime and raises `TypeError` on violation.
    - If `require_annotations=True`, calling an unannotated function in a target module
      immediately raises `TypeError`.
    - `check_modules` can be a set/tuple of module name prefixes to restrict enforcement.
    """
    tool_id = sys.monitoring.PROFILER_ID
    already_active = False
    try:
        if sys.monitoring.get_tool(tool_id) == "strict_mode":
            already_active = True
    except ValueError:
        pass
    if already_active:
        yield
        return

    sig_cache = {}

    def _should_check(code):
        if code.co_name.startswith("<"):
            return False
        fn_info = sig_cache.get(code)
        if fn_info is not None:
            return fn_info is not False
        frame = sys._getframe(2)
        mod_name = frame.f_globals.get("__name__", "")
        if mod_name == __name__ or mod_name.startswith(("contextlib", "inspect", "functools", "typing")):
            sig_cache[code] = False
            return False
        if check_modules is not None:
            if not any(mod_name == m or mod_name.startswith(m + ".") for m in check_modules):
                sig_cache[code] = False
                return False
        fn = frame.f_globals.get(code.co_name)
        if not isinstance(fn, types.FunctionType) or fn.__code__ is not code:
            # Check unwrap if decorated
            raw_fn = getattr(fn, "__strict_unwrapped__", None)
            if isinstance(raw_fn, types.FunctionType) and raw_fn.__code__ is code:
                fn = raw_fn
            else:
                sig_cache[code] = False
                return False
        ann = getattr(fn, "__annotations__", {}) or {}
        sig = inspect.signature(fn)
        sig_cache[code] = (fn, sig, ann)
        return True

    def on_py_start(code, instruction_offset):
        if not _should_check(code):
            return
        fn, sig, ann = sig_cache[code]
        if require_annotations:
            for pname in sig.parameters:
                if pname not in ("self", "cls") and pname not in ann:
                    raise TypeError(
                        f"strict_mode(require_annotations=True): third-party/legacy function "
                        f"'{fn.__module__}.{fn.__qualname__}()' is missing type annotation for '{pname}'"
                    )
            if "return" not in ann:
                raise TypeError(
                    f"strict_mode(require_annotations=True): third-party/legacy function "
                    f"'{fn.__module__}.{fn.__qualname__}()' is missing return type annotation"
                )
        if not ann:
            return
        frame = sys._getframe(1)
        f_locals = frame.f_locals
        for pname in sig.parameters:
            if pname in ann and pname not in ("self", "cls") and pname in f_locals:
                _check_value(f_locals[pname], ann[pname], f"{fn.__module__}.{fn.__qualname__}", f"argument '{pname}'")

    def on_py_return(code, instruction_offset, retval):
        info = sig_cache.get(code)
        if not info:
            return
        fn, sig, ann = info
        if "return" in ann:
            _check_value(retval, ann["return"], f"{fn.__module__}.{fn.__qualname__}", "return value")

    sys.monitoring.use_tool_id(tool_id, "strict_mode")
    try:
        sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_START, on_py_start)
        sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_RETURN, on_py_return)
        sys.monitoring.set_events(
            tool_id,
            sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN,
        )
        yield
    finally:
        sys.monitoring.set_events(tool_id, 0)
        sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_START, None)
        sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_RETURN, None)
        sys.monitoring.free_tool_id(tool_id)


def strict_module(mod, *, recursive=False, propagate=False, require_annotations=False, visited=None):
    """Explicitly instrument an imported third-party or legacy module so all its
    functions and classes inherit `@strict` enforcement.
    """
    if visited is None:
        visited = set()
    if id(mod) in visited:
        return mod
    visited.add(id(mod))
    mod_name = getattr(mod, "__name__", "")

    for name, obj in list(vars(mod).items()):
        if name.startswith("__"):
            continue
        if isinstance(obj, types.FunctionType) and getattr(obj, "__module__", None) == mod_name:
            setattr(
                mod,
                name,
                _wrap_callable_strict(
                    obj,
                    propagate=propagate,
                    require_annotations=require_annotations,
                ),
            )
        elif isinstance(obj, type) and getattr(obj, "__module__", None) == mod_name:
            for attr_name, attr_val in list(obj.__dict__.items()):
                if isinstance(attr_val, types.FunctionType):
                    try:
                        setattr(
                            obj,
                            attr_name,
                            _wrap_callable_strict(
                                attr_val,
                                propagate=propagate,
                                require_annotations=require_annotations,
                            ),
                        )
                    except TypeError:
                        # Immutable type (@struct) already sealed
                        pass
        elif recursive and isinstance(obj, types.ModuleType):
            child_name = getattr(obj, "__name__", "")
            if child_name.startswith(mod_name + "."):
                strict_module(
                    obj,
                    recursive=True,
                    propagate=propagate,
                    require_annotations=require_annotations,
                    visited=visited,
                )
    return mod
