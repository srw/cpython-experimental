"""
Duplicated Benchmark & Enforcement Suite:
Standard (Undecorated) Python vs. Decorated (@strict / @struct) Python
======================================================================
Runs every core benchmark in two forms:
1. Original Undecorated Python (baseline)
2. Duplicated @strict / @struct Python (accelerated)
3. Duplicated Negative Tests (intentionally violating @strict / @struct
   contracts and verifying that TypeError / AttributeError is raised and caught)
"""

import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from strict import strict, struct, strict_module, strict_mode


# ============================================================================
# 1. ARITHMETIC INTEGER LOOP (Undecorated vs @strict)
# ============================================================================
def arith_loop_undecorated(n: int = 2_000_000) -> int:
    acc = 0
    for i in range(1, n):
        acc = (acc + i * 3 - (i >> 1)) ^ (i & 0xFF)
    return acc


@strict
def arith_loop_strict(n: int = 2_000_000) -> int:
    acc = 0
    for i in range(1, n):
        acc = (acc + i * 3 - (i >> 1)) ^ (i & 0xFF)
    return acc


# ============================================================================
# 2. TIGHT INTEGER ACCUMULATOR LOOP (Undecorated vs @strict)
# ============================================================================
def int_loop_undecorated(n: int = 2_000_000) -> int:
    a = 1
    b = 2
    c = 3
    for _ in range(n):
        a = a + 1
        b = b + a
        c = c + b - a
    return c


@strict
def int_loop_strict(n: int = 2_000_000) -> int:
    a = 1
    b = 2
    c = 3
    for _ in range(n):
        a = a + 1
        b = b + a
        c = c + b - a
    return c


# ============================================================================
# 3. FLOATING-POINT ACCUMULATOR LOOP (Undecorated vs @strict)
# ============================================================================
def float_loop_undecorated(n: int = 2_000_000) -> float:
    x = 1.0
    y = 2.0
    z = 3.0
    for _ in range(n):
        x = x + 0.5
        y = y * 1.000001
        z = z + x - y
    return z


@strict
def float_loop_strict(n: int = 2_000_000) -> float:
    x = 1.0
    y = 2.0
    z = 3.0
    for _ in range(n):
        x = x + 0.5
        y = y * 1.000001
        z = z + x - y
    return z


# ============================================================================
# 4. RECURSIVE CALL DISPATCH (Undecorated vs @strict)
# ============================================================================
def _fib_undecorated(n: int) -> int:
    if n < 2:
        return n
    return _fib_undecorated(n - 1) + _fib_undecorated(n - 2)


def fib_recursive_undecorated() -> int:
    return _fib_undecorated(28)


@strict
def _fib_strict(n: int) -> int:
    if n < 2:
        return n
    return _fib_strict(n - 1) + _fib_strict(n - 2)


@strict
def fib_recursive_strict() -> int:
    return _fib_strict(28)


# ============================================================================
# 5. OBJECT ATTRIBUTE CHURN (Undecorated Class vs @struct + @strict)
# ============================================================================
class Point3DUndecorated:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def translate(self, dx: float, dy: float, dz: float) -> float:
        self.x = self.x + dx
        self.y = self.y + dy
        self.z = self.z + dz
        return self.x + self.y + self.z


def object_attr_churn_undecorated(n: int = 1_000_000) -> float:
    p = Point3DUndecorated(1.0, 2.0, 3.0)
    total = 0.0
    for _ in range(n):
        total = total + p.translate(0.1, 0.2, 0.3)
    return total


@struct
class Point3DStruct:
    x: float
    y: float
    z: float

    @strict
    def translate(self, dx: float, dy: float, dz: float) -> float:
        self.x = self.x + dx
        self.y = self.y + dy
        self.z = self.z + dz
        return self.x + self.y + self.z


@strict
def object_attr_churn_strict(n: int = 1_000_000) -> float:
    p = Point3DStruct(1.0, 2.0, 3.0)
    total = 0.0
    for _ in range(n):
        total = total + p.translate(0.1, 0.2, 0.3)
    return total


# ============================================================================
# 6. ATTRIBUTE ARITHMETIC LOOP (Undecorated Class vs @struct + @strict)
# ============================================================================
class PointUndecorated:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


def attr_loop_undecorated(n: int = 2_000_000) -> float:
    p = PointUndecorated(1.0, 2.0, 3.0)
    for _ in range(n):
        p.x = p.x + 0.5
        p.y = p.y * 1.000001
        p.z = p.z + p.x - p.y
    return p.z


@struct
class PointStruct:
    x: float
    y: float
    z: float


@strict
def attr_loop_strict(n: int = 2_000_000) -> float:
    p = PointStruct(1.0, 2.0, 3.0)
    for _ in range(n):
        p.x = p.x + 0.5
        p.y = p.y * 1.000001
        p.z = p.z + p.x - p.y
    return p.z


# ============================================================================
# 7. N-BODY FLOAT PHYSICS SIMULATION (Undecorated Class vs @struct + @strict)
# ============================================================================
class ParticleUndecorated:
    def __init__(self, x: float, y: float, z: float, vx: float, vy: float, vz: float, mass: float):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass

    def step(self, dt: float) -> float:
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
        self.z = self.z + self.vz * dt
        self.vx = self.vx * 0.999999
        self.vy = self.vy * 0.999999
        self.vz = self.vz * 0.999999
        return (self.x * self.x + self.y * self.y + self.z * self.z) * self.mass


def nbody_float_undecorated(n: int = 1_000_000) -> float:
    p1 = ParticleUndecorated(1.0, 2.0, 3.0, 0.5, -0.25, 0.125, 2.5)
    p2 = ParticleUndecorated(-1.5, 0.5, 2.0, -0.3, 0.4, -0.2, 1.8)
    energy = 0.0
    dt = 0.001
    for _ in range(n):
        energy = energy + p1.step(dt) + p2.step(dt)
    return energy


@struct
class ParticleStruct:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass: float

    @strict
    def step(self, dt: float) -> float:
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
        self.z = self.z + self.vz * dt
        self.vx = self.vx * 0.999999
        self.vy = self.vy * 0.999999
        self.vz = self.vz * 0.999999
        return (self.x * self.x + self.y * self.y + self.z * self.z) * self.mass


@strict
def nbody_float_strict(n: int = 1_000_000) -> float:
    p1 = ParticleStruct(1.0, 2.0, 3.0, 0.5, -0.25, 0.125, 2.5)
    p2 = ParticleStruct(-1.5, 0.5, 2.0, -0.3, 0.4, -0.2, 1.8)
    energy = 0.0
    dt = 0.001
    for _ in range(n):
        energy = energy + p1.step(dt) + p2.step(dt)
    return energy


# ============================================================================
# 8. DATACLASS vs @struct INSTANTIATION & FIELD ACCESS
# ============================================================================
@dataclass(slots=True)
class OrderItemDataclass:
    order_id: int
    price: float
    quantity: int


def dataclass_inst_undecorated(n: int = 400_000) -> float:
    total = 0.0
    for i in range(n):
        item = OrderItemDataclass(i, 19.99, (i & 7) + 1)
        total = total + item.price * item.quantity
    return total


@struct
class OrderItemStruct:
    order_id: int
    price: float
    quantity: int


@strict
def dataclass_inst_strict(n: int = 400_000) -> float:
    total = 0.0
    for i in range(n):
        item = OrderItemStruct(i, 19.99, (i & 7) + 1)
        total = total + item.price * item.quantity
    return total


# ============================================================================
# 9. LIST OPERATIONS (Undecorated vs @strict)
# ============================================================================
def list_ops_undecorated(n: int = 200_000) -> int:
    lst = [i * 2 for i in range(n)]
    for i in range(n // 2):
        lst[i] = lst[i] + i
    return sum(lst)


@strict
def list_ops_strict(n: int = 200_000) -> int:
    lst = [i * 2 for i in range(n)]
    for i in range(n // 2):
        lst[i] = lst[i] + i
    return sum(lst)


# ============================================================================
# 10. DICT OPERATIONS (Undecorated vs @strict)
# ============================================================================
def dict_ops_undecorated(n: int = 150_000) -> int:
    d = {}
    for i in range(n):
        d[i] = i * 3
    total = 0
    for i in range(n):
        total = total + d[i]
    return total


@strict
def dict_ops_strict(n: int = 150_000) -> int:
    d = {}
    for i in range(n):
        d[i] = i * 3
    total = 0
    for i in range(n):
        total = total + d[i]
    return total


# ============================================================================
# 11. 4-THREAD PARALLEL EXECUTION (Undecorated vs @strict/@struct)
# ============================================================================
def parallel_4threads_undecorated() -> float:
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(nbody_float_undecorated, 400_000) for _ in range(4)]
        return sum(f.result() for f in futs)


@strict
def parallel_4threads_strict() -> float:
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(nbody_float_strict, 400_000) for _ in range(4)]
        return sum(f.result() for f in futs)


# ============================================================================
# NEGATIVE ENFORCEMENT TESTS (Must Fail and Be Caught!)
# ============================================================================
def run_negative_enforcement_tests() -> None:
    print("\n" + "=" * 80)
    print("NEGATIVE ENFORCEMENT TESTS (Verifying violations fail & are caught)")
    print("=" * 80)

    cases = []

    # 1. @strict argument type violation on arith_loop_strict
    try:
        arith_loop_strict("not_an_int")
        raise AssertionError("arith_loop_strict('not_an_int') did not raise TypeError!")
    except TypeError as exc:
        cases.append(("@strict arg type mismatch (arith_loop_strict)", type(exc).__name__, str(exc)))

    # 2. @strict argument type violation on recursive _fib_strict
    try:
        _fib_strict(10.5)
        raise AssertionError("_fib_strict(10.5) did not raise TypeError!")
    except TypeError as exc:
        cases.append(("@strict arg type mismatch (_fib_strict)", type(exc).__name__, str(exc)))

    # 3. @strict return type violation
    @strict
    def bad_return(x: int) -> int:
        return "wrong_return_type"

    try:
        bad_return(42)
        raise AssertionError("bad_return(42) did not raise TypeError!")
    except TypeError as exc:
        cases.append(("@strict return type mismatch", type(exc).__name__, str(exc)))

    # 4. @struct field type violation during __init__
    try:
        Point3DStruct("bad_x", 2.0, 3.0)
        raise AssertionError("Point3DStruct('bad_x', 2.0, 3.0) did not raise TypeError!")
    except TypeError as exc:
        cases.append(("@struct constructor field type mismatch", type(exc).__name__, str(exc)))

    # 5. @struct C-slot write type violation inside loop
    try:
        p = Point3DStruct(1.0, 2.0, 3.0)
        p.x = "illegal_string"
        raise AssertionError("p.x = 'illegal_string' did not raise TypeError!")
    except TypeError as exc:
        cases.append(("@struct C-slot write type mismatch (p.x = str)", type(exc).__name__, str(exc)))

    # 6. @struct bool rejected for int slot (C-level exact type guard)
    try:
        item = OrderItemStruct(1, 9.99, 2)
        item.quantity = True
        raise AssertionError("item.quantity = True did not raise TypeError!")
    except TypeError as exc:
        cases.append(("@struct C-slot bool-for-int rejection", type(exc).__name__, str(exc)))

    # 7. @struct dynamic attribute creation forbidden (sealed __slots__)
    try:
        p = Point3DStruct(1.0, 2.0, 3.0)
        p.new_dynamic_attr = 99.0
        raise AssertionError("p.new_dynamic_attr = 99.0 did not raise AttributeError!")
    except AttributeError as exc:
        cases.append(("@struct dynamic attribute creation rejected", type(exc).__name__, str(exc)))

    # 8. @strict method argument violation on @struct
    try:
        p = Point3DStruct(1.0, 2.0, 3.0)
        p.translate(1.0, "bad_dy", 3.0)
        raise AssertionError("p.translate(1.0, 'bad_dy', 3.0) did not raise TypeError!")
    except TypeError as exc:
        cases.append(("@struct @strict method arg mismatch", type(exc).__name__, str(exc)))

    for label, err_type, msg in cases:
        print(f"  [CAUGHT {err_type:<14}] {label:<46} -> {msg}")
    print(f"\nAll {len(cases)}/{len(cases)} negative enforcement tests failed as expected and were caught!")


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================
BENCHMARK_PAIRS = [
    ("arith_loop", arith_loop_undecorated, arith_loop_strict),
    ("int_loop", int_loop_undecorated, int_loop_strict),
    ("float_loop", float_loop_undecorated, float_loop_strict),
    ("fib_recursive", fib_recursive_undecorated, fib_recursive_strict),
    ("object_attr_churn", object_attr_churn_undecorated, object_attr_churn_strict),
    ("attr_loop", attr_loop_undecorated, attr_loop_strict),
    ("nbody_float", nbody_float_undecorated, nbody_float_strict),
    ("dataclass_inst", dataclass_inst_undecorated, dataclass_inst_strict),
    ("list_ops", list_ops_undecorated, list_ops_strict),
    ("dict_ops", dict_ops_undecorated, dict_ops_strict),
    ("parallel_4threads", parallel_4threads_undecorated, parallel_4threads_strict),
]


def bench_one(fn, warmup: int = 2, repeats: int = 5):
    res = None
    for _ in range(warmup):
        res = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return statistics.median(times), res


def main():
    run_negative_enforcement_tests()

    print("\n" + "=" * 86)
    print(f"{'Benchmark':<20} | {'Undecorated (ms)':>16} | {'@strict/@struct (ms)':>20} | {'Speedup':>10} | {'Status':>10}")
    print("-" * 86)

    ratios = []
    for name, fn_std, fn_strict in BENCHMARK_PAIRS:
        t_std, res_std = bench_one(fn_std)
        t_strict, res_strict = bench_one(fn_strict)
        # Verify result parity
        if isinstance(res_std, float):
            assert abs(res_std - res_strict) / max(1.0, abs(res_std)) < 1e-9, (
                f"Mismatch in {name}: {res_std} vs {res_strict}"
            )
        else:
            assert res_std == res_strict, f"Mismatch in {name}: {res_std} vs {res_strict}"

        speedup = t_std / t_strict
        ratios.append(speedup)
        print(f"{name:<20} | {t_std:>13.2f} ms | {t_strict:>17.2f} ms | {speedup:>9.2f}x | {'PASS':>10}")

    geomean = statistics.geometric_mean(ratios)
    print("-" * 86)
    print(f"{'Geometric Mean':<20} | {'':>16} | {'':>20} | {geomean:>9.2f}x | {'ALL PASS':>10}")
    print("=" * 86)


if __name__ == "__main__":
    main()
