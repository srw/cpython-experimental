import types
import unittest
from strict import strict, struct, strict_mode, strict_module


@struct
class Point3D:
    x: float
    y: float
    z: float

    @strict
    def norm_sq(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    @strict
    def translate(self, dx: float, dy: float, dz: float) -> None:
        self.x += dx
        self.y += dy
        self.z += dz


@struct
class CounterItem:
    label: str
    count: int
    active: bool


@strict
def add_ints(a: int, b: int) -> int:
    return a + b


@strict
def scale_float(x: float, factor: float) -> float:
    return x * factor


@strict
def range_unused_var(n: int) -> int:
    total = 0
    for _ in range(n):
        total += 3
    return total


@strict
def range_used_var(n: int) -> int:
    acc = 0
    for i in range(n):
        acc += (i * 3 + 7) // 2 - (i % 5)
    return acc


@strict
def range_list_comp(n: int) -> int:
    lst = [i * 2 for i in range(n)]
    total = 0
    for x in lst:
        total += x
    return total


@strict
def range_dict_store(n: int) -> int:
    d = {}
    for i in range(n):
        d[i] = i * 3
    total = 0
    for k in range(n):
        total += d[k]
    return total


@strict
def sum_list_strict(items: list) -> int:
    total = 0
    for x in items:
        total += x
    return total


@strict
def lookup_dict_strict(d: dict, key: str) -> int:
    return d[key]


@strict
def make_list_strict(x: int, y: int) -> list:
    return [x, y]


@strict
def make_dict_strict(k: str, v: int) -> dict:
    return {k: v}


class StrictTests(unittest.TestCase):

    def test_strict_function_valid_and_invalid_args(self):
        self.assertEqual(add_ints(10, 20), 30)
        self.assertAlmostEqual(scale_float(2.5, 4.0), 10.0)

        with self.assertRaises(TypeError):
            add_ints("10", 20)
        with self.assertRaises(TypeError):
            add_ints(10, 20.0)
        # bool is a subclass of int in Python, but @strict enforces exact int
        with self.assertRaises(TypeError):
            add_ints(True, 1)
        # int is rejected when exact float is annotated
        with self.assertRaises(TypeError):
            scale_float(2, 4.0)

    def test_strict_return_type_enforcement(self):
        @strict
        def bad_return(x: int) -> int:
            return "not_an_int"

        with self.assertRaises(TypeError):
            bad_return(5)

    def test_strict_container_tags_zero_wrapper(self):
        # Verify list (tag 6) and dict (tag 7) use C-level __strict_sig__ without wrapper
        self.assertIsNotNone(getattr(sum_list_strict, "__strict_sig__", None))
        self.assertIs(sum_list_strict.__strict_unwrapped__, sum_list_strict)
        self.assertIsNotNone(getattr(lookup_dict_strict, "__strict_sig__", None))
        self.assertIsNotNone(getattr(make_list_strict, "__strict_sig__", None))
        self.assertIsNotNone(getattr(make_dict_strict, "__strict_sig__", None))

        self.assertEqual(sum_list_strict([1, 2, 3, 4]), 10)
        self.assertEqual(lookup_dict_strict({"a": 42}, "a"), 42)
        self.assertEqual(make_list_strict(7, 8), [7, 8])
        self.assertEqual(make_dict_strict("k", 99), {"k": 99})

        with self.assertRaises(TypeError):
            sum_list_strict((1, 2, 3))
        with self.assertRaises(TypeError):
            lookup_dict_strict([1, 2], "a")

        @strict
        def bad_list_return(x: int) -> list:
            return (x,)

        @strict
        def bad_dict_return(x: int) -> dict:
            return [x]

        with self.assertRaises(TypeError):
            bad_list_return(1)
        with self.assertRaises(TypeError):
            bad_dict_return(1)

    def test_struct_slots_and_c_descriptor_enforcement(self):
        p = Point3D(1.0, 2.0, 3.0)
        self.assertFalse(hasattr(p, "__dict__"))
        self.assertAlmostEqual(p.norm_sq(), 14.0)

        p.translate(0.5, -1.0, 2.0)
        self.assertEqual((p.x, p.y, p.z), (1.5, 1.0, 5.0))

        with self.assertRaises(TypeError):
            Point3D("bad", 2.0, 3.0)
        with self.assertRaises(TypeError):
            p.x = "bad"
        with self.assertRaises(TypeError):
            p.x = 1  # exact float required

        with self.assertRaises(AttributeError):
            p.extra_attr = 1.0

        # Immutable type check
        with self.assertRaises(TypeError):
            Point3D.new_method = lambda self: None

        item = CounterItem("bolts", 42, True)
        self.assertEqual(item.label, "bolts")
        self.assertEqual(item.count, 42)
        self.assertIs(item.active, True)
        with self.assertRaises(TypeError):
            item.count = False
        with self.assertRaises(TypeError):
            item.active = 1

    def test_tier2_range_loops_and_containers(self):
        # Run enough iterations to trigger Tier 2 / JIT trace compilation
        self.assertEqual(range_unused_var(5000), 15000)
        expected_used = sum((i * 3 + 7) // 2 - (i % 5) for i in range(5000))
        self.assertEqual(range_used_var(5000), expected_used)
        self.assertEqual(range_list_comp(2000), sum(i * 2 for i in range(2000)))
        self.assertEqual(range_dict_store(1500), sum(i * 3 for i in range(1500)))

    def test_strict_module_and_mode(self):
        mod = types.ModuleType("dummy_legacy_mod")
        exec(
            "def compute(x: int, y: int) -> int:\n"
            "    return x + y\n",
            mod.__dict__,
        )
        strict_module(mod)
        self.assertEqual(mod.compute(3, 4), 7)
        with self.assertRaises(TypeError):
            mod.compute("3", 4)

        with strict_mode(require_annotations=False):
            self.assertEqual(mod.compute(5, 6), 11)


if __name__ == "__main__":
    unittest.main()
