from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


WATCHDOG_PATH = Path(__file__).resolve().parents[2] / "w2_restage_watchdog.py"
SPEC = importlib.util.spec_from_file_location("w2_restage_watchdog", WATCHDOG_PATH)
assert SPEC is not None and SPEC.loader is not None
WATCHDOG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WATCHDOG)


class CgroupHeadroomTests(unittest.TestCase):
    def test_soft_high_does_not_reduce_hard_limit_headroom(self):
        gib = 1 << 30
        maximum, high, current = 100 * gib, 96 * gib, 92 * gib
        max_headroom, high_headroom = WATCHDOG.cgroup_headrooms(current, maximum, high)
        self.assertEqual(max_headroom, 8 * gib)
        self.assertEqual(high_headroom, 4 * gib)

    def test_crossing_soft_high_keeps_remaining_hard_headroom_visible(self):
        gib = 1 << 30
        max_headroom, high_headroom = WATCHDOG.cgroup_headrooms(
            97 * gib, 100 * gib, 96 * gib
        )
        self.assertEqual(max_headroom, 3 * gib)
        self.assertEqual(high_headroom, -1 * gib)

    def test_unlimited_soft_high_tracks_hard_limit(self):
        gib = 1 << 30
        self.assertEqual(
            WATCHDOG.cgroup_headrooms(20 * gib, 100 * gib, None),
            (80 * gib, 80 * gib),
        )


if __name__ == "__main__":
    unittest.main()
