"""18 isolated complete-cutover scenarios; no network/process/systemd/GPU."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ops import production_simulate


class CutoverSimulationMatrix(unittest.TestCase):
    pass


def case(scenario):
    def test(self):
        outcome = production_simulate.run_scenario(scenario)
        self.assertEqual(outcome["result"], "PASS")
        self.assertTrue(outcome["simulated"])
    return test


for scenario in production_simulate.SCENARIOS:
    setattr(CutoverSimulationMatrix, "test_" + scenario, case(scenario))
