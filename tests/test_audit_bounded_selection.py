"""Training-side selector checks for the physical-bound sensitivity."""

import unittest

from audit_bounded_selection import choose


class BoundedSelectionTests(unittest.TestCase):
    def test_model_only_selector_ignores_rules_and_uses_declared_ties(self):
        order = ["surface_area", "random", "ridge", "svr"]
        scores = {
            "surface_area": {"mae": float("nan"), "loss": 0.0},
            "random": {"mae": float("nan"), "loss": 0.0},
            "ridge": {"mae": 1.0, "loss": 2.0},
            "svr": {"mae": 1.0, "loss": 2.0},
        }
        self.assertEqual(choose(scores, order, "fixed_models_mae"), "ridge")
        self.assertEqual(choose(scores, order, "fixed_models_selection_loss"), "ridge")
        self.assertEqual(choose(scores, order, "models_and_rules_selection_loss"),
                         "surface_area")


if __name__ == "__main__":
    unittest.main()
