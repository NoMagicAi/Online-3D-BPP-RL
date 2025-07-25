import unittest
import numpy as np

# We assume the following classes are in a reachable path.
# Adjust the import path if your project structure is different.
from envs.bpp0.stability import Box, StackingTree

class TestRecursiveStabilityValidation(unittest.TestCase):
    """
    A focused test suite to validate the corrected recursive stability logic.
    This suite checks multi-level stacks to ensure the force propagation and
    cumulative Center of Mass (CoM) calculations are correct.
    """

    def setUp(self):
        """Set up a fresh StackingTree for each test."""
        self.stacking_tree = StackingTree()
        print("\n" + "="*70)

    def test_A_stable_pillar_stack(self):
        """🧪 VALIDATION 1: A perfectly centered pillar stack should be stable."""
        print("DIAGNOSING: Stable Pillar Stack (Validates previous failure)")

        # Step 1: Place a large base on the floor.
        base_box = Box(box_id=1, x=0, y=0, z=0, lx=10, ly=10, lz=2)
        self.stacking_tree.add_box_permanently(base_box)
        print("  - Step 1: Placed a 10x10 base.")

        # Step 2: Place a smaller pillar in the center of the base.
        pillar_box = Box(box_id=2, x=4, y=4, z=2, lx=2, ly=2, lz=5) # CoM at (5,5)
        self.stacking_tree.add_box_permanently(pillar_box)
        print("  - Step 2: Placed a 2x2 pillar at the center.")

        # Step 3: Check if a perfectly centered top box is stable.
        # This is the exact scenario that failed before.
        top_box_stable = Box(box_id=None, x=3, y=3, z=7, lx=4, ly=4, lz=2) # CoM at (5,5)
        print("  - Step 3: Checking stability of a centered box on the pillar...")
        
        is_stable = self.stacking_tree.is_placement_stable(top_box_stable, self.stacking_tree.boxes)
        
        self.assertTrue(is_stable, "FAIL: A centered box on a stable pillar should be STABLE.")
        print("    - Result: Centered top box is correctly identified as STABLE.")

    def test_B_recursively_unstable_stack(self):
        """🧪 VALIDATION 2: A stack that becomes unstable due to a new load."""
        print("DIAGNOSING: True Recursive Instability")

        # Step 1: Place a large base.
        base = Box(box_id=1, x=0, y=0, z=0, lx=10, ly=10, lz=2)
        self.stacking_tree.add_box_permanently(base)
        print("  - Step 1: Placed a 10x10 base.")

        # Step 2: Place Box B precariously on the edge of the base.
        # Its CoM is at x=9.9, which is just inside the support polygon from the base (ends at x=10).
        box_b = Box(box_id=2, x=7.9, y=4, z=2, lx=4, ly=4, lz=2) # CoM at x=9.9
        self.stacking_tree.add_box_permanently(box_b)
        print("  - Step 2: Placed a precarious box (B) with CoM at x=9.9.")

        # Step 3: Place Box C on Box B, shifted slightly further out.
        # Box C's CoM is at x=10.3.
        box_c = Box(box_id=None, x=8.3, y=4, z=4, lx=4, ly=4, lz=2) # CoM at x=10.3
        print("  - Step 3: Checking stability of a tipping box (C) on top of B...")

        # The new cumulative CoM of {B+C} will be pulled over the edge.
        # If masses are equal, new CoM x-coord = (9.9 + 10.3) / 2 = 10.1.
        # This new CoM is outside the support polygon for B (which ends at x=10).
        # The recursive check should detect this and return False.
        is_stable = self.stacking_tree.is_placement_stable(box_c, self.stacking_tree.boxes)

        self.assertFalse(is_stable, "FAIL: The stack should become recursively UNSTABLE.")
        print("    - Result: The stack is correctly identified as UNSTABLE.")

    def test_C_tall_stable_tower(self):
        """🧪 VALIDATION 3: A tall, stable tower should not fail due to accumulated error."""
        print("DIAGNOSING: Tall Stable Tower")

        # Create a stable base
        current_base = Box(box_id=1, x=0, y=0, z=0, lx=10, ly=10, lz=2)
        self.stacking_tree.add_box_permanently(current_base)
        print("  - Placed Level 0 (Base)")

        # Add 5 more stable levels
        for i in range(1, 6):
            level_box = Box(box_id=i+1, x=2, y=2, z=i*2, lx=6, ly=6, lz=2)
            is_stable = self.stacking_tree.is_placement_stable(level_box, self.stacking_tree.boxes)
            self.assertTrue(is_stable, f"FAIL: Level {i} of the tower should be STABLE.")
            self.stacking_tree.add_box_permanently(level_box)
            print(f"  - Placed Level {i}. Stability check PASSED.")
        
        print("    - Result: Tall stable tower was correctly built.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
