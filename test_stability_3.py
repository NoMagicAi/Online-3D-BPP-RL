import unittest
import numpy as np

# We assume the following classes are in a reachable path.
# Adjust the import path if your project structure is different.
from envs.bpp0.stability import Box, StackingTree

class TestComplexStabilityScenarios(unittest.TestCase):
    """
    A suite of more complex and challenging tests to validate the robustness
    of the recursive stability and force distribution logic.
    """

    def setUp(self):
        """Set up a fresh StackingTree for each test."""
        self.stacking_tree = StackingTree()
        print("\n" + "="*70)

    def test_A_chain_reaction_instability(self):
        """🧪 SCENARIO 1: A multi-level stack where the top box causes a failure two levels down."""
        print("DIAGNOSING: Chain-Reaction Instability")

        # Step 1: Create a base and a precarious second level.
        box_a = Box(box_id=1, x=0, y=0, z=0, lx=10, ly=10, lz=2) # Base
        self.stacking_tree.add_box_permanently(box_a)
        # Box B's CoM is at x=9.9, just inside the support from A (ends at x=10). Stable.
        box_b = Box(box_id=2, x=7.9, y=4, z=2, lx=4, ly=4, lz=2) 
        self.stacking_tree.add_box_permanently(box_b)
        print("  - Step 1: Placed Base (A) and a precarious Box (B).")

        # Step 2: Add a stable third level.
        # Box C is perfectly centered on B. The {B+C} stack is still stable on A.
        # Cumulative CoM of {B+C} is still at x=9.9.
        box_c = Box(box_id=3, x=7.9, y=4, z=4, lx=4, ly=4, lz=2)
        self.stacking_tree.add_box_permanently(box_c)
        print("  - Step 2: Placed a centered Box (C) on B. Stack is stable.")
        
        # Step 3: Add a final box (D) that tips the whole structure.
        # Box D is placed slightly offset on C.
        # This should shift the {C+D} CoM, which in turn shifts the {B+C+D} CoM
        # just over the edge of the support from A.
        box_d = Box(box_id=None, x=8.1, y=4, z=6, lx=4, ly=4, lz=2) # CoM at x=10.1
        print("  - Step 3: Checking placement of a final tipping box (D)...")

        is_stable = self.stacking_tree.is_placement_stable(box_d, self.stacking_tree.boxes)

        self.assertFalse(is_stable, "FAIL: The final box should cause a recursive chain-reaction instability.")
        print("    - Result: Chain reaction correctly identified as UNSTABLE.")

    def test_B_asymmetric_bridge_tipping(self):
        """🧪 SCENARIO 2: A stable bridge that becomes unstable when weighted on its weaker side."""
        print("DIAGNOSING: Asymmetric Bridge Tipping")

        # Step 1: An asymmetric base. A wide platform and a narrow pillar.
        platform = Box(box_id=1, x=0, y=0, z=0, lx=8, ly=2, lz=5)
        pillar = Box(box_id=2, x=9, y=0, z=0, lx=1, ly=2, lz=5)
        self.stacking_tree.add_box_permanently(platform)
        self.stacking_tree.add_box_permanently(pillar)
        print("  - Step 1: Placed a wide platform and a narrow pillar.")

        # Step 2: Place a long bridge across both. It is stable.
        bridge = Box(box_id=3, x=0, y=0, z=5, lx=10, ly=2, lz=2)
        self.stacking_tree.add_box_permanently(bridge)
        print("  - Step 2: Placed a stable bridge across them.")
        self.assertTrue(self.stacking_tree.is_placement_stable(bridge, [platform, pillar]))

        # Step 3: Place a heavy box on the bridge, directly over the narrow pillar.
        # This load should propagate down, and the recursive check should find
        # that the bridge's new cumulative CoM is no longer stable on its combined support.
        load = Box(box_id=None, x=9, y=0, z=7, lx=1, ly=2, lz=10, density=5.0) # Heavy
        print("  - Step 3: Checking placement of a heavy load over the narrow pillar...")

        is_stable = self.stacking_tree.is_placement_stable(load, self.stacking_tree.boxes)

        self.assertFalse(is_stable, "FAIL: Heavy load on the weak side of the bridge should be UNSTABLE.")
        print("    - Result: Asymmetric bridge tipping correctly identified as UNSTABLE.")

    def test_C_counterweight_stability(self):
        """🧪 SCENARIO 3: An unstable cantilever that is stabilized by a counterweight."""
        print("DIAGNOSING: Counterweight Stabilization")

        # Step 1: Place a base.
        base = Box(box_id=1, x=5, y=5, z=0, lx=2, ly=2, lz=5)
        self.stacking_tree.add_box_permanently(base)
        print("  - Step 1: Placed a central pillar base.")

        # Step 2: Place a long plank on the base.
        # The plank's CoM is at x=6, which is inside the support from the base (x=5 to x=7). Stable.
        plank = Box(box_id=2, x=1, y=5, z=5, lx=10, ly=2, lz=1)
        self.stacking_tree.add_box_permanently(plank)
        print("  - Step 2: Placed a stable plank on the pillar.")
        
        # Step 3: Place a weight on one end of the plank, making it unstable.
        # This weight's CoM is at x=10.5. The combined CoM of {plank+weight1} will shift far to the right.
        weight1 = Box(box_id=None, x=10, y=5, z=6, lx=1, ly=2, lz=2, density=5.0)
        print("  - Step 3: Checking that adding one weight makes the plank unstable...")
        is_stable_with_one_weight = self.stacking_tree.is_placement_stable(weight1, self.stacking_tree.boxes)
        self.assertFalse(is_stable_with_one_weight, "FAIL: The plank should be UNSTABLE with only one weight.")
        print("    - Correctly identified as UNSTABLE.")
        self.stacking_tree.add_box_permanently(weight1) # Add it anyway for the next step

        # Step 4: Place a counterweight on the other end. This should re-stabilize the structure.
        # The counterweight's CoM is at x=1.5. This should pull the cumulative CoM of
        # {plank+weight1+counterweight} back over the central pillar.
        counterweight = Box(box_id=None, x=1, y=5, z=6, lx=1, ly=2, lz=2, density=5.0)
        print("  - Step 4: Checking if a counterweight re-stabilizes the plank...")
        
        is_stable_with_counterweight = self.stacking_tree.is_placement_stable(counterweight, self.stacking_tree.boxes)

        self.assertTrue(is_stable_with_counterweight, "FAIL: The counterweight should make the plank STABLE.")
        print("    - Result: Counterweight effect correctly identified as STABLE.")

    def test_D_hollow_structure_instability(self):
        """🧪 SCENARIO 4: Placing a load in the center of a box bridging a hollow structure."""
        print("DIAGNOSING: Hollow Structure Instability")

        # Step 1: Build a 10x10 hollow square with four pillars.
        p1 = Box(box_id=1, x=0, y=0, z=0, lx=2, ly=10, lz=5) # Left
        p2 = Box(box_id=2, x=8, y=0, z=0, lx=2, ly=10, lz=5) # Right
        p3 = Box(box_id=3, x=2, y=0, z=0, lx=6, ly=2, lz=5) # Top
        p4 = Box(box_id=4, x=2, y=8, z=0, lx=6, ly=2, lz=5) # Bottom
        for p in [p1, p2, p3, p4]: self.stacking_tree.add_box_permanently(p)
        print("  - Step 1: Built a hollow square base.")

        # Step 2: Place a large, flat lid on top. This is stable.
        lid = Box(box_id=5, x=0, y=0, z=5, lx=10, ly=10, lz=1)
        self.stacking_tree.add_box_permanently(lid)
        print("  - Step 2: Placed a stable lid on top.")

        # Step 3: Place a heavy, concentrated load in the center of the lid.
        # The load's CoM is at (5,5).
        # The support polygon for the lid is the frame of the hollow square.
        # The point (5,5) is NOT inside this support polygon.
        # Therefore, this placement should be unstable.
        load = Box(box_id=None, x=4, y=4, z=6, lx=2, ly=2, lz=5, density=10.0) # Heavy
        print("  - Step 3: Checking placement of a heavy load in the center of the lid...")

        is_stable = self.stacking_tree.is_placement_stable(load, self.stacking_tree.boxes)

        self.assertFalse(is_stable, "FAIL: A load placed over the hole in the support should be UNSTABLE.")
        print("    - Result: Load on hollow structure correctly identified as UNSTABLE.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
