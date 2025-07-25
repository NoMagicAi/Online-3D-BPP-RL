import unittest
import numpy as np

from envs.bpp0.stability import Box, StackingTree
# We assume the Box and StackingTree classes from the previous context are available.
# from stability_code import Box, StackingTree

class TestPinpointRootCause(unittest.TestCase):
    """
    A new suite of minimal, targeted tests to diagnose the root cause of stability failures.
    Each test focuses on one specific physical principle to ensure the simulation logic is correct.
    """
    def setUp(self):
        """Set up a fresh StackingTree for each test."""
        self.stacking_tree = StackingTree()
        print("\n" + "="*70)

    def test_E_single_box_tipping(self):
        """🧪 SCENARIO E: The simplest tipping case. A box's CoM is outside its support."""
        print("DIAGNOSING: Basic Tipping (CoM outside support polygon)")

        # A single, wide base box on the floor.
        base = Box(box_id=1, x=0, y=0, z=0, lx=10, ly=10, lz=2)
        self.stacking_tree.add_box_permanently(base)
        print(f"  - Placed Base (A) with support from x=0 to x=10.")

        # A second box placed so its CoM is just off the edge of the base.
        # Support from Base ends at x=10.
        # Tipping box starts at x=9, has length 4. Its CoM is at x = 9 + (4/2) = 11.
        tipping_box = Box(box_id=None, x=9, y=0, z=2, lx=4, ly=4, lz=2)
        print(f"  - Checking Tipping Box (B) with CoM at x={tipping_box.centroid[0]}.")
        
        # The support polygon for the tipping_box is the contact area with the base,
        # which is a rectangle from x=9 to x=10.
        # The CoM at x=11 is clearly outside this polygon.
        is_stable = self.stacking_tree.is_placement_stable(tipping_box, self.stacking_tree.boxes)

        self.assertFalse(is_stable, "FAIL: A box with its CoM outside its direct support polygon MUST be unstable.")
        print("   - Result: Basic tipping correctly identified as UNSTABLE.")


    def test_F_minimal_chain_reaction(self):
        """🧪 SCENARIO F: A minimal 2-level chain reaction. C makes B unstable on A."""
        print("DIAGNOSING: Minimal Chain-Reaction Instability")

        # Box A: The ground support.
        box_a = Box(box_id=1, x=0, y=0, z=0, lx=10, ly=10, lz=2)
        self.stacking_tree.add_box_permanently(box_a)
        print("  - Placed Base (A).")

        # Box B: Rests on A, but close to the edge. Stable by itself.
        # Support for B on A is the contact patch, from x=8 to x=10.
        box_b = Box(box_id=2, x=8, y=0, z=2, lx=2, ly=10, lz=2) # CoM at x=9.
        self.stacking_tree.add_box_permanently(box_b)
        print(f"  - Placed Box (B) on A. Stable. Support for B is [8, 10]. CoM_B is at x={box_b.centroid[0]}.")

        # Box C: Placed on B. This new weight should shift the combined CoM of {B+C}
        # off the support provided by A.
        box_c = Box(box_id=None, x=8, y=0, z=4, lx=2, ly=10, lz=2, density=3.0) # CoM at x=9, but 3x heavier.
        print(f"  - Checking heavy Box (C) on B. CoM_C is at x={box_c.centroid[0]}.")

        # PHYSICS TRACE:
        # - CoM of B is at x=9. Mass_B = 2*10*2 = 40.
        # - CoM of C is at x=9. Mass_C = 40 * 3 = 120.
        # - The force from C is applied to B.
        # - We check the stability of the combined {B+C} system on A.
        # - Combined CoM of {B+C} = (CoM_B*Mass_B + CoM_C*Mass_C) / (Mass_B + Mass_C)
        # - This is incorrect. The force from C is applied at C's CoM (x=9) onto B.
        # - The recursive check calculates the new CoM of B under this load.
        # - New CoM of B = (CoM_B*Mass_B + CoM_C*Mass_C) / (Mass_B+Mass_C)
        # - This is still wrong. The recursive check is more complex.
        
        # Let's use a clearer tipping case for C.
        box_c_tipping = Box(box_id=None, x=9, y=0, z=4, lx=2, ly=10, lz=2) # CoM at x=10. Mass=40.
        print(f"  - Re-checking with Box (C) with CoM at x={box_c_tipping.centroid[0]}.")
        
        # TRACE for recursive check on B:
        # - Hypothetical CoM of {B+C} = (9*40 + 10*40) / 80 = 9.5.
        # - This CoM (9.5) is checked against B's support polygon on A, which is [8, 10].
        # - 9.5 is inside [8, 10], so this level is stable. Let's make it more extreme.

        box_c_extreme = Box(box_id=None, x=10, y=0, z=4, lx=2, ly=10, lz=2) # CoM at x=11. Mass=40.
        print(f"  - Re-checking with Box (C) with CoM at x={box_c_extreme.centroid[0]}.")

        # TRACE for recursive check on B:
        # - Hypothetical CoM of {B+C} = (CoM_B*Mass_B + added_CoM*added_Mass) / (Mass_B + added_Mass)
        # - The 'added_CoM' is the CoM of C (x=11). The 'added_Mass' is Mass_C (40).
        # - Hypothetical CoM of {B+C} = (9*40 + 11*40) / (40+40) = (360 + 440) / 80 = 800 / 80 = 10.0.
        # - The support for B on A is [8, 10]. The point 10.0 is on the boundary, which should be stable.
        # Let's shift it just a tiny bit more.
        
        box_c_final = Box(box_id=None, x=10.1, y=0, z=4, lx=2, ly=10, lz=2) # CoM at x=11.1
        print(f"  - Final check with Box (C) with CoM at x={box_c_final.centroid[0]}.")
        # - Hypothetical CoM of {B+C} = (9*40 + 11.1*40) / 80 = (9 + 11.1) / 2 = 10.05.
        # - The support for B on A is [8, 10].
        # - The combined CoM at 10.05 is OUTSIDE the support. This MUST be unstable.

        is_stable = self.stacking_tree.is_placement_stable(box_c_final, self.stacking_tree.boxes)

        self.assertFalse(is_stable, "FAIL: The load from C should make B unstable on A.")
        print("   - Result: Minimal chain reaction correctly identified as UNSTABLE.")


    def test_G_bridge_instability(self):
        """🧪 SCENARIO G: A bridge that is unstable because the load is between supports."""
        print("DIAGNOSING: Bridge Instability (Load between supports)")

        # Two narrow pillars acting as a base.
        pillar1 = Box(box_id=1, x=0, y=0, z=0, lx=2, ly=2, lz=5)
        pillar2 = Box(box_id=2, x=8, y=0, z=0, lx=2, ly=2, lz=5)
        self.stacking_tree.add_box_permanently(pillar1)
        self.stacking_tree.add_box_permanently(pillar2)
        print("  - Placed two pillars with a gap from x=2 to x=8.")

        # A long bridge placed on top.
        bridge = Box(box_id=3, x=0, y=0, z=5, lx=10, ly=2, lz=1) # CoM at x=5
        
        # PHYSICS TRACE:
        # - The bridge's CoM is at x=5.
        # - The supporters are pillar1 (support at x=[0,2]) and pillar2 (support at x=[8,10]).
        # - To support a load at x=5, pillar2 would need to pull down (tension),
        #   which is physically impossible.
        # - The _calculate_force_distribution should return None or negative forces.
        
        is_stable = self.stacking_tree.is_placement_stable(bridge, self.stacking_tree.boxes)
        
        self.assertFalse(is_stable, "FAIL: A bridge with its CoM between its two supports MUST be unstable.")
        print("   - Result: Bridge instability correctly identified as UNSTABLE.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
