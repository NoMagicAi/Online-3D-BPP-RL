import unittest
import numpy as np

# We assume the following classes are in a reachable path.
# Adjust the import path if your project structure is different.
from envs.bpp0.stability import Box, StackingTree


class TestGeometricHelpers(unittest.TestCase):
    """
    This test suite is a focused, diagnostic tool to validate the
    correctness of the two fundamental geometric helper functions
    in the StackingTree class: `get_supporters` and `_get_contact_center`.
    
    This version corrects the call to the Box constructor, using 'box_id'
    instead of 'id' to match the user's original code.
    """

    def setUp(self):
        """Create a dummy StackingTree instance to call the methods from."""
        self.stacking_tree = StackingTree()
        print("\n" + "="*70)

    # --- Tests for the get_supporters function ---

    def test_supporters_A_simple_overlap(self):
        """[Supporters] Case 1: A single, clear supporter."""
        # CORRECTED: Using box_id instead of id
        box_a = Box(box_id=1, x=0, y=0, z=0, lx=10, ly=10, lz=5)
        hypo_box_b = Box(box_id=None, x=2, y=2, z=5, lx=5, ly=5, lz=5)

        print(f"Running: {self.id()}")
        print(f"  - Supporter Box A: pos=(x:{box_a.x}, y:{box_a.y}, z:{box_a.z}), size=(lx:{box_a.lx}, ly:{box_a.ly}, lz:{box_a.lz})")
        print(f"  - Hypo Box B:      pos=(x:{hypo_box_b.x}, y:{hypo_box_b.y}, z:{hypo_box_b.z}), size=(lx:{hypo_box_b.lx}, ly:{hypo_box_b.ly}, lz:{hypo_box_b.lz})")

        supporters = self.stacking_tree.get_supporters(hypo_box_b, [box_a])
        
        self.assertEqual(len(supporters), 1, "FAIL: Expected to find exactly one supporter.")
        self.assertEqual(supporters[0].id, box_a.id, "FAIL: The wrong box was identified as a supporter.")
        print("  - Result: PASS")

    def test_supporters_B_no_overlap_xy(self):
        """[Supporters] Case 2: No overlap in the XY plane."""
        box_a = Box(box_id=1, x=0, y=0, z=0, lx=5, ly=5, lz=5)
        hypo_box_b = Box(box_id=None, x=10, y=10, z=5, lx=5, ly=5, lz=5)

        print(f"Running: {self.id()}")
        print(f"  - Supporter Box A: pos=(x:{box_a.x}, y:{box_a.y}, z:{box_a.z}), size=(lx:{box_a.lx}, ly:{box_a.ly}, lz:{box_a.lz})")
        print(f"  - Hypo Box B:      pos=(x:{hypo_box_b.x}, y:{hypo_box_b.y}, z:{hypo_box_b.z}), size=(lx:{hypo_box_b.lx}, ly:{hypo_box_b.ly}, lz:{hypo_box_b.lz})")

        supporters = self.stacking_tree.get_supporters(hypo_box_b, [box_a])

        self.assertEqual(len(supporters), 0, "FAIL: Found a supporter where none should exist (no XY overlap).")
        print("  - Result: PASS")

    def test_supporters_C_no_overlap_z(self):
        """[Supporters] Case 3: No overlap in the Z dimension (box is floating)."""
        box_a = Box(box_id=1, x=0, y=0, z=0, lx=10, ly=10, lz=5)
        hypo_box_b = Box(box_id=None, x=2, y=2, z=6, lx=5, ly=5, lz=5) # z=6 is incorrect

        print(f"Running: {self.id()}")
        print(f"  - Supporter Box A: pos=(x:{box_a.x}, y:{box_a.y}, z:{box_a.z}), size=(lx:{box_a.lx}, ly:{box_a.ly}, lz:{box_a.lz})")
        print(f"  - Hypo Box B:      pos=(x:{hypo_box_b.x}, y:{hypo_box_b.y}, z:{hypo_box_b.z}), size=(lx:{hypo_box_b.lx}, ly:{hypo_box_b.ly}, lz:{hypo_box_b.lz})")

        supporters = self.stacking_tree.get_supporters(hypo_box_b, [box_a])
        
        self.assertEqual(len(supporters), 0, "FAIL: Found a supporter where none should exist (incorrect Z level).")
        print("  - Result: PASS")
        
    def test_supporters_D_touching_edges(self):
        """[Supporters] Case 4: Boxes touching at the edges are NOT supporters."""
        box_a = Box(box_id=1, x=0, y=0, z=0, lx=5, ly=10, lz=5)
        hypo_box_b = Box(box_id=None, x=5, y=0, z=5, lx=5, ly=10, lz=5)

        print(f"Running: {self.id()}")
        print(f"  - Supporter Box A: pos=(x:{box_a.x}, y:{box_a.y}, z:{box_a.z}), size=(lx:{box_a.lx}, ly:{box_a.ly}, lz:{box_a.lz})")
        print(f"  - Hypo Box B:      pos=(x:{hypo_box_b.x}, y:{hypo_box_b.y}, z:{hypo_box_b.z}), size=(lx:{hypo_box_b.lx}, ly:{hypo_box_b.ly}, lz:{hypo_box_b.lz})")

        supporters = self.stacking_tree.get_supporters(hypo_box_b, [box_a])
        
        self.assertEqual(len(supporters), 0, "FAIL: Boxes only touching at the edge should not be supporters.")
        print("  - Result: PASS")

    def test_supporters_E_multiple_supporters(self):
        """[Supporters] Case 5: A box bridging two other boxes."""
        box_a = Box(box_id=1, x=0, y=0, z=0, lx=4, ly=10, lz=5)
        box_b = Box(box_id=2, x=6, y=0, z=0, lx=4, ly=10, lz=5)
        hypo_box_c = Box(box_id=None, x=2, y=0, z=5, lx=6, ly=10, lz=5)

        print(f"Running: {self.id()}")
        print(f"  - Supporter Box A: pos=(x:{box_a.x}, y:{box_a.y}, z:{box_a.z}), size=(lx:{box_a.lx}, ly:{box_a.ly}, lz:{box_a.lz})")
        print(f"  - Supporter Box B: pos=(x:{box_b.x}, y:{box_b.y}, z:{box_b.z}), size=(lx:{box_b.lx}, ly:{box_b.ly}, lz:{box_b.lz})")
        print(f"  - Hypo Box C:      pos=(x:{hypo_box_c.x}, y:{hypo_box_c.y}, z:{hypo_box_c.z}), size=(lx:{hypo_box_c.lx}, ly:{hypo_box_c.ly}, lz:{hypo_box_c.lz})")

        supporters = self.stacking_tree.get_supporters(hypo_box_c, [box_a, box_b])
        supporter_ids = {s.id for s in supporters}

        self.assertEqual(len(supporters), 2, "FAIL: Expected to find two supporters.")
        self.assertIn(box_a.id, supporter_ids, "FAIL: Box A was not found as a supporter.")
        self.assertIn(box_b.id, supporter_ids, "FAIL: Box B was not found as a supporter.")
        print("  - Result: PASS")

    # --- Tests for the _get_contact_center function ---

    def test_contact_center_A_partial_overlap(self):
        """[Contact Center] Case 1: Simple partial overlap."""
        bottom_box = Box(box_id=1, x=0, y=0, z=0, lx=10, ly=10, lz=5)
        top_box = Box(box_id=None, x=5, y=5, z=5, lx=10, ly=10, lz=5)
        expected_center = np.array([7.5, 7.5])

        print(f"Running: {self.id()}")
        print(f"  - Bottom Box: pos=(x:{bottom_box.x}, y:{bottom_box.y}), size=(lx:{bottom_box.lx}, ly:{bottom_box.ly})")
        print(f"  - Top Box:    pos=(x:{top_box.x}, y:{top_box.y}), size=(lx:{top_box.lx}, ly:{top_box.ly})")
        print(f"  - Expected Center of Contact: {expected_center}")

        contact_center = self.stacking_tree._get_contact_center(top_box, bottom_box)
        
        print(f"  - Calculated Center of Contact: {contact_center}")
        np.testing.assert_array_almost_equal(contact_center, expected_center,
                                              err_msg="FAIL: Incorrect contact center for partial overlap.")
        print(f"  - Calculated Center: {contact_center} -> PASS")

    def test_contact_center_B_full_containment(self):
        """[Contact Center] Case 2: One box fully contained in another."""
        bottom_box = Box(box_id=1, x=0, y=0, z=0, lx=20, ly=20, lz=5)
        top_box = Box(box_id=None, x=5, y=5, z=5, lx=4, ly=4, lz=5)
        expected_center = np.array([7.0, 7.0])

        print(f"Running: {self.id()}")
        print(f"  - Bottom Box: pos=(x:{bottom_box.x}, y:{bottom_box.y}), size=(lx:{bottom_box.lx}, ly:{bottom_box.ly})")
        print(f"  - Top Box:    pos=(x:{top_box.x}, y:{top_box.y}), size=(lx:{top_box.lx}, ly:{top_box.ly})")
        print(f"  - Expected Center of Contact: {expected_center}")

        contact_center = self.stacking_tree._get_contact_center(top_box, bottom_box)
        print(f"  - Calculated Center of Contact: {contact_center}")

        np.testing.assert_array_almost_equal(contact_center, expected_center,
                                              err_msg="FAIL: Incorrect contact center for a fully contained box.")
        print(f"  - Calculated Center: {contact_center} -> PASS")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
