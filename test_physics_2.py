import unittest
import numpy as np
from scipy.spatial import ConvexHull as ScipyConvexHull

# We assume the following classes are in a reachable path.
# Adjust the import path if your project structure is different.
from stability import Box, StackingTree

# --- Test Suite 1: The Most Fundamental Class, Box ---
# If these tests fail, nothing else in the stability code can work correctly.

class TestBoxConstructor(unittest.TestCase):
    """
    This suite tests the VERY first step: Is the Box object created correctly?
    It specifically checks the mass and centroid calculations in your Box.__init__ method.
    """
    def setUp(self):
        print("\n" + "="*70)
        print("DIAGNOSING: Box Class Constructor (`__init__`)")

    def test_A_mass_calculation(self):
        """[Box] Test 1: Is mass calculated from DIMENSIONS (lx, ly, lz)?"""
        # A box of size 2x3x4 should have a mass/volume of 2*3*4 = 24.
        # Its position (10, 20) should be irrelevant to its mass.
        print("  - Running: test_A_mass_calculation")
        box = Box(box_id=1, x=10, y=20, z=0, lx=2, ly=3, lz=4)
        expected_mass = 2 * 3 * 4 * 1.0
        
        print(f"    - Box created with size (lx=2, ly=3, lz=4) at pos (x=10, y=20)")
        print(f"    - EXPECTED mass: {expected_mass}")
        print(f"    - ACTUAL mass:   {box.mass}")
        
        self.assertAlmostEqual(box.mass, expected_mass, 
                               msg="FAIL: Mass is not calculated correctly. It should be lx * ly * lz.")
        print("    - RESULT: PASS")

    def test_B_centroid_calculation(self):
        """[Box] Test 2: Is the centroid calculated from POSITION and DIMENSIONS?"""
        # A box starting at (x=10, y=20) with size (lx=4, ly=6) should have
        # a centroid at (x + lx/2, y + ly/2) = (10 + 2, 20 + 3) = (12, 23).
        print("  - Running: test_B_centroid_calculation")
        box = Box(box_id=1, x=10, y=20, z=0, lx=4, ly=6, lz=1)
        expected_centroid = np.array([12.0, 23.0])

        print(f"    - Box created with size (lx=4, ly=6) at pos (x=10, y=20)")
        print(f"    - EXPECTED centroid: {expected_centroid}")
        print(f"    - ACTUAL centroid:   {box.centroid}")

        np.testing.assert_array_almost_equal(box.centroid, expected_centroid,
                                             err_msg="FAIL: Centroid is not calculated correctly. It should be [x + lx/2, y + ly/2].")
        print("    - RESULT: PASS")


# --- Test Suite 2: Fundamental Geometric Helper Functions ---
# These tests depend on the Box constructor being correct.

class TestGeometricHelpers(unittest.TestCase):
    """
    This suite re-tests the geometric helpers `_get_contact_center` and `get_supporters`.
    It uses the same logic as before, but a failure here will now definitively
    point to a bug in these specific functions, assuming the Box constructor tests pass.
    """
    def setUp(self):
        self.stacking_tree = StackingTree()
        print("\n" + "="*70)
        print("DIAGNOSING: Geometric Helper Functions")

    def test_C_contact_center(self):
        """[Helpers] Test 3: Does _get_contact_center find the correct intersection?"""
        print("  - Running: test_C_contact_center")
        bottom_box = Box(box_id=1, x=0, y=0, z=0, lx=10, ly=10, lz=5)
        top_box = Box(box_id=None, x=5, y=5, z=5, lx=10, ly=10, lz=5)
        expected_center = np.array([7.5, 7.5])

        print(f"    - Testing contact between box at (0,0) and box at (5,5)")
        print(f"    - EXPECTED contact center: {expected_center}")
        
        contact_center = self.stacking_tree._get_contact_center(top_box, bottom_box)
        print(f"    - ACTUAL contact center:   {contact_center}")

        np.testing.assert_array_almost_equal(contact_center, expected_center,
                                              err_msg="FAIL: _get_contact_center is incorrect.")
        print("    - RESULT: PASS")

    def test_D_get_supporters(self):
        """[Helpers] Test 4: Does get_supporters find a simple, valid supporter?"""
        print("  - Running: test_D_get_supporters")
        box_a = Box(box_id=1, x=0, y=0, z=0, lx=10, ly=10, lz=5)
        hypo_box_b = Box(box_id=None, x=2, y=2, z=5, lx=5, ly=5, lz=5)

        print(f"    - Testing if box at (0,0) supports box at (2,2) on top")
        supporters = self.stacking_tree.get_supporters(hypo_box_b, [box_a])
        
        print(f"    - EXPECTED supporters: 1")
        print(f"    - ACTUAL supporters:   {len(supporters)}")

        self.assertEqual(len(supporters), 1, "FAIL: get_supporters failed to find a valid supporter.")
        print("    - RESULT: PASS")


# --- Test Suite 3: Advanced Physics Logic ---
# These tests depend on all previous tests passing.

class TestPhysicsLogic(unittest.TestCase):
    """
    This suite tests the high-level physics: support polygons and force distribution.
    A failure here, if the above tests pass, points to a bug in the physics calculations.
    """
    def setUp(self):
        self.stacking_tree = StackingTree()
        print("\n" + "="*70)
        print("DIAGNOSING: Physics and Force Distribution Logic")

    def test_E_support_polygon(self):
        """[Physics] Test 5: Does _get_support_polygon form the correct shape?"""
        print("  - Running: test_E_support_polygon")
        # A box supported by two smaller boxes.
        supporter1 = Box(box_id=1, x=0, y=0, z=0, lx=2, ly=2, lz=2)
        supporter2 = Box(box_id=2, x=4, y=0, z=0, lx=2, ly=2, lz=2)
        top_box = Box(box_id=3, x=1, y=0, z=2, lx=4, ly=2, lz=2)
        top_box.supported_by = [supporter1, supporter2]

        # The contact area with supporter1 is from (1,0) to (2,2).
        # The contact area with supporter2 is from (4,0) to (5,2).
        # The convex hull should be a rectangle from (1,0) to (5,2).
        # The expected vertices are (1,0), (5,0), (5,2), (1,2) in some order.
        support_polygon = self.stacking_tree._get_support_polygon(top_box)
        self.assertIsNotNone(support_polygon, "FAIL: Support polygon should not be None.")
        
        # Check if the area is correct: (5-1) * (2-0) = 8
        # Scipy's ConvexHull object has an 'area' attribute.
        hull = ScipyConvexHull(support_polygon)
        print(support_polygon)
        print(f"Data type of support_polygon: {support_polygon.dtype}")
        print(f"    - EXPECTED support polygon area: ~8.0")
        print(f"    - ACTUAL support polygon area:   {hull.volume}")
        self.assertAlmostEqual(hull.volume, 8.0, msg="FAIL: The support polygon has the wrong shape/area.")
        print("    - RESULT: PASS")


if __name__ == '__main__':
    # Running all test suites
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBoxConstructor))
    suite.addTest(unittest.makeSuite(TestGeometricHelpers))
    suite.addTest(unittest.makeSuite(TestPhysicsLogic))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)
