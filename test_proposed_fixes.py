import unittest
import numpy as np

# We assume the following classes are in a reachable path.
# Adjust the import path if your project structure is different.
from envs.bpp0.stability import Box, StackingTree

class TestSpaceVisualizer:
    """A helper class to manage and visualize a heightmap for the tests."""
    def __init__(self, width, length):
        self.width = width
        self.length = length
        self.plain = np.zeros((length, width), dtype=int)

    def add_box(self, box):
        """Updates the heightmap with a new box."""
        # Cast to int for grid placement
        x, y = int(box.x), int(box.y)
        lx, ly = int(box.lx), int(box.ly)
        z, lz = int(box.z), int(box.lz)
        
        # Ensure indices are within bounds
        end_y, end_x = min(y + ly, self.length), min(x + lx, self.width)
        self.plain[y:end_y, x:end_x] = z + lz

    def print_visualization(self, title, new_box=None):
        """Creates and prints a string visualization of the heightmap."""
        vis_map = self.plain.astype(str)
        vis_map[vis_map == '0'] = '.'

        if new_box:
            vis_map_with_box = vis_map.copy()
            
            # --- BUG FIX ---
            # Cast all box coordinates and dimensions to integers for grid drawing.
            # This allows the tests to use floats for precision while the viz works.
            x, y = int(new_box.x), int(new_box.y)
            lx, ly = int(new_box.lx), int(new_box.ly)
            
            # Draw the footprint with 'B'
            for r in range(y, y + ly):
                for c in range(x, x + lx):
                    if 0 <= r < self.length and 0 <= c < self.width:
                        vis_map_with_box[r, c] = 'B'
            
            # Mark the center of mass with 'X'
            com_x, com_y = int(new_box.centroid[0]), int(new_box.centroid[1])
            if 0 <= com_y < self.length and 0 <= com_x < self.width:
                 vis_map_with_box[com_y, com_x] = 'X'
            
            vis_map = vis_map_with_box

        header = f"\n--- {title} ---"
        col_headers = "   " + "".join([f"{i:<2}" for i in range(self.width)])
        separator = "  +" + "-" * (self.width * 2 - 1) + "+"
        
        map_str = ""
        for i, row in enumerate(vis_map):
            map_str += f"\n{i:2}| {' '.join(row)} |"

        print(header)
        print(col_headers)
        print(separator)
        print(map_str)
        print(separator)


class TestComprehensiveFeasibilityWithViz(unittest.TestCase):
    """
    A suite of focused tests for various stability scenarios, with visualizations.
    """

    def setUp(self):
        """Set up a fresh StackingTree for each test."""
        self.stacking_tree = StackingTree()
        print("\n" + "="*80)

    def test_A_stable_bridge(self):
        """🧪 SCENARIO 1: A stable bridge between two supporters."""
        print("DIAGNOSING: Stable Bridge")
        space = TestSpaceVisualizer(width=12, length=12)
        
        pillar_a = Box(box_id=1, x=0, y=1, z=0, lx=2, ly=10, lz=5)
        pillar_b = Box(box_id=2, x=8, y=1, z=0, lx=2, ly=10, lz=5)
        self.stacking_tree.add_box_permanently(pillar_a)
        self.stacking_tree.add_box_permanently(pillar_b)
        space.add_box(pillar_a)
        space.add_box(pillar_b)
        space.print_visualization("Initial State: Two Pillars")

        bridge = Box(box_id=3, x=1, y=1, z=5, lx=8, ly=10, lz=2)
        space.print_visualization("Testing Bridge Placement (CoM at 'X')", new_box=bridge)
        
        is_stable = self.stacking_tree.is_placement_stable(bridge, self.stacking_tree.boxes)
        self.assertTrue(is_stable, "FAIL: A centrally supported bridge should be STABLE.")
        print("    - Result: Bridge is correctly identified as STABLE.")

    def test_B_unstable_bridge_gap(self):
        """🧪 SCENARIO 2: An unstable bridge where the CoM falls in the gap."""
        print("DIAGNOSING: Unstable Bridge (Gap)")
        space = TestSpaceVisualizer(width=12, length=4)

        pillar_a = Box(box_id=1, x=0, y=1, z=0, lx=2, ly=2, lz=5)
        pillar_b = Box(box_id=2, x=8, y=1, z=0, lx=2, ly=2, lz=5)
        self.stacking_tree.add_box_permanently(pillar_a)
        self.stacking_tree.add_box_permanently(pillar_b)
        space.add_box(pillar_a)
        space.add_box(pillar_b)
        space.print_visualization("Initial State: Two Pillars")
        
        unstable_box = Box(box_id=3, x=3, y=1, z=5, lx=4, ly=2, lz=2)
        space.print_visualization("Testing Box Over Gap (CoM at 'X')", new_box=unstable_box)

        is_stable = self.stacking_tree.is_placement_stable(unstable_box, self.stacking_tree.boxes)
        self.assertFalse(is_stable, "FAIL: A box with its CoM in a gap should be UNSTABLE.")
        print("    - Result: Box over gap is correctly identified as UNSTABLE.")

    def test_C_stable_cantilever(self):
        """🧪 SCENARIO 3: A stable cantilever (box hanging off an edge)."""
        print("DIAGNOSING: Stable Cantilever")
        space = TestSpaceVisualizer(width=14, length=6)

        base = Box(box_id=1, x=1, y=1, z=0, lx=10, ly=4, lz=5)
        self.stacking_tree.add_box_permanently(base)
        space.add_box(base)
        space.print_visualization("Initial State: Base")

        cantilever = Box(box_id=2, x=9, y=2, z=5, lx=4, ly=2, lz=2) # CoM at x=11, on the edge
        space.print_visualization("Testing Cantilever (CoM at 'X')", new_box=cantilever)

        is_stable = self.stacking_tree.is_placement_stable(cantilever, self.stacking_tree.boxes)
        self.assertTrue(is_stable, "FAIL: A box with its CoM on the support edge should be STABLE.")
        print("    - Result: Cantilever is correctly identified as STABLE.")

    def test_D_unstable_cantilever(self):
        """🧪 SCENARIO 4: An unstable cantilever (box hanging too far)."""
        print("DIAGNOSING: Unstable Cantilever")
        space = TestSpaceVisualizer(width=14, length=6)

        base = Box(box_id=1, x=1, y=1, z=0, lx=10, ly=4, lz=5)
        self.stacking_tree.add_box_permanently(base)
        space.add_box(base)
        space.print_visualization("Initial State: Base")

        unstable_cantilever = Box(box_id=2, x=9.1, y=2, z=5, lx=4, ly=2, lz=2) # CoM at x=11.1
        space.print_visualization("Testing Unstable Cantilever (CoM at 'X')", new_box=unstable_cantilever)

        is_stable = self.stacking_tree.is_placement_stable(unstable_cantilever, self.stacking_tree.boxes)
        self.assertFalse(is_stable, "FAIL: A box with its CoM past the support edge should be UNSTABLE.")
        print("    - Result: Unstable cantilever is correctly identified as UNSTABLE.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
