# envs/bpp0/space.py

import numpy as np
from scipy.ndimage import maximum_filter

# Import our new stability logic and the rich Box class
from .stability import StackingTree, Box


class Space(object):
    def __init__(self, width=10, length=10, height=10):
        self.width = width
        self.length = length
        self.height = height  # This is now a fixed constant

        # The state is composed of the heightmap and the stacking tree
        self.plain = np.zeros(shape=(width, length), dtype=np.int32)
        self.stacking_tree = StackingTree()

    def get_stability_map(self, item_size, density=1.0):
        """
        Generates a 2D map indicating all stable placement positions for a new item.
        """
        item_x, item_y, item_z = item_size

        # 1. Efficiently find the height of the surface at every possible placement location
        footprint = np.ones((item_x, item_y))
        max_h_map = maximum_filter(
            self.plain, footprint=footprint, mode="constant", cval=0
        )

        # 2. Find all positions that are vertically feasible (the overpacking check)
        vertically_feasible_mask = (max_h_map + item_z) <= self.height

        # 3. Iterate through only the feasible candidates to check for structural stability
        candidate_coords = np.argwhere(vertically_feasible_mask)
        feasibility_map = np.zeros_like(self.plain, dtype=bool)

        # Create a single reusable box object to avoid overhead in the loop
        hypothetical_box = Box(
            box_id=None, x=item_x, y=item_y, z=0, lx=0, ly=0, lz=item_z, density=density
        )

        for r, c in candidate_coords:
            placement_height = max_h_map[r, c]

            # Update the hypothetical box with the current candidate's properties
            hypothetical_box.z = placement_height
            hypothetical_box.lx = r
            hypothetical_box.ly = c
            hypothetical_box.centroid[0] = r + item_x / 2.0
            hypothetical_box.centroid[1] = c + item_y / 2.0

            # Use the stacking tree to check for full structural stability
            if self.stacking_tree.is_placement_stable(
                hypothetical_box, self.stacking_tree.boxes
            ):
                feasibility_map[r, c] = True

        return feasibility_map

    def drop_box(self, box_size, position, flag, density=1.0):
        x_pos, y_pos = position
        item_x, item_y, item_z = box_size
        if flag:
            item_x, item_y = item_y, item_x
        surface_height = np.max(
            self.plain[x_pos : x_pos + item_x, y_pos : y_pos + item_y]
        )
        final_box = Box(
            box_id=None,
            x=x_pos,
            y=y_pos,
            z=surface_height,
            lx=item_x,
            ly=item_y,
            lz=item_z,
            density=density,
        )
        self.stacking_tree.add_box_permanently(final_box)
        self.plain[x_pos : x_pos + item_x, y_pos : y_pos + item_y] = (
            surface_height + item_z
        )
        # NOTE: The buggy line 'self.height = max(...)' is correctly removed here.

    def get_ratio(self):
        if not self.stacking_tree.boxes:
            return 0.0

        # THIS IS THE CORRECT FORMULA: It uses the dimensions (lx, ly, lz)
        # from the new Box class.
        total_box_volume = sum(
            box.lx * box.ly * box.lz for box in self.stacking_tree.boxes
        )

        total_bin_volume = self.width * self.length * self.height
        return total_box_volume / total_bin_volume if total_bin_volume > 0 else 0.0

        # --- DEBUGGING PRINT STATEMENTS ---
        # This will show us exactly what the function is "seeing"
        # print("\n--- DEBUG: Inside get_ratio() ---")
        # print(f"Number of boxes: {len(self.stacking_tree.boxes)}")
        # Print dimensions of the first 10 boxes for inspection
        # for i, box in enumerate(self.stacking_tree.boxes[:10]):
        #    print(f"  Box {i}: Dims (lx, ly, lz) = ({box.lx}, {box.ly}, {box.lz})")
        # print(f"Calculated Total Box Volume: {total_box_volume}")
        # print(f"Total Bin Volume: {total_bin_volume}")
        # print(f"Calculated Ratio: {ratio:.4f}")
        # print("---------------------------------\n")

        return ratio
