import numpy as np
from scipy.ndimage import maximum_filter

# Assuming these imports point to your custom classes
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
        Calculates a feasibility map for placing an item.

        This version uses scipy.ndimage.maximum_filter for efficient calculation
        of the placement surface height across all possible positions.
        """
        item_x, item_y, item_z = item_size

        # If the item is larger than the bin in any dimension, it's impossible to place.
        if item_x > self.width or item_y > self.length or item_z > self.height:
            return np.zeros_like(self.plain, dtype=bool)

        # --- OPTIMIZED LOGIC USING maximum_filter ---

        # 1. Calculate the surface height for all possible footprints at once.
        # The 'origin' parameter is key. It shifts the filter's anchor to its
        # top-left corner, so the output at [r, c] corresponds to the max
        # of the input slice starting at [r, c].
        origin_x = -((item_x) // 2)
        origin_y = -((item_y) // 2)

        surface_heights = maximum_filter(
            self.plain,
            size=(item_x, item_y),
            origin=(origin_x, origin_y),
            mode="constant",  # Use 'constant' to handle edges of the bin
            cval=0.0,  # Treat area outside the bin as floor level 0
        )

        # Create the map that will hold the feasibility for each possible placement.
        feasibility_map = np.zeros_like(self.plain, dtype=bool)

        # Create a single hypothetical box to reuse for stability checks
        hypo_box = Box(
            box_id=None, x=0, y=0, z=0, lx=item_x, ly=item_y, lz=item_z, density=density
        )

        # Iterate through all possible TOP-LEFT corner positions (r, c)
        for r in range(self.width - item_x + 1):
            for c in range(self.length - item_y + 1):
                # 2. Get the pre-calculated surface height for this placement
                surface_height = surface_heights[r, c]

                # 3. Check for vertical feasibility (does it stick out the top?)
                if surface_height + item_z > self.height:
                    continue  # This placement is invalid, move to the next one

                # 4. If vertically feasible, check for physical stability.
                # Update the hypothetical box with the current placement info.
                hypo_box.x = r
                hypo_box.y = c
                hypo_box.z = surface_height
                hypo_box.centroid[0] = r + item_x / 2.0
                hypo_box.centroid[1] = c + item_y / 2.0

                if self.stacking_tree.is_placement_stable(
                    hypo_box, self.stacking_tree.boxes
                ):
                    # If all checks pass, mark this corner as a valid placement
                    feasibility_map[r, c] = True

        return feasibility_map
        # --- END OF OPTIMIZED LOGIC ---

    def drop_box(self, box_size, position, flag, density=1.0):
        x_pos, y_pos = position
        item_x, item_y, item_z = box_size
        if flag:
            item_x, item_y = item_y, item_x

        # This calculation is correct because it uses the actual footprint
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

    def get_ratio(self):
        if not self.stacking_tree.boxes:
            return 0.0

        total_box_volume = sum(
            box.lx * box.ly * box.lz for box in self.stacking_tree.boxes
        )
        total_bin_volume = self.width * self.length * self.height

        return total_box_volume / total_bin_volume if total_bin_volume > 0 else 0.0
