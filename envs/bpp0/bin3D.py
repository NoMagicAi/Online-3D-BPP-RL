import random
import numpy as np
import gym
import time
# Import the function for vectorized window operations
from numpy.lib.stride_tricks import sliding_window_view

from .space import Space
from .cutCreator import CuttingBoxCreator
from .mdCreator import MDlayerBoxCreator
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator


class PackingGame(gym.Env):
    def __init__(
        self,
        box_creator=None,
        container_size=(10, 10, 10),
        box_set=None,
        data_name=None,
        test=False,
        data_type="rs",
        enable_rotation=False,
        **kwags
    ):

        self.box_creator = box_creator
        self.bin_size = container_size
        self.width = self.bin_size[0]
        self.length = self.bin_size[1]
        self.height = self.bin_size[2]
        self.area = int(self.width * self.length)
        self.space = Space(*self.bin_size)
        self.can_rotate = enable_rotation

        # Stored masks to avoid re-computation
        self.mask_o0 = None
        self.mask_o1 = None

        if not test and box_creator is None:
            assert box_set is not None
            if data_type == "cut1":
                low = list(box_set[0])
                up = list(box_set[-1])
                low.extend(up)
                self.box_creator = CuttingBoxCreator(
                    container_size, low, self.can_rotate
                )
            elif data_type == "cut2":
                self.box_creator = MDlayerBoxCreator(
                    container_size, [box_set[0][0], box_set[-1][0]]
                )
            else:  # Defaults to 'rs'
                self.box_creator = RandomBoxCreator(box_set)

        self.obs_len = self.area * (1 + 3 + 2)
        num_orientations = 2 if self.can_rotate else 1
        self.action_space = gym.spaces.MultiDiscrete(
            [num_orientations, self.width, self.length]
        )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=self.height, shape=(self.obs_len,)
        )

    def _get_vectorized_stability_map(self, box_size):
        """
        Calculates the stability map using a highly optimized vectorized approach,
        incorporating realistic physical stability checks.
        
        A position is valid if it meets three criteria:
        1. Vertical Fit: The box does not exceed the container's height.
        2. Corner Support: At least 3 of the 4 corners of the box's base are
           supported at the same maximum corner height.
        3. Surface Area Support: A sufficient percentage of the box's base is
           supported at the true maximum height of the footprint.
        
        Args:
            box_size (tuple): The (width, length, height) of the box to check.
            
        Returns:
            np.ndarray: A 2D integer mask of valid placement positions.
        """
        heightmap = self.space.plain
        bin_w, bin_l = heightmap.shape
        box_w, box_l, box_h = box_size

        if box_w > bin_w or box_l > bin_l or box_w < 1 or box_l < 1:
            return np.zeros_like(heightmap, dtype=np.int32)

        # 1. Create a view of all possible (box_w x box_l) patches.
        patches = sliding_window_view(heightmap, window_shape=(box_w, box_l))

        # 2. Find the true maximum height for each potential placement footprint.
        max_heights = np.max(patches, axis=(2, 3))

        # 3. VERTICAL FIT check: Ensure the box doesn't exceed the container height.
        fits_vertically = (max_heights + box_h <= self.height)

        # 4. CORNER SUPPORT check (3-Corner Rule)
        # Extract heights of the four corners for all patches simultaneously.
        corners_c00 = patches[:, :, 0, 0]
        corners_c10 = patches[:, :, box_w - 1, 0]
        corners_c01 = patches[:, :, 0, box_l - 1]
        corners_c11 = patches[:, :, box_w - 1, box_l - 1]

        # Find the maximum height among the four corners for each patch.
        max_corner_heights = np.maximum.reduce([corners_c00, corners_c10, corners_c01, corners_c11])
        
        # Count how many corners are at that maximum height for each patch.
        supported_corners = (corners_c00 == max_corner_heights).astype(np.int8) + \
                            (corners_c10 == max_corner_heights).astype(np.int8) + \
                            (corners_c01 == max_corner_heights).astype(np.int8) + \
                            (corners_c11 == max_corner_heights).astype(np.int8)
        
        corner_rule_passed = (supported_corners >= 3)

        # 5. SURFACE AREA SUPPORT check.
        # Calculate the area supported at the true max height for each patch.
        # We expand dims of max_heights to allow direct comparison with patches.
        support_area = np.sum(patches == max_heights[:, :, np.newaxis, np.newaxis], axis=(2, 3))
        footprint_area = box_w * box_l
        support_ratio = support_area / footprint_area

        # Apply the tiered stability thresholds.
        rule1 = (support_ratio > 0.95)
        rule2 = (max_corner_heights == max_heights) & (supported_corners == 3) & (support_ratio > 0.85)
        rule3 = (max_corner_heights == max_heights) & (supported_corners == 4) & (support_ratio > 0.50)
        
        surface_rule_passed = rule1 | rule2 | rule3

        # 6. Combine all checks. A position is valid only if all rules pass.
        valid_mask_small = fits_vertically & corner_rule_passed & surface_rule_passed
        
        # 7. Create a full-sized mask and place the result in the top-left.
        full_mask = np.zeros_like(heightmap, dtype=np.int32)
        full_mask[:valid_mask_small.shape[0], :valid_mask_small.shape[1]] = valid_mask_small

        return full_mask

    def _update_masks(self):
        """
        A private helper to compute and store the feasibility masks for the current state
        using the new optimized vectorized method.
        """
        original_box = self.next_box
        self.mask_o0 = self._get_vectorized_stability_map(original_box)

        if self.can_rotate:
            rotated_box = (original_box[1], original_box[0], original_box[2])
            self.mask_o1 = self._get_vectorized_stability_map(rotated_box)
        else:
            self.mask_o1 = np.zeros_like(self.mask_o0)

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def get_box_plain(self):
        x_plain = np.ones((self.width, self.length), dtype=np.int32) * self.next_box[0]
        y_plain = np.ones((self.width, self.length), dtype=np.int32) * self.next_box[1]
        z_plain = np.ones((self.width, self.length), dtype=np.int32) * self.next_box[2]
        return (x_plain, y_plain, z_plain)

    @property
    def cur_observation(self):
        hmap = self.space.plain
        size = self.get_box_plain()
        return np.reshape(
            np.stack((hmap, *size, self.mask_o0, self.mask_o1)), newshape=(-1,)
        )

    @property
    def next_box(self):
        return self.box_creator.preview(1)[0]

    def reset(self):
        self.box_creator.reset()
        self.space = Space(*self.bin_size)
        self.box_creator.generate_box_size()
        self._update_masks()
        return self.cur_observation

    def get_box_ratio(self):
        coming_box = self.next_box
        box_vol = coming_box[0] * coming_box[1] * coming_box[2]
        bin_vol = self.space.width * self.space.length * self.space.height
        return box_vol / bin_vol if bin_vol > 0 else 0.0

    def step(self, action):
        orientation, x_pos, y_pos = action
        mask_to_check = self.mask_o1 if bool(orientation) else self.mask_o0
        
        info = {}

        if mask_to_check[x_pos, y_pos]:
            # Action is VALID
            alpha = 1.0  # Volumetric reward hyperparameter
            beta = 0.5   # Contact reward hyperparameter
            gamma = 1.5  # Air pocket penalty hyperparameter
            
            # --- REWARD CALCULATION (based on state BEFORE action) ---
            volumetric_reward = alpha * self.get_box_ratio()
            
            box = self.next_box
            if bool(orientation):
                box = (box[1], box[0], box[2])
            box_w, box_l, box_h = box
            heightmap = self.space.plain

            # Determine placement height based on the highest point(s) in the footprint
            footprint_under_box = heightmap[x_pos : x_pos + box_w, y_pos : y_pos + box_l]
            z_pos = np.max(footprint_under_box)

            # --- 1. Contact Surface Reward ---
            total_surface_area = 2 * (box_w * box_l + box_w * box_h + box_l * box_h)
            touching_area = 0.0
            
            bottom_contact_area = np.sum(footprint_under_box == z_pos)
            touching_area += bottom_contact_area
            
            # Side faces contact
            if x_pos == 0: touching_area += box_l * box_h
            else: touching_area += np.sum(np.maximum(0, np.minimum(z_pos + box_h, heightmap[x_pos - 1, y_pos:y_pos + box_l]) - z_pos))
            if x_pos + box_w == self.width: touching_area += box_l * box_h
            else: touching_area += np.sum(np.maximum(0, np.minimum(z_pos + box_h, heightmap[x_pos + box_w, y_pos:y_pos + box_l]) - z_pos))
            if y_pos == 0: touching_area += box_w * box_h
            else: touching_area += np.sum(np.maximum(0, np.minimum(z_pos + box_h, heightmap[x_pos:x_pos + box_w, y_pos - 1]) - z_pos))
            if y_pos + box_l == self.length: touching_area += box_w * box_h
            else: touching_area += np.sum(np.maximum(0, np.minimum(z_pos + box_h, heightmap[x_pos:x_pos + box_w, y_pos + box_l]) - z_pos))
            
            contact_reward = 0.0
            if total_surface_area > 0:
                contact_reward = beta * (touching_area / total_surface_area)

            # --- 2. Air Pocket Penalty ---
            # This is the volume of the empty space created under the box
            air_pocket_volume = z_pos * box_w * box_l - np.sum(footprint_under_box)
            bin_volume = self.width * self.length * self.height
            air_pocket_penalty = gamma * (air_pocket_volume / bin_volume)

            # --- 3. Final Reward ---
            reward = volumetric_reward + contact_reward - air_pocket_penalty

            # Execute the action and change the state
            self.space.drop_box(self.next_box, (x_pos, y_pos), bool(orientation))
            
            # Advance to the next box and update the state for the next step
            self.box_creator.drop_box()
            self.box_creator.generate_box_size()
            self._update_masks()
            done = not (self.mask_o0.any() or self.mask_o1.any())
            
            if done:
                info['failed_box_dims'] = self.next_box
                info['failed_box_pos'] = None
        else:
            # Action is INVALID
            done = True
            reward = 0.0
            
            box_being_placed = self.next_box
            if bool(orientation):
                box_being_placed = (box_being_placed[1], box_being_placed[0], box_being_placed[2])
            
            info['failed_box_dims'] = box_being_placed
            info['failed_box_pos'] = (x_pos, y_pos)

        info["counter"] = len(self.space.stacking_tree.boxes)
        info["ratio"] = self.space.get_ratio()

        if done:
            info['final_heightmap'] = self.space.plain
            info['final_boxes'] = self.space.stacking_tree.boxes
            
        return self.cur_observation, reward, done, info