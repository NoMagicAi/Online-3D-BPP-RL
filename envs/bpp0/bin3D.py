# envs/bpp0/bin3D.py

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
        Calculates the stability map using a highly optimized vectorized approach.
        
        Args:
            box_size (tuple): The (width, length, height) of the box to check.
            
        Returns:
            np.ndarray: A 2D integer mask of valid placement positions.
        """
        heightmap = self.space.plain
        bin_w, bin_l = heightmap.shape
        box_w, box_l, box_h = box_size

        # If the box can't fit horizontally, no positions are valid.
        if box_w > bin_w or box_l > bin_l:
            return np.zeros_like(heightmap, dtype=np.int32)

        # 1. Create a view of all possible (box_w x box_l) patches.
        patches = sliding_window_view(heightmap, window_shape=(box_w, box_l))

        # 2. Calculate the max and min height for each patch vectorized.
        max_heights = np.max(patches, axis=(2, 3))
        min_heights = np.min(patches, axis=(2, 3))

        # 3. Check stability (all ground points are level) and vertical fit.
        is_stable = (max_heights == min_heights)
        fits_vertically = (max_heights + box_h <= self.height)
        
        # 4. A position is valid only if both conditions are met.
        valid_mask_small = np.logical_and(is_stable, fits_vertically)
        
        # 5. Create a full-sized mask and place the result in the top-left.
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

        if mask_to_check[x_pos, y_pos]:
            # Action is VALID
            alpha = 1.0  # Hyperparameter for volumetric reward
            
            # --- REWARD CALCULATION WITH AIR POCKET PENALTY ---
            # Calculate reward components based on the state BEFORE the action
            volumetric_reward = alpha * self.get_box_ratio()
            bin_volume = self.space.width * self.space.length * self.space.height

            box_to_place = self.next_box
            box_volume = box_to_place[0] * box_to_place[1] * box_to_place[2]
            sum_before = np.sum(self.space.plain)

            # Execute the action and change the state
            self.space.drop_box(self.next_box, (x_pos, y_pos), bool(orientation))
            
            # Calculate the penalty based on the change in state
            sum_after = np.sum(self.space.plain)
            air_pocket_volume = sum_after - sum_before - box_volume
            
            # The final reward is the volumetric reward minus the normalized penalty
            reward = volumetric_reward - 1.5 * (air_pocket_volume / bin_volume)

            # Advance to the next box and update the state for the next step
            self.box_creator.drop_box()
            self.box_creator.generate_box_size()
            self._update_masks()
            done = not (self.mask_o0.any() or self.mask_o1.any())
        else:
            # Action is INVALID
            done = True
            reward = 0.0

        info = {
            "counter": len(self.space.stacking_tree.boxes),
            "ratio": self.space.get_ratio(),
        }

        # --- ADDED CODE: Add final state to info dict on completion ---
        if done:
            # This logs the state only if the episode ended because there were no more valid moves,
            # which indicates a successful packing attempt.
            if not (self.mask_o0.any() or self.mask_o1.any()):
                info['final_heightmap'] = self.space.plain
                info['final_boxes'] = self.space.stacking_tree.boxes
        # --- END OF ADDED CODE ---

        return self.cur_observation, reward, done, info