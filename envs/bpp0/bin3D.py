import random
import numpy as np
import gym
import time

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
        self.area = int(self.width * self.length)
        self.space = Space(*self.bin_size)
        self.can_rotate = enable_rotation

        # Stored masks to avoid re-computation
        self.mask_o0 = None
        self.mask_o1 = None

        if not test and box_creator is None:
            assert box_set is not None

            # --- THE FIX ---
            # This logic correctly selects the BoxCreator based on data_type
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
            low=0.0, high=self.space.height, shape=(self.obs_len,)
        )

    def _update_masks(self):
        """A private helper to compute and store the feasibility masks for the current state."""
        original_box = self.next_box
        self.mask_o0 = self.space.get_stability_map(original_box).astype(np.int32)

        if self.can_rotate:
            rotated_box = (original_box[1], original_box[0], original_box[2])
            self.mask_o1 = self.space.get_stability_map(rotated_box).astype(np.int32)
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
        """This property is now very fast as it just reads the pre-computed masks."""
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
        self._update_masks()  # Compute initial masks
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

            # --- REWARD CALCULATION FIX ---
            # Calculate the full reward based on the CURRENT state, BEFORE any changes.
            alpha = 1.0
            beta = 0.1

            volumetric_reward = alpha * self.get_box_ratio()

            box_to_place = self.next_box
            box_volume = box_to_place[0] * box_to_place[1] * box_to_place[2]
            sum_before = np.sum(self.space.plain)

            bin_volume = self.space.width * self.space.length * self.space.height
            #v_safe = self._calculate_v_safe()  # V_safe of the current state
            #safety_reward = beta * (v_safe / bin_volume)


            # Now, execute the action and change the state
            self.space.drop_box(self.next_box, (x_pos, y_pos), bool(orientation))
            sum_after = np.sum(self.space.plain)
            air_pocket_penalty = sum_after - sum_before - box_volume

            reward = volumetric_reward - 1.5 * (air_pocket_penalty/bin_volume)

            # Advance the item queue and compute masks for the *next* state
            self.box_creator.drop_box()
            self.box_creator.generate_box_size()
            #start_time = time.perf_counter()
            self._update_masks()
            #end_time = time.perf_counter()
            #print(f"Updating mask: {end_time - start_time} seconds")
            # Check if the new state is terminal
            done = not (self.mask_o0.any() or self.mask_o1.any())
        else:
            # Action is INVALID
            done = True
            reward = 0.0

        info = {
            "counter": len(self.space.stacking_tree.boxes),
            "ratio": self.space.get_ratio(),
        }
        return self.cur_observation, reward, done, info
