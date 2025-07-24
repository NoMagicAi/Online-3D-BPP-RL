# test_advanced_features.py
import torch
import gym
import numpy as np
import envs

from acktr.envs import make_vec_envs
from acktr.model import Policy
# We need to import Space and Box to manipulate the environment state directly for testing
from envs.bpp0.space import Space
from envs.bpp0.stability import Box

class MockArgs:
    pass

# --- PREVIOUS TESTS (Still valuable) ---

def test_far_to_near_reward():
    print("\n--- Testing Far-to-Near Reward ---")
    mock_args = MockArgs()
    mock_args.enable_rotation = False
    mock_args.box_size_set = [(2, 2, 2), (5, 5, 5)]
    mock_args.container_size = (10, 10, 10)
    mock_args.data_type = 'rs'
    env = make_vec_envs(
        env_name='Bpp-v0', seed=1, num_processes=1, gamma=None,
        log_dir=None, device='cpu', allow_early_resets=False, args=mock_args
    ).envs[0].env
    env.reset()
    assert env._calculate_v_safe() == 1000
    print("[PASS] V_safe calculation is correct for an empty bin.")
    env.space.plain[:, 0] = 5
    assert env._calculate_v_safe() == 0
    print("[PASS] V_safe calculation is correct for a blocked bin.")
    env.reset()
    next_item_dims = env.next_box
    item_volume = next_item_dims[0] * next_item_dims[1] * next_item_dims[2]
    bin_volume = env.space.width * env.space.length * env.space.height
    expected_vol_reward = 10.0 * (item_volume / bin_volume)
    expected_safety_reward = 0.1 * (env._calculate_v_safe() / bin_volume)
    expected_reward = expected_vol_reward + expected_safety_reward
    _, reward, _, _ = env.step((0, 0, 0))
    assert abs(reward - expected_reward) < 1e-6
    print("[PASS] Combined reward in env.step() is calculated correctly.")

def test_infeasibility_loss():
    print("\n--- Testing Infeasibility Loss (E_inf) ---")
    mock_args = MockArgs()
    mock_args.container_size = (10,10,10)
    action_space = gym.spaces.MultiDiscrete([2, 10, 10])
    obs_shape = (10 * 10 * 6,)
    policy = Policy(obs_shape, action_space, base_kwargs={'hidden_size': 256})
    batch_size = 4
    obs = torch.rand(batch_size, *obs_shape)
    action = torch.tensor([action_space.sample() for _ in range(batch_size)])
    rnn_hxs = torch.zeros(batch_size, policy.recurrent_hidden_state_size)
    masks = torch.zeros(batch_size, 1)
    gt_masks_all_valid = torch.ones(batch_size, 2, 10, 10)
    _, _, _, _, e_inf_valid = policy.evaluate_actions(
        obs, rnn_hxs, masks, action, gt_masks_all_valid)
    assert e_inf_valid.item() < 1e-6
    print("[PASS] Infeasibility loss is near zero for a fully feasible mask.")
    gt_masks_all_invalid = torch.zeros(batch_size, 2, 10, 10)
    _, _, _, _, e_inf_invalid = policy.evaluate_actions(
        obs, rnn_hxs, masks, action, gt_masks_all_invalid)
    assert abs(e_inf_invalid.item() - 1.0) < 1e-3
    print("[PASS] Infeasibility loss is near 1.0 for a fully infeasible mask.")

# --- NEW ADVANCED TESTS ---

def test_cascading_instability():
    """
    Tests that the stacking tree correctly identifies a placement that makes a
    box *lower* in the stack unstable.
    """
    print("\n--- Testing Cascading Instability ---")
    space = Space(width=10, length=10, height=10)
    
    # 1. Place a wide base box (Box A)
    box_A = Box(box_id=None, x=0, y=0, z=0, lx=1, ly=8, lz=1)
    space.drop_box(box_size=None, position=(0,0), flag=False, _box_obj=box_A)

    # 2. Place a small box (Box B) precariously on the edge of Box A
    box_B = Box(box_id=None, x=0, y=7, z=1, lx=1, ly=1, lz=1)
    space.drop_box(box_size=None, position=(0,7), flag=False, _box_obj=box_B)

    # 3. Define a new box (Box C) to place directly on top of Box B
    item_to_place = (1, 1, 1) # Dimensions of Box C
    
    # The combined center of mass of (B+C) will fall off the edge of A,
    # making the whole stack unstable.
    
    # 4. Generate the stability map
    stability_map = space.get_stability_map(item_to_place)
    
    # 5. Assert that the placement location is correctly marked as INFEASIBLE
    is_placement_feasible = stability_map[0, 7] # The position of Box B
    assert not is_placement_feasible, "System failed to detect cascading instability."
    print("[PASS] Cascading instability correctly identified as infeasible.")

def test_convex_hull_support():
    """
    Tests that a box is stable if its center of mass is supported by the
    convex hull of multiple boxes below, as per the article (Fig. 3).
    """
    print("\n--- Testing Convex Hull Support ---")
    space = Space(width=10, length=10, height=10)
    
    # 1. Place two supporter boxes (A and B) with a gap between them
    box_A = Box(box_id=None, x=2, y=0, z=0, lx=2, ly=2, lz=2)
    space.drop_box(None, (2,0), False, _box_obj=box_A)
    box_B = Box(box_id=None, x=2, y=5, z=0, lx=2, ly=2, lz=2)
    space.drop_box(None, (2,5), False, _box_obj=box_B)
    
    # 2. Define a new box (Box C) that bridges the gap
    # Its center of mass will be at (x=3, y=3.5), which is between A and B
    item_to_place = (2, 7, 1)
    
    # 3. Generate the stability map
    stability_map = space.get_stability_map(item_to_place)
    
    # 4. Assert that the bridging placement is correctly marked as FEASIBLE
    is_placement_feasible = stability_map[2, 0]
    assert is_placement_feasible, "System failed to detect stability from convex hull support."
    print("[PASS] Stability via convex hull support correctly identified.")

def test_episode_termination():
    """
    Tests that the environment correctly sets done=True when the bin is full.
    """
    print("\n--- Testing Episode Termination ---")
    
    # 1. Setup a small environment that is easy to fill
    mock_args = MockArgs()
    mock_args.enable_rotation = False
    mock_args.box_size_set = [(2, 2, 2)] # Only one item type
    mock_args.container_size = (2, 2, 3) # Make the bin short
    mock_args.data_type = 'rs'
    
    env = make_vec_envs(
        env_name='Bpp-v0', seed=1, num_processes=1, gamma=None,
        log_dir=None, device='cpu', allow_early_resets=False, args=mock_args
    ).envs[0].env
    env.reset()

    # 2. Place the one and only possible box.
    # The 'done' flag returned by this step should be True, because after this
    # placement, there are no more valid moves for the next item.
    obs, reward, done, info = env.step((0, 0, 0))
    
    assert done, "Episode failed to terminate after the last possible move."
    print("[PASS] Environment correctly terminates the episode when the bin is full.")

if __name__ == '__main__':
    # Add a try-except block to the new tests as they manipulate the env directly
    # and need a modified drop_box method for easy setup.
    def add_temp_method(cls):
        def drop_box_for_test(self, box_size, position, flag, _box_obj=None, density=1.0):
            if _box_obj:
                final_box = _box_obj
            else:
                 # Original logic for real steps
                 lx, ly = position; item_x, item_y, item_z = box_size
                 if flag: item_x, item_y = item_y, item_x
                 surface_height = np.max(self.plain[lx : lx + item_x, ly : ly + item_y])
                 final_box = Box(box_id=None, x=lx, y=ly, z=surface_height, lx=item_x, ly=item_y, lz=item_z, density=density)
            self.stacking_tree.add_box_permanently(final_box)
            self.plain[final_box.x : final_box.x + final_box.lx, final_box.y : final_box.y + final_box.ly] = final_box.z + final_box.lz
            return True
        cls.drop_box = drop_box_for_test
    
    add_temp_method(Space)
    
    # Run all tests
    test_far_to_near_reward()
    test_infeasibility_loss()
    test_cascading_instability()
    test_convex_hull_support()
    test_episode_termination()