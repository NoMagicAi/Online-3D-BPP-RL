# test_advanced_features.py
import torch
import gym
import numpy as np
import envs

from acktr.envs import make_vec_envs
from acktr.model import Policy

class MockArgs:
    pass

def test_far_to_near_reward():
    """
    Tests the implementation of the far-to-near reward shaping.
    """
    print("\n--- Testing Far-to-Near Reward ---")
    
    # 1. Setup the environment
    mock_args = MockArgs()
    mock_args.enable_rotation = False
    mock_args.box_size_set = [(2, 2, 2), (5, 5, 5)]
    mock_args.container_size = (10, 10, 10)
    mock_args.data_type = 'rs'
    
    env = make_vec_envs(
        env_name='Bpp-v0', seed=1, num_processes=1, gamma=None,
        log_dir=None, device='cpu', allow_early_resets=False, args=mock_args
    ).envs[0].env

    # 2. Test Case: Empty & Blocked Bin V_safe
    env.reset()
    assert env._calculate_v_safe() == 1000, "V_safe for empty bin is incorrect."
    print("[PASS] V_safe calculation is correct for an empty bin.")
    env.space.plain[:, 0] = 5
    assert env._calculate_v_safe() == 0, "V_safe for a blocked bin should be 0."
    print("[PASS] V_safe calculation is correct for a blocked bin.")

    # 3. Test Case: Robust reward calculation in a single step
    env.reset() # Reset to empty bin
    
    # --- ROBUSTNESS FIX ---
    # Get the actual next box from the environment instead of assuming a size
    next_item_dims = env.next_box
    item_volume = next_item_dims[0] * next_item_dims[1] * next_item_dims[2]
    bin_volume = env.space.width * env.space.length * env.space.height
    
    # Calculate expected reward based on the ACTUAL next box.
    # On the first step of an empty bin, V_safe is the total bin volume.
    expected_vol_reward = 10.0 * (item_volume / bin_volume)
    expected_safety_reward = 0.1 * (env._calculate_v_safe() / bin_volume)
    expected_reward = expected_vol_reward + expected_safety_reward
    
    action = (0, 0, 0) # Place at (0,0) with no rotation
    obs, reward, done, info = env.step(action)
    
    assert abs(reward - expected_reward) < 1e-6, f"Combined reward is incorrect. Expected ~{expected_reward:.4f}, got {reward:.4f}"
    print("[PASS] Combined reward in env.step() is calculated correctly.")

def test_infeasibility_loss():
    """
    Tests the calculation of the E_inf (infeasibility_loss).
    """
    print("\n--- Testing Infeasibility Loss (E_inf) ---")
    
    # 1. Setup the model and dummy inputs
    mock_args = MockArgs() # Use a dummy args for model setup
    mock_args.container_size = (10,10,10)
    
    action_space = gym.spaces.MultiDiscrete([2, 10, 10])
    obs_shape = (10 * 10 * 6,)
    
    policy = Policy(obs_shape, action_space, base_kwargs={'hidden_size': 256})
    
    batch_size = 4
    obs = torch.rand(batch_size, *obs_shape)
    action = torch.tensor([action_space.sample() for _ in range(batch_size)])
    rnn_hxs = torch.zeros(batch_size, policy.recurrent_hidden_state_size)
    masks = torch.zeros(batch_size, 1)

    # 2. Test Case: Fully Feasible Mask
    # Create a ground-truth mask where all actions are valid (all ones)
    gt_masks_all_valid = torch.ones(batch_size, 2, 10, 10)
    
    _, _, _, _, e_inf_valid = policy.evaluate_actions(
        obs, rnn_hxs, masks, action, gt_masks_all_valid)
        
    assert e_inf_valid.item() < 1e-6, "E_inf should be near zero for a fully valid mask."
    print("[PASS] Infeasibility loss is near zero for a fully feasible mask.")

    # 3. Test Case: Fully Infeasible Mask
    # Create a ground-truth mask where all actions are invalid (all zeros)
    gt_masks_all_invalid = torch.zeros(batch_size, 2, 10, 10)

    _, _, _, _, e_inf_invalid = policy.evaluate_actions(
        obs, rnn_hxs, masks, action, gt_masks_all_invalid)
        
    # The sum of probabilities for an entire orientation's map should be ~1.0.
    # Since the mask is all zeros, the infeasibility loss should be ~1.0.
    assert abs(e_inf_invalid.item() - 1.0) < 1e-3, "E_inf should be near 1.0 for a fully invalid mask."
    print("[PASS] Infeasibility loss is near 1.0 for a fully infeasible mask.")

if __name__ == '__main__':
    test_far_to_near_reward()
    test_infeasibility_loss()