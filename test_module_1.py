# test_module_1.py
import gym
import torch

# By importing 'envs', we run the __init__.py file inside it,
# which executes the gym.register() command.
import envs

from acktr.envs import make_vec_envs
from acktr.storage import RolloutStorage


# A simple class to replace SimpleNamespace for compatibility
class MockArgs:
    pass


def test_environment_and_storage():
    print("--- Testing Module 1: Environment and Storage ---")

    # 1. Create a mock 'args' object to configure the environment
    mock_args = MockArgs()
    mock_args.enable_rotation = True
    mock_args.box_size_set = [(2, 2, 2), (5, 5, 5)]
    mock_args.container_size = (10, 10, 10)
    mock_args.data_type = "rs"

    # 2. Call make_vec_envs, which will internally call gym.make('Bpp-v0')
    env = make_vec_envs(
        env_name="Bpp-v0",
        seed=1,
        num_processes=1,
        gamma=None,
        log_dir=None,
        device="cpu",
        allow_early_resets=False,
        args=mock_args,
    )

    # --- Verification Steps ---

    # Test 1: Action space
    assert isinstance(
        env.action_space, gym.spaces.MultiDiscrete
    ), "Action space is not MultiDiscrete"
    expected_shape = [2, 10, 10]  # [num_orientations, width, length]
    assert (
        env.action_space.nvec.tolist() == expected_shape
    ), f"Action space shape is wrong: {env.action_space.nvec.tolist()}"
    print("[PASS] Action space is correctly configured.")

    # Test 2: Observation space and Reset
    obs = env.reset()
    # The expected length is area * 6 channels (1 hmap + 3 item_dim + 2 feasibility_masks)
    expected_obs_len = mock_args.container_size[0] * mock_args.container_size[1] * 6
    # The vec_env adds a batch dimension, so we check the last dimension of the shape
    assert (
        obs.shape[-1] == expected_obs_len
    ), f"Observation length is wrong. Expected {expected_obs_len}, got {obs.shape[-1]}"
    print("[PASS] Observation space has the correct 6-channel length.")

    # Test 3: Step method
    sample_action = env.action_space.sample()
    try:
        # The vec_env expects actions to have a batch dimension
        obs, reward, done, info = env.step(torch.tensor(sample_action).unsqueeze(0))
        print(f"[PASS] env.step() executed with action: {sample_action}")
    except Exception as e:
        print(f"[FAIL] env.step() crashed with error: {e}")
        raise e

    # Test 4: RolloutStorage shape
    storage = RolloutStorage(
        num_steps=5,
        num_processes=1,
        obs_shape=env.observation_space.shape,
        action_space=env.action_space,
        recurrent_hidden_state_size=1,
        can_give_up=False,
        enable_rotation=True,
        pallet_size=10,
    )

    expected_action_shape = (5, 1, 3)
    assert (
        storage.actions.shape == expected_action_shape
    ), f"RolloutStorage actions shape is wrong: {storage.actions.shape}"
    # This assertion also implicitly tests the observation shape for the storage object
    expected_obs_shape = (
        6,
        1,
        expected_obs_len,
    )  # (num_steps + 1, num_processes, obs_len)
    assert (
        storage.obs.shape == expected_obs_shape
    ), f"RolloutStorage obs shape is wrong: {storage.obs.shape}"
    print("[PASS] RolloutStorage tensors have the correct shapes.")

    print("\n--- Module 1 Verification Complete ---")


if __name__ == "__main__":
    test_environment_and_storage()
