# run_center_agent.py
import torch
import gym

# Import the necessary local modules
import envs
from acktr.envs import make_vec_envs

# A simple class to replace SimpleNamespace for compatibility
class MockArgs:
    pass

def run_center_agent():
    """
    Initializes the environment and runs a single episode with an agent
    that always tries to place items in the center of the bin.
    """
    print("--- Running Center Agent ---")

    # 1. Configure and create the environment
    mock_args = MockArgs()
    mock_args.enable_rotation = True
    mock_args.box_size_set = [(4, 4, 4), (5, 5, 5)]
    mock_args.container_size = (10, 10, 10)
    mock_args.data_type = 'rs'
    
    env = make_vec_envs(
        env_name='Bpp-v0',
        seed=49,
        num_processes=1,
        gamma=None,
        log_dir=None,
        device='cpu',
        allow_early_resets=False,
        args=mock_args
    )

    # 2. Define the agent's simple policy
    # The action is [x_pos, y_pos, orientation]
    # We choose the center (5, 5) and no rotation (0).
    center_action = torch.tensor([[0, 5, 5]], dtype=torch.long)
    
    # 3. Run a single episode
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    max_steps = 50 # Set a limit to prevent infinite loops

    while not done and step_count < max_steps:
        step_count += 1
        obs, reward, done, info = env.step(center_action)
        
        # The info dictionary is wrapped in a list by the VecEnv
        current_info = info[0]
        
        print(
            f"Step: {step_count}, "
            f"Items Packed: {current_info.get('counter', 0)}, "
            f"Reward: {reward.item():.4f}, "
            f"Done: {done.item()}"
        )
        total_reward += reward.item()

    # 4. Print final results
    final_info = info[0]
    print("\n--- Episode Finished ---")
    print(f"Total steps: {step_count}")
    print(f"Final items packed: {final_info.get('counter', 0)}")
    print(f"Final space utilization: {final_info.get('ratio', 0) * 100:.2f}%")
    print(f"Total reward: {total_reward:.4f}")

    env.close()

if __name__ == '__main__':
    run_center_agent()