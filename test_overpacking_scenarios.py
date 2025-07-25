# test_overpacking_scenarios.py
import gym
import numpy as np
import random
import envs  # This registers your correct environment

# Import the actual classes from your project
from envs.bpp0.binCreator import RandomBoxCreator

def run_overpacking_test():
    """
    Runs multiple packing episodes using the RandomBoxCreator to check if
    the environment correctly prevents overpacking.
    """
    print("\n--- Running Overpacking Scenarios Test ---")

    num_episodes = 5
    bin_size = (4, 4, 4)
    box_set = RandomBoxCreator.default_box_set
    overpacking_detected = False

    for i in range(num_episodes):
        print(f"\n--- Starting Episode {i + 1}/{num_episodes} ---")

        env = gym.make("Bpp-v0", container_size=bin_size, box_set=box_set, data_type="rs")
        env.seed(i)
        env.reset()

        done = False
        info = {"counter": 0}
        while not done:
            # --- NEW: Print Feasibility Masks Before Placement ---
            print(f"\n  --- Step {info.get('counter', 0) + 1} Analysis ---")
            print(f"  Next Box Dimensions: {env.next_box}")
            print("  Feasibility Mask (Orientation 0):")
            # Transpose for intuitive (x, y) printing and convert bool to int for clarity
            print(np.transpose(env.mask_o0).astype(int))
            # ---

            # Agent policy: find the first available valid spot
            action_to_take = None
            if env.mask_o0.any():
                valid_coords = np.argwhere(env.mask_o0)
                # np.argwhere gives (row, col) which is (y, x), action is (o, x, y)
                action_to_take = (0, valid_coords[0][1], valid_coords[0][0])
            
            if action_to_take:
                o, x, y = action_to_take
                obs, reward, done, info = env.step(action_to_take)
                print(
                    f"  Agent chose action (o={o}, x={x}, y={y}). Placed a {env.box_creator.box_list[-1]} box. Done={done}"
                )
            else:
                print("  Greedy agent found no valid moves. Ending episode.")
                done = True

        # Report final results for the episode
        final_info = info
        final_ratio = final_info.get("ratio", 0)
        final_items = final_info.get("counter", 0)

        print(f"\n--- Episode {i + 1} Finished ---")
        print(f"  Final items packed: {final_items}")
        print(f"  Final space ratio: {final_ratio:.4f} ({final_ratio*100:.2f}%)")
        
        print("  Final Bin State (Heightmap):")
        print(np.transpose(env.space.plain))

        if final_ratio > 1.0:
            overpacking_detected = True
            print("  [BUG DETECTED] Overpacking occurred in this episode!")

    print("\n--- Final Test Conclusion ---")
    if overpacking_detected:
        print("FAIL: The test detected at least one episode where the container was overpacked.")
    else:
        print("[PASS] No overpacking was detected. The environment is correctly enforcing its physical limits.")


if __name__ == "__main__":
    run_overpacking_test()