# test_overpacking_scenarios.py
import gym
import numpy as np
import random
import envs  # This registers your correct environment

# Import the actual classes from your project
from envs.bpp0.bin3D import PackingGame
from envs.bpp0.binCreator import RandomBoxCreator  # Import RandomBoxCreator


def run_overpacking_test():
    """
    Runs multiple packing episodes using the RandomBoxCreator to check if
    the environment correctly prevents overpacking.
    """
    print("\n--- Running Overpacking Scenarios Test ---")

    num_episodes = 5
    bin_size = (4, 4, 4)

    # --- THIS IS THE FIX ---
    # Use the default box set directly from the RandomBoxCreator class
    box_set = RandomBoxCreator.default_box_set

    overpacking_detected = False

    for i in range(num_episodes):
        print(f"\n--- Starting Episode {i + 1}/{num_episodes} ---")

        # 1. Create a fresh instance of the ACTUAL environment for each episode
        env = gym.make(
            "Bpp-v0", container_size=bin_size, box_set=box_set, data_type="rs"
        )  # data_type='rs' ensures RandomBoxCreator is used
        env.seed(i)
        env.reset()

        # 2. Run a simple greedy agent until the episode is done
        done = False
        step_count = 0
        info = {"counter": 0}
        while not done:
            step_count += 1

            # Agent policy: find the first available valid spot
            action_to_take = None
            if env.mask_o0.any():
                valid_coords = np.argwhere(env.mask_o0)
                action_to_take = (0, valid_coords[0][0], valid_coords[0][1])

            if action_to_take:
                obs, reward, done, info = env.step(action_to_take)
                print(
                    f"  Step {info['counter']}: Placed a {env.box_creator.box_list[-1]} box. Done={done}"
                )
            else:
                print("  Greedy agent found no valid moves. Ending episode.")
                done = True

        # 3. Report final results for the episode
        final_info = info
        final_ratio = final_info.get("ratio", 0)
        final_items = final_info.get("counter", 0)

        print(f"--- Episode {i + 1} Finished ---")
        print(f"  Final items packed: {final_items}")
        print(f"  Final space ratio: {final_ratio:.4f} ({final_ratio*100:.2f}%)")

        if final_ratio > 1.0:
            overpacking_detected = True
            print("  [BUG DETECTED] Overpacking occurred in this episode!")

    # 4. Final Conclusion
    print("\n--- Final Test Conclusion ---")
    if overpacking_detected:
        print(
            "FAIL: The test detected at least one episode where the container was overpacked."
        )
    else:
        print(
            "[PASS] No overpacking was detected. The environment is correctly enforcing its physical limits."
        )


if __name__ == "__main__":
    run_overpacking_test()
