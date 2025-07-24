# test_endless_stream_bug.py
import gym
import numpy as np
import random
import envs  # This registers your correct environment, which we will inherit from

# Import the actual classes from your project
from envs.bpp0.bin3D import PackingGame
from envs.bpp0.binCreator import RandomBoxCreator


# 1. Create a temporary, buggy version of the environment for this test.
# This class inherits everything from your actual PackingGame but forces the bug.
class BuggyPackingGame(PackingGame):
    def __init__(self, **kwargs):
        # Call the original __init__ to set up all the correct logic (action space, etc.)
        super().__init__(**kwargs)

        # --- FORCED BUG ---
        # Now, override the box_creator selection with the buggy logic.
        # This simulates the bug that was in your old __init__ method.
        print(
            "\n--- INFO: Forcing use of RandomBoxCreator to demonstrate the 'endless stream' bug ---"
        )
        self.box_creator = RandomBoxCreator(kwargs.get("box_set"))


# 2. Register this temporary, buggy environment with Gym
gym.register(
    id="BuggyBpp-v0",
    entry_point="__main__:BuggyPackingGame",  # The entry point is this script
)


def test_the_final_bug():
    print("\n--- Running Test to Validate the 'Endless Stream' Bug ---")

    # 1. Create an instance of the BUGGY environment
    bin_size = (4, 4, 4)
    item_set = [(2, 2, 2)]
    env = gym.make(
        "BuggyBpp-v0", container_size=bin_size, box_set=item_set, data_type="rs"
    )
    env.reset()

    # 2. Run a simple greedy agent until the episode ends
    done = False
    max_steps = 20
    step_counter = 0
    info = {"counter": 0}  # Initialize info dict

    while not done and step_counter < max_steps:
        item_to_place = env.next_box

        # Agent policy: find the first available valid spot
        action_to_take = None
        # Check orientation 0
        if env.mask_o0.any():
            valid_coords = np.argwhere(env.mask_o0)
            action_to_take = (0, valid_coords[0][0], valid_coords[0][1])

        if action_to_take:
            # Update info with the results of the step
            obs, reward, done, info = env.step(action_to_take)
            step_counter = info["counter"]
            print(f"Step {step_counter}: Placed a box. Done={done}")
        else:
            print("No valid moves found, ending episode.")
            done = True

    # 3. Assert that the bug was successfully reproduced
    print("\n--- Test Results ---")
    final_box_count = info.get("counter", 0)
    physical_limit = (bin_size[0] * bin_size[1] * bin_size[2]) // (
        item_set[0][0] * item_set[0][1] * item_set[0][2]
    )

    print(f"Physical limit of the bin: {physical_limit} boxes.")
    print(f"Agent successfully packed: {final_box_count} boxes.")

    assert (
        final_box_count > physical_limit
    ), "FAIL: The 'endless stream' bug was not reproduced. The environment correctly stopped."
    print(
        "\n[PASS] The bug was successfully reproduced. The agent packed more boxes than physically possible."
    )
    print("This confirms the bug was in the __init__ method's BoxCreator selection.")


if __name__ == "__main__":
    test_the_final_bug()
