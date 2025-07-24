# test_overpacking_bug.py
import numpy as np
import random

# --- Helper classes to simulate the original, buggy environment ---

class BuggyBox:
    """A simple box class matching the old space.py."""
    def __init__(self, x, y, z, lx, ly, lz):
        self.x, self.y, self.z = x, y, z # Dimensions
        self.lx, self.ly, self.lz = lx, ly, lz # Coordinates

class BuggySpace:
    """Simulates the old space.py with the 'raising the ceiling' bug."""
    def __init__(self, width, length, height):
        self.width, self.length = width, length
        self.height = height # This will be incorrectly modified
        self.plain = np.zeros(shape=(width, length), dtype=np.int32)
        self.boxes = []

    def check_box(self, plain, x, y, lx, ly, z):
        if lx + x > self.width or ly + y > self.length: return -1
        rec = plain[lx : lx + x, ly : ly + y]
        max_h = np.max(rec)
        if max_h + z > self.height: # The overpacking check
            return -1
        return max_h
        
    def drop_box(self, box_size, position):
        lx, ly = position
        x, y, z = box_size
        new_h = self.check_box(self.plain, x, y, lx, ly, z)
        if new_h != -1:
            self.boxes.append(BuggyBox(x, y, z, lx, ly, new_h))
            self.plain[lx:lx+x, ly:ly+y] = new_h + z
            # --- THE BUG ---
            # The container's height limit is incorrectly increased.
            self.height = max(self.height, new_h + z)
            return True
        return False

class BuggyRandomBoxCreator:
    """Simulates a creator that provides an endless stream of boxes."""
    def __init__(self, box_set):
        self.box_set = box_set
    def preview(self, num):
        return [random.choice(self.box_set)]
    def reset(self): pass
    def drop_box(self): pass
    def generate_box_size(self): pass

def test_overpacking_scenario():
    print("\n--- Running Test to Validate Overpacking Bug ---")
    
    # 1. Setup the buggy environment
    initial_height = 4
    bin_size = (4, 4, initial_height)
    item_size = (2, 2, 2)
    
    # We use the buggy classes defined above
    space = BuggySpace(*bin_size)
    box_creator = BuggyRandomBoxCreator([item_size])

    # 2. Run a simple agent that repeatedly tries to place boxes
    max_steps = 20
    for i in range(max_steps):
        # A simple policy: try to find the lowest available spot
        possible_placements = []
        for r in range(space.width - item_size[0] + 1):
            for c in range(space.length - item_size[1] + 1):
                height = np.max(space.plain[r:r+item_size[0], c:c+item_size[1]])
                possible_placements.append({'pos': (r,c), 'height': height})
        
        # Sort by height to find the lowest spot
        possible_placements.sort(key=lambda p: p['height'])
        
        action_pos = possible_placements[0]['pos']
        
        succeeded = space.drop_box(box_creator.preview(1)[0], action_pos)

        print(f"Step {i+1}: Action=Place at {action_pos} -> Success={succeeded}, "
              f"Boxes Packed={len(space.boxes)}, "
              f"Current Bin Height Limit={space.height}")
              
        if not succeeded:
            print("\nPlacement failed. Episode would terminate.")
            break
    
    # 3. Final Assertions to prove the bug
    print("\n--- Test Results ---")
    final_box_count = len(space.boxes)
    physical_limit = (bin_size[0]*bin_size[1]*bin_size[2]) // (item_size[0]*item_size[1]*item_size[2])
    
    print(f"Physical limit of the bin: {physical_limit} boxes.")
    print(f"Agent successfully packed: {final_box_count} boxes.")
    print(f"Original height limit: {initial_height}. Final height limit: {space.height}")
    
    assert final_box_count > physical_limit, "FAIL: Overpacking did not occur."
    print("[PASS] Agent successfully packed more boxes than physically possible.")
    
    assert space.height > initial_height, "FAIL: Bin height limit did not increase."
    print("[PASS] The 'raising the ceiling' bug was successfully reproduced.")

if __name__ == "__main__":
    test_overpacking_scenario()