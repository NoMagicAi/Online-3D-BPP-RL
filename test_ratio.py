# test_ratio_calculation.py
import envs
from envs.bpp0.space import Space

def test_volume_logic():
    print("\n--- Running Definitive Ratio Calculation Test ---")
    
    # 1. Create the exact environment from the sanity check
    bin_size = (4, 4, 4)
    item_size = (2, 2, 2)
    space = Space(*bin_size)
    
    # 2. Manually place the 8 boxes that should perfectly fill the bin
    positions_to_place = [
        # Layer 1
        (0, 0), (2, 0), (0, 2), (2, 2),
        # Layer 2
        (0, 0), (2, 0), (0, 2), (2, 2)
    ]
    
    print(f"Placing {len(positions_to_place)} boxes of size {item_size} into a {bin_size} bin...")
    
    for i, pos in enumerate(positions_to_place):
        # We call drop_box directly, simulating 8 successful steps
        space.drop_box(box_size=item_size, position=pos, flag=False)
        
    # 3. Call get_ratio() once on the final, full state
    final_ratio = space.get_ratio()
    
    print("\n--- Results ---")
    print(f"Number of boxes in final state: {len(space.stacking_tree.boxes)}")
    print(f"Final Calculated Ratio: {final_ratio}")
    
    # 4. Assert the final correctness
    assert abs(final_ratio - 1.0) < 1e-6, "FAIL: The final ratio for a perfect packing is not 1.0!"
    print("\n[PASS] The volume and ratio calculation is correct.")

if __name__ == "__main__":
    test_volume_logic()