# test_stability_bugs.py
import numpy as np
import envs  # To ensure the project is in the path

# Import the actual classes from your project to be tested
from envs.bpp0.stability import Box, StackingTree


def test_mass_calculation_bugs():
    """
    This test directly validates the mass calculation within the Box class
    from envs/bpp0/stability.py.
    """
    print("\n--- Running Tests on Stability Module ---")

    # --- Test 1: The Zero-Mass Bug ---
    print("\n[Test 1] Checking for the zero-mass bug...")

    # Create a box with non-zero dimensions at coordinate (0, 0)
    box_at_origin = Box(box_id=1, x=0, y=0, z=0, lx=2, ly=2, lz=2)

    # Correct physical mass should be dimensions multiplied: 2 * 2 * 2 = 8
    correct_mass = 8.0

    print(f"Box at (0,0) with dims (2,2,2) was created.")
    print(f"  - Expected Mass (from dims lx*ly*lz): {correct_mass}")
    print(f"  - Actual Mass in code: {box_at_origin.mass}")

    try:
        assert box_at_origin.mass > 0, "Mass should be positive."
        assert (
            abs(box_at_origin.mass - correct_mass) < 1e-6
        ), "Mass calculation is incorrect."
        print("[PASS] Box at origin has correct, positive mass.")
    except AssertionError as e:
        print(f"[FAIL] {e}")
        print(
            "       This proves the bug: mass was likely calculated using coordinates (x,y) instead of dimensions (lx,ly)."
        )

    # --- Test 2: The Coordinate-Dependent Mass Bug ---
    print("\n[Test 2] Checking for coordinate-dependent mass bug...")

    # Create two boxes with identical dimensions but different coordinates
    box_A = Box(box_id=2, x=1, y=1, z=0, lx=2, ly=2, lz=2)
    box_B = Box(box_id=3, x=5, y=5, z=0, lx=2, ly=2, lz=2)

    print("Created two identical boxes at different coordinates (1,1) and (5,5).")
    print(f"  - Mass of Box A (at (1,1)): {box_A.mass}")
    print(f"  - Mass of Box B (at (5,5)): {box_B.mass}")

    try:
        assert (
            abs(box_A.mass - box_B.mass) < 1e-6
        ), "Boxes of the same size should have the same mass."
        print("[PASS] Boxes of the same size have equal mass regardless of position.")
    except AssertionError as e:
        print(f"[FAIL] {e}")
        print(
            "       This proves the bug: a box's mass should not depend on its coordinates."
        )


if __name__ == "__main__":
    test_mass_calculation_bugs()
