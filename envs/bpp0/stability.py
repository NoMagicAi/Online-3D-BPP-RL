import copy
import json
import os
import random
import time

import numpy as np
from matplotlib.path import Path
from numpy.linalg import lstsq
from scipy.ndimage import maximum_filter

# Import scientific libraries for high-efficiency geometry calculations
from scipy.spatial import ConvexHull as ScipyConvexHull


# ==============================================================================
# SECTION 1: GEOMETRY UTILS (Unchanged)
# ==============================================================================


class Line2D(object):
    def __init__(self, point1, point2):
        self.p1 = point1
        self.p2 = point2
        if self.p2[0] != self.p1[0]:
            self.slope = (self.p2[1] - self.p1[1]) / (self.p2[0] - self.p1[0])
        else:
            self.slope = (self.p2[1] - self.p1[1]) * np.inf

    def orientation(self, line2):
        slope1 = self.slope
        slope2 = line2.slope
        if abs(slope1) == np.inf and abs(slope2) == np.inf:
            return 0
        diff = slope2 - slope1
        if diff > 0:
            return -1
        elif diff == 0:
            return 0
        else:
            return 1


def sortPoints(point_list):
    return sorted(point_list, key=lambda x: (x[0], x[1]))


def ConvexHull(point_list):
    """
    Computes the convex hull of a set of 2D points using Scipy's optimized algorithm.
    """
    points = np.array(point_list)
    # The calling function (_get_support_polygon) ensures there are at least 3 unique points.
    # Scipy's ConvexHull is highly optimized (written in C).
    hull = ScipyConvexHull(points)
    # Return the vertices of the hull in order.
    return points[hull.vertices]


def point_in_polygen(point, poly_coords):
    """
    Checks if a point is inside a polygon using Matplotlib's highly efficient Path object.
    A small radius is used to include points that lie on the boundary of the polygon.
    """
    if len(poly_coords) < 3:
        return False
    # The Path object provides a highly optimized C implementation for point-in-polygon tests.
    return Path(poly_coords).contains_point(point, radius=1e-9)


# ==============================================================================
# SECTION 2: ADAPTIVE STACKING TREE IMPLEMENTATION (Unchanged)
# ==============================================================================


class Box:
    def __init__(self, box_id, x, y, z, lx, ly, lz, density=1.0):
        self.id = box_id
        # x, y, z are COORDS of bottom-left corner
        # lx, ly, lz are DIMS of the box (width, length, height)
        self.x, self.y, self.z = x, y, z
        self.lx, self.ly, self.lz = lx, ly, lz

        # --- THE FIX ---
        # Mass must be calculated from dimensions (lx, ly, lz), not coordinates (x, y)
        self.mass = self.lx * self.ly * self.lz * density

        self.centroid = np.array([self.x + self.lx / 2.0, self.y + self.ly / 2.0])
        self.supported_by = []
        self.supports = []
        self.cumulative_mass = self.mass
        self._cumulative_com_weighted = self.centroid * self.mass

    @property
    def cumulative_com(self):
        if self.cumulative_mass < 1e-9:
            return self.centroid
        return self._cumulative_com_weighted / self.cumulative_mass


class StackingTree:
    """
    Manages the state of all packed boxes and their stability relationships
    using the adaptive stacking tree mechanism.
    """

    def __init__(self):
        self.boxes = []
        self.box_id_counter = 0
        self.boxes_by_top_z = {}

    def get_supporters(self, hypo_box, packed_boxes):
        """Finds which of the already packed boxes would support a hypothetical new box."""
        supporters = []
        # This optimized lookup instantly finds boxes at the correct height,
        # avoiding a linear scan through all packed_boxes.
        potential_supporters = self.boxes_by_top_z.get(hypo_box.z, [])

        for packed_box in potential_supporters:
            # The z-level check is already satisfied by the dictionary lookup.
            # We only need to check for overlap in the XY plane.
            if (
                hypo_box.lx < packed_box.lx + packed_box.x
                and hypo_box.lx + hypo_box.x > packed_box.lx
                and hypo_box.ly < packed_box.ly + packed_box.y
                and hypo_box.ly + hypo_box.y > packed_box.ly
            ):
                supporters.append(packed_box)
        return supporters

    def _get_contact_center(self, top_box, bottom_box):
        """Calculates the center of the rectangular contact patch between two boxes."""
        x1 = max(top_box.lx, bottom_box.lx)
        y1 = max(top_box.ly, bottom_box.ly)
        x2 = min(top_box.lx + top_box.x, bottom_box.lx + bottom_box.x)
        y2 = min(top_box.ly + top_box.y, bottom_box.ly + bottom_box.y)
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    def _calculate_force_distribution(self, box, supporters):
        """
        Calculates the distributed forces from a box onto its supporters.
        Returns a list of forces. Returns None if the distribution is unstable.
        """
        num_supporters = len(supporters)
        if num_supporters == 0:
            return []

        box_weight = box.mass  # Using mass as a proxy for force (g is constant)

        if num_supporters == 1:
            return [box_weight]

        contact_points = [self._get_contact_center(box, s) for s in supporters]

        if num_supporters == 2:
            p0, p1 = contact_points
            cn_proj = box.centroid

            v = p1 - p0
            dist_p0_p1_sq = np.dot(v, v)

            # If supporters are at the same point, it's unstable.
            if dist_p0_p1_sq < 1e-9:
                return None

            # Project the centroid onto the line segment connecting the two contact points
            u = cn_proj - p0
            t = np.dot(u, v) / dist_p0_p1_sq

            # If projection is outside the support segment, it's unstable (tension would be required).
            if not (0 <= t <= 1):
                return None

            # Leverage principle: force is inversely proportional to the distance from the projected CoM.
            f1 = box_weight * t
            f0 = box_weight * (1.0 - t)

            return [f0, f1]

        # Case for > 2 supporters: Use least-squares
        cn = box.centroid
        A = np.zeros((3, num_supporters))
        A[0, :] = 1.0  # Force balance: sum(Fi) = W
        for i, p in enumerate(contact_points):
            A[1, i] = p[0] - cn[0]  # Torque balance (y-axis): sum(Fi * ri_x) = 0
            A[2, i] = p[1] - cn[1]  # Torque balance (x-axis): sum(Fi * ri_y) = 0

        b = np.array([box_weight, 0, 0])

        forces, _, _, _ = lstsq(A, b, rcond=None)

        # If any force is negative, it implies tension is needed, so it's unstable.
        if np.any(forces < -1e-6):  # Use a small tolerance for floating point errors
            return None

        return forces

    def _get_support_polygon(self, box):
        """Calculates the supporting convex hull for a given box from its supporters."""
        contact_points = []
        for s_box in box.supported_by:
            x1 = max(box.x, s_box.x)
            y1 = max(box.y, s_box.y)
            x2 = min(box.x + box.lx, s_box.x + s_box.lx)
            y2 = min(box.y + box.ly, s_box.y + s_box.ly)
            contact_points.extend([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

        # --- FIX: Add a check for co-linearity ---
        unique_points = {tuple(p) for p in contact_points}

        # 1. Check if there are enough unique points to form a polygon.
        if len(unique_points) < 3:
            return None

        # 2. Check if the points are co-linear (all on the same line).
        unique_points_list = np.array(list(unique_points))
        # Check if all x-coordinates are the same OR all y-coordinates are the same.
        all_same_x = np.all(unique_points_list[:, 0] == unique_points_list[0, 0])
        all_same_y = np.all(unique_points_list[:, 1] == unique_points_list[0, 1])

        if all_same_x or all_same_y:
            return None  # Points are co-linear and cannot form a 2D hull.

        return ConvexHull(unique_points_list)

    def _is_structure_stable_recursively(
        self, box_to_check, added_mass, added_com_weighted, memo
    ):
        memo_key = (box_to_check.id, round(added_mass, 6))
        if memo_key in memo:
            return memo[memo_key]

        # Calculate hypothetical new state for the box being checked
        hypothetical_mass = box_to_check.cumulative_mass + added_mass

        # --- FIX: Add a safety check to prevent division by zero ---
        if hypothetical_mass < 1e-9:
            hypothetical_com = box_to_check.centroid  # Fallback to a safe default
        else:
            hypothetical_com_weighted = (
                box_to_check._cumulative_com_weighted + added_com_weighted
            )
            hypothetical_com = hypothetical_com_weighted / hypothetical_mass

        if not box_to_check.supported_by:
            memo[memo_key] = True
            return True

        support_polygon = self._get_support_polygon(box_to_check)
        if support_polygon is None or not point_in_polygen(
            hypothetical_com, support_polygon
        ):
            memo[memo_key] = False
            return False

        hypothetical_box_state = copy.copy(box_to_check)
        hypothetical_box_state.mass = hypothetical_mass
        hypothetical_box_state.centroid = hypothetical_com

        distributed_forces = self._calculate_force_distribution(
            hypothetical_box_state, box_to_check.supported_by
        )
        if distributed_forces is None:
            memo[memo_key] = False
            return False

        for supporter, force in zip(box_to_check.supported_by, distributed_forces):
            force_com_weighted = hypothetical_com * force
            if not self._is_structure_stable_recursively(
                supporter, force, force_com_weighted, memo
            ):
                memo[memo_key] = False
                return False

        memo[memo_key] = True
        return True

    def is_placement_stable(self, hypo_box, packed_boxes):
        """
        Public method to check if placing a new box is fully stable, using a memoized
        recursive check to implement the efficiency of the adaptive stacking tree.
        """
        if hypo_box.z == 0:
            return True

        supporters = self.get_supporters(hypo_box, packed_boxes)
        if not supporters:
            return False

        # 1. Initial stability of the new box itself
        temp_box = copy.copy(hypo_box)
        temp_box.supported_by = supporters  # Temporarily link for polygon calculation
        support_polygon = self._get_support_polygon(temp_box)

        if support_polygon is None or not point_in_polygen(
            hypo_box.centroid, support_polygon
        ):
            return False

        # 2. Check force distribution
        distributed_forces = self._calculate_force_distribution(hypo_box, supporters)
        if distributed_forces is None:
            return False

        # 3. Recursively check the stability of the entire affected sub-structure (the adaptive tree)
        # The memoization cache prevents re-calculating stability for the same box under the same load.
        memo = {}
        for s_box, force in zip(supporters, distributed_forces):
            added_com_weighted = hypo_box.centroid * force
            if not self._is_structure_stable_recursively(
                s_box, force, added_com_weighted, memo
            ):
                return False

        return True

    def _propagate_mass_update_adaptively(self, start_box):
        """
        Updates the cumulative mass and CoM for all boxes supporting the start_box (the adaptive tree).
        Uses a queue-based (BFS) approach to ensure each box in the affected structure is updated exactly once.
        """
        q = [start_box]
        visited = {start_box.id}

        head = 0
        while head < len(q):
            box_on_top = q[head]
            head += 1

            if not box_on_top.supported_by:
                continue

            # Use the box's current *cumulative* state to calculate the load it exerts on its supporters
            total_state_box = copy.copy(box_on_top)
            total_state_box.mass = box_on_top.cumulative_mass
            total_state_box.centroid = box_on_top.cumulative_com

            distributed_forces = self._calculate_force_distribution(
                total_state_box, box_on_top.supported_by
            )

            if distributed_forces is None:
                print(
                    f"WARNING: Unstable distribution found during permanent update for box {box_on_top.id}."
                )
                continue

            for supporter, force in zip(box_on_top.supported_by, distributed_forces):
                force_com_weighted = total_state_box.centroid * force
                supporter.cumulative_mass += force
                supporter._cumulative_com_weighted += force_com_weighted

                if supporter.id not in visited:
                    visited.add(supporter.id)
                    q.append(supporter)

    def add_box_permanently(self, final_box):
        """Adds a box to the bin state and adaptively updates the mass of its supporters."""
        self.box_id_counter += 1
        final_box.id = self.box_id_counter

        supporters = self.get_supporters(final_box, self.boxes)
        final_box.supported_by = supporters
        for s_box in supporters:
            s_box.supports.append(final_box)

        # The new box is now part of the permanent structure
        self.boxes.append(final_box)

        top_z = final_box.z + final_box.lz
        if top_z not in self.boxes_by_top_z:
            self.boxes_by_top_z[top_z] = []
        self.boxes_by_top_z[top_z].append(final_box)

        # Propagate the mass update efficiently through the adaptive tree (all supporters, direct and indirect)
        self._propagate_mass_update_adaptively(final_box)
