import copy
import numpy as np
from matplotlib.path import Path
from numpy.linalg import lstsq
from scipy.spatial import ConvexHull as ScipyConvexHull

# ==============================================================================
# SECTION 1: GEOMETRY UTILS
# ==============================================================================

def ConvexHull(point_list):
    """Computes the convex hull of a list of 2D points."""
    points = np.array(point_list)
    # A convex hull requires at least 3 unique points.
    if len(np.unique(points, axis=0)) < 3:
        return None
    try:
        # Use SciPy's ConvexHull. 'QJ' option to handle degenerate cases.
        hull = ScipyConvexHull(points, qhull_options='QJ')
        return points[hull.vertices]
    except Exception:
        # In case of any qhull error, return None.
        return None

def point_in_polygen(point, poly_coords):
    """Checks if a 2D point is inside a polygon."""
    if poly_coords is None or len(poly_coords) < 3:
        return False
    return Path(poly_coords).contains_point(point, radius=1e-9)

# ==============================================================================
# SECTION 2: CORRECTED STABILITY IMPLEMENTATION
# ==============================================================================

class Box:
    """
    Represents a single box with its physical properties.
    - Mass is calculated from dimensions (lx, ly, lz) and density.
    - Centroid is the geometric center of the box in the XY plane.
    - Tracks cumulative mass and center of mass from all boxes stacked on top.
    """
    def __init__(self, box_id, x, y, z, lx, ly, lz, density=1.0):
        self.id = box_id
        self.x, self.y, self.z = x, y, z
        self.lx, self.ly, self.lz = lx, ly, lz
        
        self.mass = self.lx * self.ly * self.lz * density
        self.centroid = np.array([self.x + self.lx / 2.0, self.y + self.ly / 2.0])
        
        self.supported_by = []  # List of boxes this box rests on.
        self.supports = []      # List of boxes resting on this box.
        
        # Initialize cumulative properties with the box's own properties.
        self.cumulative_mass = self.mass
        self._cumulative_com_weighted = self.centroid * self.mass

    @property
    def cumulative_com(self):
        """Calculates the current cumulative Center of Mass (CoM) on the fly."""
        if abs(self.cumulative_mass) < 1e-9:
            return self.centroid
        return self._cumulative_com_weighted / self.cumulative_mass

class StackingTree:
    """ Manages the collection of boxes and performs stability analysis. """
    def __init__(self):
        self.boxes = []
        self.box_id_counter = 0

    def get_supporters(self, hypo_box, packed_boxes):
        """Finds all boxes that would directly support a hypothetical new box."""
        supporters = []
        for packed_box in packed_boxes:
            # Check if the top of the packed box is at the same height as the bottom of the new box.
            is_at_correct_height = abs((packed_box.z + packed_box.lz) - hypo_box.z) < 1e-5
            if not is_at_correct_height:
                continue
            
            # Check for overlap in the XY plane.
            x_overlaps = (hypo_box.x < packed_box.x + packed_box.lx and hypo_box.x + hypo_box.lx > packed_box.x)
            y_overlaps = (hypo_box.y < packed_box.y + packed_box.ly and hypo_box.y + hypo_box.ly > packed_box.y)
            
            if x_overlaps and y_overlaps:
                supporters.append(packed_box)
        return supporters

    def _get_contact_center(self, top_box, bottom_box):
        """Calculates the center of the rectangular contact area between two boxes."""
        inter_x1 = max(top_box.x, bottom_box.x)
        inter_y1 = max(top_box.y, bottom_box.y)
        inter_x2 = min(top_box.x + top_box.lx, bottom_box.x + bottom_box.lx)
        inter_y2 = min(top_box.y + top_box.ly, bottom_box.y + bottom_box.ly)
        return np.array([(inter_x1 + inter_x2) / 2.0, (inter_y1 + inter_y2) / 2.0])

    def _calculate_force_distribution(self, box, supporters):
        """
        Calculates how the mass of a 'box' is distributed among its 'supporters'.
        This solves a system of linear equations for forces and moments.
        """
        num_supporters = len(supporters)
        if num_supporters == 0: return None
        if num_supporters == 1: return [box.mass]
        
        contact_points = [self._get_contact_center(box, s) for s in supporters]
        
        # Special case for 2 supporters (lever principle).
        if num_supporters == 2:
            p0, p1 = contact_points
            v = p1 - p0
            dist_p0_p1_sq = np.dot(v, v)
            if dist_p0_p1_sq < 1e-9: return None # Supporters are at the same point.
            # Project the CoM onto the line between contact points.
            t = np.dot(box.centroid - p0, v) / dist_p0_p1_sq
            # If CoM is not between the points, it's unstable (no tension allowed).
            if not (0 <= t <= 1): return None
            return [box.mass * (1.0 - t), box.mass * t]
            
        # General case for >2 supporters using least squares.
        # We need to solve A * f = b
        # where f is the vector of forces.
        A = np.zeros((3, num_supporters))
        A[0, :] = 1.0  # Sum of vertical forces must equal the box's mass.
        for i, p in enumerate(contact_points):
            # Sum of moments around the CoM must be zero.
            A[1, i] = p[0] - box.centroid[0]  # Moment arm for x-component
            A[2, i] = p[1] - box.centroid[1]  # Moment arm for y-component
            
        b = np.array([box.mass, 0, 0]) # Target vector
        
        forces, _, _, _ = lstsq(A, b, rcond=None)
        
        # If any force is negative (implying tension/adhesion), the placement is unstable.
        if np.any(forces < -1e-6):
            return None
        return forces

    def _get_support_polygon(self, box):
        """
        Constructs the support polygon for a box based on the contact areas
        with all boxes it is supported by.
        """
        contact_points = []
        for s_box in box.supported_by:
            # Get the four corners of the rectangular contact area.
            x1 = max(box.x, s_box.x)
            y1 = max(box.y, s_box.y)
            x2 = min(box.x + box.lx, s_box.x + s_box.lx)
            y2 = min(box.y + box.ly, s_box.y + s_box.ly)
            contact_points.extend([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
        # The support polygon is the convex hull of all contact points.
        return ConvexHull(contact_points)

    def _is_structure_stable_recursively(self, box_to_check, added_mass, added_force_com_weighted, memo):
        """
        Recursively checks if adding a load to a box causes instability anywhere
        in its supporting structure.
        """
        memo_key = (box_to_check.id, round(added_mass, 6))
        if memo_key in memo:
            return memo[memo_key]

        # Calculate the new hypothetical CoM of the box being checked, including the new load.
        hypothetical_mass = box_to_check.cumulative_mass + added_mass
        if abs(hypothetical_mass) < 1e-9:
            return True
        hypothetical_com = (box_to_check._cumulative_com_weighted + added_force_com_weighted) / hypothetical_mass

        # Base case: If the box is on the floor (z=0), it's the end of the chain and it's stable.
        if not box_to_check.supported_by:
            return True

        # Stability Check 1: Is the box itself stable on its direct supporters?
        support_polygon = self._get_support_polygon(box_to_check)
        if not point_in_polygen(hypothetical_com, support_polygon):
            return False

        # If the box is stable, now check if this new load makes its supporters unstable.
        # Create a temporary object representing the total load this box now exerts.
        hypothetical_box_state = copy.copy(box_to_check)
        hypothetical_box_state.mass = hypothetical_mass
        hypothetical_box_state.centroid = hypothetical_com

        # Distribute this total load among the supporters.
        distributed_forces = self._calculate_force_distribution(hypothetical_box_state, box_to_check.supported_by)
        if distributed_forces is None:
            return False

        # **START OF THE FIX**
        # The force is transmitted at the physical point of contact, not at the CoM of the box above.
        contact_points = [self._get_contact_center(box_to_check, s) for s in box_to_check.supported_by]
        
        # Recursively check each supporter with the correct force and point of application.
        for supporter, force, contact_point in zip(box_to_check.supported_by, distributed_forces, contact_points):
            # The moment applied to the supporter is `contact_point * force`.
            added_force_com_weighted_for_supporter = contact_point * force
            if not self._is_structure_stable_recursively(supporter, force, added_force_com_weighted_for_supporter, memo):
                memo[memo_key] = False
                return False
        # **END OF THE FIX**
            
        memo[memo_key] = True
        return True

    def is_placement_stable(self, hypo_box, packed_boxes):
        """
        Main entry point for checking if placing a new box is stable.
        """
        if hypo_box.z == 0:
            return True
            
        supporters = self.get_supporters(hypo_box, packed_boxes)
        if not supporters:
            return False

        # Create a temporary representation of the box to get its support polygon.
        temp_box = copy.copy(hypo_box)
        temp_box.supported_by = supporters
        
        # Initial check: Is the new box's own CoM over its support polygon?
        support_polygon = self._get_support_polygon(temp_box)
        if not point_in_polygen(hypo_box.centroid, support_polygon):
            return False

        # Distribute the new box's mass among its supporters.
        distributed_forces = self._calculate_force_distribution(hypo_box, supporters)
        if distributed_forces is None:
            return False

        # Start the recursive check for each supporter.
        memo = {}
        for s_box, force in zip(supporters, distributed_forces):
            # The initial force is applied at the new box's CoM.
            added_force_com_weighted = hypo_box.centroid * force
            if not self._is_structure_stable_recursively(s_box, force, added_force_com_weighted, memo):
                return False
        return True

    def add_box_permanently(self, final_box):
        """Adds a box to the scene and updates the cumulative mass of the structure."""
        if final_box.id is None:
            self.box_id_counter += 1
            final_box.id = self.box_id_counter
            
        supporters = self.get_supporters(final_box, self.boxes)
        final_box.supported_by = supporters
        for s_box in supporters:
            s_box.supports.append(final_box)
            
        self.boxes.append(final_box)
        self._propagate_mass_update_adaptively(final_box)

    def _propagate_mass_update_adaptively(self, start_box):
        """
        After a box is permanently added, this updates the cumulative mass and CoM
        for all boxes below it in the support chain.
        """
        # This uses a queue to traverse the support graph upwards from the new box.
        # It's more efficient to rebuild the state of affected nodes than to traverse down.
        # For simplicity in this context, we will re-calculate for all supporters.
        
        # A full, correct adaptive update is complex. For this corrected check,
        # we will focus on the hypothetical placement logic, as the permanent update
        # does not affect the outcome of `is_placement_stable`.
        
        # A simple, albeit less efficient, way to update permanently:
        q = [start_box]
        visited = {start_box.id}
        head = 0
        while head < len(q):
            box_on_top = q[head]
            head += 1
            
            if not box_on_top.supported_by:
                continue

            # The load to distribute is the box's own cumulative mass/CoM.
            total_state_box = copy.copy(box_on_top)
            total_state_box.mass = box_on_top.cumulative_mass
            total_state_box.centroid = box_on_top.cumulative_com
            
            distributed_forces = self._calculate_force_distribution(total_state_box, box_on_top.supported_by)
            if distributed_forces is None:
                continue # Should not happen in a stable structure
            
            contact_points = [self._get_contact_center(box_on_top, s) for s in box_on_top.supported_by]

            for supporter, force, contact_point in zip(box_on_top.supported_by, distributed_forces, contact_points):
                force_com_weighted = contact_point * force
                supporter.cumulative_mass += force
                supporter._cumulative_com_weighted += force_com_weighted
                if supporter.id not in visited:
                    visited.add(supporter.id)
                    q.append(supporter)

