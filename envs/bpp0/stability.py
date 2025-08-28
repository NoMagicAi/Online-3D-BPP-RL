import copy
import numpy as np
from matplotlib.path import Path
from numpy.linalg import lstsq
from scipy.spatial import ConvexHull as ScipyConvexHull

# Make sure these utility functions and the Box class are also present and correct.
def ConvexHull(point_list):
    points = np.array(point_list)
    if len(np.unique(points, axis=0)) < 3: return points
    try:
        hull = ScipyConvexHull(points, qhull_options='QJ')
        return points[hull.vertices]
    except Exception: return None

def point_in_polygen(point, poly_coords):
    if poly_coords is None: return False
    if len(poly_coords) < 3:
        if len(poly_coords) == 1: return np.allclose(point, poly_coords[0])
        if len(poly_coords) == 2:
            p0, p1 = poly_coords
            dist_p0_p1 = np.linalg.norm(p1 - p0)
            dist_point_p0 = np.linalg.norm(point - p0)
            dist_point_p1 = np.linalg.norm(point - p1)
            return np.isclose(dist_point_p0 + dist_point_p1, dist_p0_p1)
        return False
    return Path(poly_coords).contains_point(point, radius=1e-9)

class Box:
    def __init__(self, box_id, x, y, z, lx, ly, lz, density=1.0):
        self.id = box_id
        self.x, self.y, self.z = x, y, z
        self.lx, self.ly, self.lz = lx, ly, lz
        self.mass = self.lx * self.ly * self.lz * density
        self.centroid = np.array([self.x + self.lx / 2.0, self.y + self.ly / 2.0])
        self.supported_by = []
        self.supports = []

class StackingTree:
    """Manages the box state and performs physically correct, stateless stability analysis."""
    def __init__(self):
        self.boxes = []
        self.box_id_counter = 0

    def add_box_permanently(self, final_box):
        """Adds a box to the scene, establishing its support relationships."""
        if final_box.id is None:
            self.box_id_counter += 1
            final_box.id = self.box_id_counter

        supporters = self.get_supporters(final_box, self.boxes)
        final_box.supported_by = supporters
        for s_box in supporters:
            s_box.supports.append(final_box)

        self.boxes.append(final_box)

    def get_supporters(self, hypo_box, packed_boxes):
        """Finds all boxes that would directly support a hypothetical new box."""
        supporters = []
        for packed_box in packed_boxes:
            is_at_correct_height = abs((packed_box.z + packed_box.lz) - hypo_box.z) < 1e-5
            if not is_at_correct_height: continue
            x_overlaps = (hypo_box.x < packed_box.x + packed_box.lx and hypo_box.x + hypo_box.lx > packed_box.x)
            y_overlaps = (hypo_box.y < packed_box.y + packed_box.ly and hypo_box.y + hypo_box.ly > packed_box.y)
            if x_overlaps and y_overlaps: supporters.append(packed_box)
        return supporters

    def _get_contact_center(self, top_box, bottom_box):
        """Calculates the center of the rectangular contact area between two boxes."""
        inter_x1 = max(top_box.x, bottom_box.x)
        inter_y1 = max(top_box.y, bottom_box.y)
        inter_x2 = min(top_box.x + top_box.lx, bottom_box.x + bottom_box.lx)
        inter_y2 = min(top_box.y + top_box.ly, bottom_box.y + bottom_box.ly)
        return np.array([(inter_x1 + inter_x2) / 2.0, (inter_y1 + inter_y2) / 2.0])

    def _calculate_force_distribution(self, total_mass, total_com, supporters, contact_points):
        """Calculates force distribution. Returns None if tension is required."""
        num_supporters = len(supporters)
        if num_supporters == 0: return None
        if num_supporters == 1: return np.array([total_mass])
        
        A = np.zeros((3, num_supporters)); A[0, :] = 1.0
        for i, p in enumerate(contact_points):
            A[1, i] = p[0] - total_com[0]; A[2, i] = p[1] - total_com[1]
        b = np.array([total_mass, 0, 0])
        forces, _, _, _ = lstsq(A, b, rcond=None)
        
        if np.any(forces < -1e-6): return None
        return forces

    def _get_support_polygon(self, box, supporters):
        """Constructs the support polygon from the convex hull of contact areas."""
        if not supporters: return None
        contact_points = []
        for s_box in supporters:
            x1 = max(box.x, s_box.x); y1 = max(box.y, s_box.y)
            x2 = min(box.x + box.lx, s_box.x + s_box.lx); y2 = min(box.y + box.ly, s_box.y + s_box.ly)
            contact_points.extend([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
        return ConvexHull(contact_points)

    def _get_stack_properties(self, box, memo_properties):
        """
        Recursively calculates the total mass and combined CoM for a stack
        rooted at 'box'.
        """
        if box.id in memo_properties:
            return memo_properties[box.id]

        total_mass = box.mass
        weighted_com = box.mass * box.centroid

        for box_on_top in box.supports:
            mass_above, com_above = self._get_stack_properties(box_on_top, memo_properties)
            total_mass += mass_above
            weighted_com += mass_above * com_above

        if abs(total_mass) < 1e-9:
            combined_com = box.centroid
        else:
            combined_com = weighted_com / total_mass

        memo_properties[box.id] = (total_mass, combined_com)
        return total_mass, combined_com

    def is_placement_stable(self, hypo_box, packed_boxes, debug=False):
        """
        Main entry point for checking stability.
        Set debug=True to get a detailed report.
        """
        if hypo_box.z == 0:
            if debug: print("DEBUG: Box on floor. Stable.")
            return True

        temp_boxes = [copy.copy(b) for b in packed_boxes]
        temp_box_map = {b.id: b for b in temp_boxes}

        for b in temp_boxes:
            b.supported_by = [temp_box_map[s.id] for s in b.supported_by if s.id in temp_box_map]
            b.supports = [temp_box_map[s.id] for s in b.supports if s.id in temp_box_map]

        hypo_box_copy = copy.copy(hypo_box)
        supporters = self.get_supporters(hypo_box_copy, temp_boxes)
        if not supporters:
            if debug: print("DEBUG: No supporters found. Unstable.")
            return False

        hypo_box_copy.supported_by = supporters
        for s in supporters:
            s.supports.append(hypo_box_copy)

        all_temp_boxes = temp_boxes + [hypo_box_copy]
        sorted_boxes = sorted(all_temp_boxes, key=lambda b: b.z, reverse=True)

        if debug:
            np.set_printoptions(precision=3, suppress=True)
            print("\n=============== STARTING STABILITY DIAGNOSIS ===============")
            print(f"Hypothetical Box: ID 'None' at (x:{hypo_box.x}, y:{hypo_box.y}, z:{hypo_box.z})")
            print(f"Checking a total of {len(sorted_boxes)} boxes, from top to bottom.")
            print("==============================================================\n")

        memo_properties = {}
        for box_to_check in sorted_boxes:
            if not box_to_check.supported_by:
                if debug:
                    print(f"--- Checking Box ID: {box_to_check.id} ---")
                    print("STATUS: On floor. Skipping.")
                    print("--------------------------------\n")
                continue

            total_mass, total_com = self._get_stack_properties(box_to_check, memo_properties)
            
            if debug:
                print(f"--- Checking Box ID: {box_to_check.id} ---")
                print(f"Supporters: {[s.id for s in box_to_check.supported_by]}")
                print(f"Stack Mass (from this box up): {total_mass:.3f}")
                print(f"Stack CoM (x,y): {total_com}")

            if abs(total_mass) < 1e-9:
                if debug: print("STATUS: Massless stack. Skipping.\n")
                continue

            support_polygon = self._get_support_polygon(box_to_check, box_to_check.supported_by)
            is_in_polygon = point_in_polygen(total_com, support_polygon)
            
            if debug:
                print(f"Support Polygon Vertices:\n{support_polygon}")
                print(f"CHECK 1: Is CoM inside support polygon? -> {is_in_polygon}")
            
            if not is_in_polygon:
                if debug:
                    print("\n** RESULT: UNSTABLE (CoM outside support polygon) **")
                    print("==============================================================\n")
                return False

            contact_points = [self._get_contact_center(box_to_check, s) for s in box_to_check.supported_by]
            forces = self._calculate_force_distribution(total_mass, total_com, box_to_check.supported_by, contact_points)

            if debug:
                print(f"Forces on supporters: {forces}")
                print(f"CHECK 2: Are any forces negative (tension)? -> {forces is None}")
            
            if forces is None:
                if debug:
                    print("\n** RESULT: UNSTABLE (Tension required) **")
                    print("==============================================================\n")
                return False

            if debug:
                print("STATUS: This level is stable.")
                print("--------------------------------\n")

        if debug:
            print("** FINAL RESULT: STABLE (All checks passed) **")
            print("==============================================================\n")
        return True