# envs/bpp0/stability.py
import copy
import numpy as np
from matplotlib.path import Path
from numpy.linalg import lstsq
from scipy.spatial import ConvexHull as ScipyConvexHull


class Box:
    def __init__(self, box_id, x, y, z, lx, ly, lz, density=1.0):
        self.id = box_id
        self.x, self.y, self.z = x, y, z
        self.lx, self.ly, self.lz = lx, ly, lz
        # BUG: Mass is calculated from coordinates (x,y) instead of dimensions (lx,ly)
        self.mass = self.x * self.y * self.lz * density
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
    # ... (The rest of this class is correct from our previous fixes) ...
    def __init__(self):
        self.boxes, self.box_id_counter, self.boxes_by_top_z = [], 0, {}

    def get_supporters(self, hypo_box, packed_boxes):
        supporters, p_supporters = [], self.boxes_by_top_z.get(hypo_box.z, [])
        for p_box in p_supporters:
            if (
                hypo_box.x < p_box.x + p_box.lx
                and hypo_box.x + hypo_box.lx > p_box.x
                and hypo_box.y < p_box.y + p_box.ly
                and hypo_box.y + hypo_box.ly > p_box.y
            ):
                supporters.append(p_box)
        return supporters

    def _get_contact_center(self, t_box, b_box):
        x1 = max(t_box.x, b_box.x)
        y1 = max(t_box.y, b_box.y)
        x2 = min(t_box.x + t_box.lx, b_box.x + b_box.lx)
        y2 = min(t_box.y + t_box.ly, b_box.y + b_box.ly)
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    def _calculate_force_distribution(self, box, supporters):
        n_sup = len(supporters)
        if n_sup == 0:
            return []
        if n_sup == 1:
            return [box.mass]
        c_pts = [self._get_contact_center(box, s) for s in supporters]
        if n_sup == 2:
            p0, p1 = c_pts
            v = p1 - p0
            dist_sq = np.dot(v, v)
            if dist_sq < 1e-9:
                return None
            t = np.dot(box.centroid - p0, v) / dist_sq
            if not (0 <= t <= 1):
                return None
            return [box.mass * (1.0 - t), box.mass * t]
        cn = box.centroid
        A = np.zeros((3, n_sup))
        A[0, :] = 1.0
        for i, p in enumerate(c_pts):
            A[1, i] = p[0] - cn[0]
            A[2, i] = p[1] - cn[1]
        forces, _, _, _ = lstsq(np.array([box.mass, 0, 0]), A, rcond=None)
        if np.any(forces < -1e-6):
            return None
        return forces

    def _get_support_polygon(self, box):
        c_pts = []
        for s_box in box.supported_by:
            x1 = max(box.x, s_box.x)
            y1 = max(box.y, s_box.y)
            x2 = min(box.x + box.lx, s_box.x + s_box.lx)
            y2 = min(box.y + box.ly, s_box.y + s_box.ly)
            c_pts.extend([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
        u_pts = {tuple(p) for p in c_pts}
        if len(u_pts) < 3:
            return None
        u_pts_list = np.array(list(u_pts))
        if np.all(u_pts_list[:, 0] == u_pts_list[0, 0]) or np.all(
            u_pts_list[:, 1] == u_pts_list[0, 1]
        ):
            return None
        return ConvexHull(u_pts_list)

    def _is_structure_stable_recursively(
        self, box_to_check, added_mass, added_com_weighted, memo
    ):
        memo_key = (box_to_check.id, round(added_mass, 6))
        if memo_key in memo:
            return memo[memo_key]
        hypo_mass = box_to_check.cumulative_mass + added_mass
        if hypo_mass < 1e-9:
            hypo_com = box_to_check.centroid
        else:
            hypo_com = (
                box_to_check._cumulative_com_weighted + added_com_weighted
            ) / hypo_mass
        if not box_to_check.supported_by:
            memo[memo_key] = True
            return True
        support_polygon = self._get_support_polygon(box_to_check)
        if support_polygon is None or not point_in_polygen(hypo_com, support_polygon):
            memo[memo_key] = False
            return False
        hypo_box = copy.copy(box_to_check)
        hypo_box.mass = hypo_mass
        hypo_box.centroid = hypo_com
        dist_forces = self._calculate_force_distribution(
            hypo_box, box_to_check.supported_by
        )
        if dist_forces is None:
            memo[memo_key] = False
            return False
        for supporter, force in zip(box_to_check.supported_by, dist_forces):
            if not self._is_structure_stable_recursively(
                supporter, force, hypo_com * force, memo
            ):
                memo[memo_key] = False
                return False
        memo[memo_key] = True
        return True

    def is_placement_stable(self, hypo_box, packed_boxes):
        if hypo_box.z == 0:
            return True
        supporters = self.get_supporters(hypo_box, packed_boxes)
        if not supporters:
            return False
        temp_box = copy.copy(hypo_box)
        temp_box.supported_by = supporters
        support_polygon = self._get_support_polygon(temp_box)
        if support_polygon is None or not point_in_polygen(
            hypo_box.centroid, support_polygon
        ):
            return False
        dist_forces = self._calculate_force_distribution(hypo_box, supporters)
        if dist_forces is None:
            return False
        memo = {}
        for s_box, force in zip(supporters, dist_forces):
            if not self._is_structure_stable_recursively(
                s_box, force, hypo_box.centroid * force, memo
            ):
                return False
        return True

    def _propagate_mass_update_adaptively(self, start_box):
        q, visited, head = [start_box], {start_box.id}, 0
        while head < len(q):
            box_on_top = q[head]
            head += 1
            if not box_on_top.supported_by:
                continue
            total_state_box = copy.copy(box_on_top)
            total_state_box.mass = box_on_top.cumulative_mass
            total_state_box.centroid = box_on_top.cumulative_com
            dist_forces = self._calculate_force_distribution(
                total_state_box, box_on_top.supported_by
            )
            if dist_forces is None:
                print(
                    f"WARNING: Unstable distribution found during permanent update for box {box_on_top.id}."
                )
                continue
            for supporter, force in zip(box_on_top.supported_by, dist_forces):
                supporter.cumulative_mass += force
                supporter._cumulative_com_weighted += total_state_box.centroid * force
                if supporter.id not in visited:
                    visited.add(supporter.id)
                    q.append(supporter)

    def add_box_permanently(self, final_box):
        self.box_id_counter += 1
        final_box.id = self.box_id_counter
        supporters = self.get_supporters(final_box, self.boxes)
        final_box.supported_by = supporters
        for s_box in supporters:
            s_box.supports.append(final_box)
        self.boxes.append(final_box)
        top_z = final_box.z + final_box.lz
        if top_z not in self.boxes_by_top_z:
            self.boxes_by_top_z[top_z] = []
        self.boxes_by_top_z[top_z].append(final_box)
        self._propagate_mass_update_adaptively(final_box)
