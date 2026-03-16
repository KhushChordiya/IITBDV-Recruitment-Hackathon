import numpy as np
from scipy.spatial import Delaunay

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _sort_waypoints_by_path(points: np.ndarray, start_x: float, start_y: float) -> np.ndarray:
    """
    Greedily sort waypoints by nearest-neighbour starting from car position.
    Prevents the path from jumping around randomly.
    """
    if len(points) == 0:
        return points

    remaining = list(range(len(points)))
    ordered   = []
    current   = np.array([start_x, start_y])

    while remaining:
        dists   = [np.hypot(points[i][0] - current[0],
                            points[i][1] - current[1]) for i in remaining]
        nearest = remaining[int(np.argmin(dists))]
        ordered.append(nearest)
        current = points[nearest]
        remaining.remove(nearest)

    return points[ordered]


def _smooth_path(waypoints: list[dict], iterations: int = 3, alpha: float = 0.4) -> list[dict]:
    """
    Chaikin's corner-cutting smoothing.
    Each iteration replaces every segment with two points at 25%/75% along it.
    More iterations = smoother path.
    """
    pts = np.array([[wp["x"], wp["y"]] for wp in waypoints])

    for _ in range(iterations):
        new_pts = [pts[0]]  # keep start
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            new_pts.append(p0 * 0.75 + p1 * 0.25)
            new_pts.append(p0 * 0.25 + p1 * 0.75)
        new_pts.append(pts[-1])  # keep end
        pts = np.array(new_pts)

    return [{"x": float(p[0]), "y": float(p[1])} for p in pts]


def _filter_cross_track_edges(
    tri: Delaunay,
    all_cones: np.ndarray,
    left_set: set,
    right_set: set,
) -> list[tuple[int, int]]:
    """
    From the Delaunay triangulation, keep only edges that cross the track
    (i.e. connect a left cone to a right cone). These midpoints form the racing line.
    """
    valid_edges = []
    seen = set()

    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            edge = (min(a, b), max(a, b))
            if edge in seen:
                continue
            seen.add(edge)

            a_left  = a in left_set
            a_right = a in right_set
            b_left  = b in left_set
            b_right = b in right_set

            # Only keep edges that cross from left to right boundary
            if (a_left and b_right) or (a_right and b_left):
                valid_edges.append(edge)

    return valid_edges


# ─── MAIN PLANNER ─────────────────────────────────────────────────────────────

def plan(
    left_cones:  list[dict],
    right_cones: list[dict],
    state:       dict,
) -> list[dict]:
    """
    Generate a smooth racing line from left/right cone positions.

    Strategy:
      1. Delaunay triangulate all cones together
      2. Keep only edges crossing from left→right boundary
      3. Take midpoints of those edges → raw centerline waypoints
      4. Sort waypoints greedily from car position
      5. Smooth with Chaikin's algorithm
      6. Append a few forward-projected points so the car never runs out of path
    """

    # ── Fallback: not enough cones ────────────────────────────────────────────
    if len(left_cones) < 2 or len(right_cones) < 2:
        # Simple midpoint fallback
        n = min(len(left_cones), len(right_cones))
        return [
            {
                "x": (left_cones[i]["x"] + right_cones[i]["x"]) / 2.0,
                "y": (left_cones[i]["y"] + right_cones[i]["y"]) / 2.0,
            }
            for i in range(n)
        ]

    # ── Build cone arrays with index tracking ─────────────────────────────────
    left_arr  = np.array([[c["x"], c["y"]] for c in left_cones])
    right_arr = np.array([[c["x"], c["y"]] for c in right_cones])

    n_left  = len(left_arr)
    n_right = len(right_arr)

    all_cones = np.vstack([left_arr, right_arr])
    left_set  = set(range(n_left))
    right_set = set(range(n_left, n_left + n_right))

    # ── Delaunay triangulation ────────────────────────────────────────────────
    if len(all_cones) < 3:
        return []

    try:
        tri = Delaunay(all_cones)
    except Exception:
        # Degenerate geometry — fall back to simple midpoints
        n = min(n_left, n_right)
        return [
            {
                "x": (left_arr[i][0] + right_arr[i][0]) / 2.0,
                "y": (left_arr[i][1] + right_arr[i][1]) / 2.0,
            }
            for i in range(n)
        ]

    # ── Extract cross-track edges and their midpoints ─────────────────────────
    valid_edges = _filter_cross_track_edges(tri, all_cones, left_set, right_set)

    if len(valid_edges) < 2:
        # Fallback to simple midpoints
        n = min(n_left, n_right)
        return [
            {
                "x": (left_arr[i][0] + right_arr[i][0]) / 2.0,
                "y": (left_arr[i][1] + right_arr[i][1]) / 2.0,
            }
            for i in range(n)
        ]

    midpoints = np.array([
        (all_cones[a] + all_cones[b]) / 2.0
        for a, b in valid_edges
    ])

    # ── Sort midpoints from car position outward ──────────────────────────────
    sorted_mids = _sort_waypoints_by_path(midpoints, state["x"], state["y"])

    # ── Build waypoint list ───────────────────────────────────────────────────
    raw_path = [{"x": float(p[0]), "y": float(p[1])} for p in sorted_mids]

    # ── Smooth the path ───────────────────────────────────────────────────────
    smooth = _smooth_path(raw_path, iterations=3)

    # ── Extend path: project forward so car never runs out of waypoints ───────
    if len(smooth) >= 2:
        last  = np.array([smooth[-1]["x"],  smooth[-1]["y"]])
        prev  = np.array([smooth[-2]["x"],  smooth[-2]["y"]])
        direction = last - prev
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
            for i in range(1, 6):
                ext = last + direction * (i * 2.0)
                smooth.append({"x": float(ext[0]), "y": float(ext[1])})

    return smooth
