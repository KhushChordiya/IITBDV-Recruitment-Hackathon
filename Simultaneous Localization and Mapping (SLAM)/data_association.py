import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment   # ← Hungarian algorithm
import pandas as pd

# ── Load Track from CSV ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_HERE, "small_track.csv"))

BLUE_CONES   = df[df["tag"] == "blue"      ][["x", "y"]].values.astype(float)
YELLOW_CONES = df[df["tag"] == "yellow"    ][["x", "y"]].values.astype(float)
BIG_ORANGE   = df[df["tag"] == "big_orange"][["x", "y"]].values.astype(float)

_cs               = df[df["tag"] == "car_start"].iloc[0]
CAR_START_POS     = np.array([float(_cs["x"]), float(_cs["y"])])
CAR_START_HEADING = float(_cs["direction"])   # radians (0 = east)

MAP_CONES = np.vstack([BLUE_CONES, YELLOW_CONES])


# ── Build Approximate Centerline ──────────────────────────────────────────────
def _build_centerline():
    """
    Pair each blue cone with its nearest yellow cone, take the midpoint,
    then sort CLOCKWISE around the track centroid so pure-pursuit drives CW.
    """
    center = np.mean(MAP_CONES, axis=0)
    D      = distance.cdist(BLUE_CONES, YELLOW_CONES)
    mids   = np.array(
        [(BLUE_CONES[i] + YELLOW_CONES[np.argmin(D[i])]) / 2.0
         for i in range(len(BLUE_CONES))]
    )
    angles = np.arctan2(mids[:, 1] - center[1], mids[:, 0] - center[0])
    return mids[np.argsort(angles)[::-1]]   # descending angle = clockwise


CENTERLINE = _build_centerline()


# ── Simulation Parameters ─────────────────────────────────────────────────────
SENSOR_RANGE = 12.0   # metres – sensor visibility radius
NOISE_STD    = 0.20   # metres – measurement noise std-dev
WHEELBASE    = 3.0    # metres – bicycle model wheelbase
DT           = 0.1    # seconds – time step
SPEED        = 7.0    # m/s
LOOKAHEAD    = 5.5    # pure-pursuit lookahead distance (m)
N_FRAMES     = 130    # ≈ one full lap

# ── Data-Association Configuration ───────────────────────────────────────────
CHI2_GATE_95_2DOF = 5.991   # 95th-percentile chi-squared, 2 degrees of freedom
LARGE_COST        = 1e9     # sentinel cost for gated-out pairs
MEAS_COV          = np.eye(2) * (NOISE_STD ** 2)  # measurement noise covariance


# ── Utility Functions ─────────────────────────────────────────────────────────
def angle_wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def pure_pursuit(pos: np.ndarray, heading: float, path: np.ndarray) -> float:
    """Compute steering angle (rad) to follow *path* via pure-pursuit."""
    dists   = np.linalg.norm(path - pos, axis=1)
    nearest = int(np.argmin(dists))
    n       = len(path)
    target  = path[(nearest + 5) % n]       # fallback lookahead
    for k in range(nearest, nearest + n):
        pt = path[k % n]
        if np.linalg.norm(pt - pos) >= LOOKAHEAD:
            target = pt
            break
    alpha = angle_wrap(
        np.arctan2(target[1] - pos[1], target[0] - pos[0]) - heading
    )
    steer = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), LOOKAHEAD)
    return float(np.clip(steer, -0.6, 0.6))


def local_to_global(local_pts: np.ndarray,
                    pos: np.ndarray, heading: float) -> np.ndarray:
    """Rotate + translate points from the car's local frame to world frame."""
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, -s], [s, c]])       # local → world rotation
    return (R @ local_pts.T).T + pos


def get_measurements(pos: np.ndarray, heading: float) -> np.ndarray:
    """
    Simulate a 2-D lidar: return visible cone positions as noisy
    measurements in the car's LOCAL frame (x = forward, y = left).
    """
    dists   = np.linalg.norm(MAP_CONES - pos, axis=1)
    visible = MAP_CONES[dists < SENSOR_RANGE]
    if len(visible) == 0:
        return np.zeros((0, 2))
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, s], [-s, c]])       # world → local (transpose of above)
    local = (R @ (visible - pos).T).T
    return local + np.random.normal(0, NOISE_STD, local.shape)


def step_kinematic(pos: np.ndarray, heading: float,
                   velocity: float, steering: float):
    """One bicycle-model step; returns (new_pos, new_heading)."""
    new_pos = pos.copy()
    new_pos[0] += velocity * np.cos(heading) * DT
    new_pos[1] += velocity * np.sin(heading) * DT
    new_heading = angle_wrap(
        heading + (velocity / WHEELBASE) * np.tan(steering) * DT
    )
    return new_pos, new_heading


def draw_track(ax, alpha_b: float = 0.4, alpha_y: float = 0.4) -> None:
    ax.scatter(BLUE_CONES[:, 0],   BLUE_CONES[:, 1],
               c="royalblue", marker="^", s=65,  alpha=alpha_b,
               zorder=2, label="Blue cones")
    ax.scatter(YELLOW_CONES[:, 0], YELLOW_CONES[:, 1],
               c="gold",      marker="^", s=65,  alpha=alpha_y,
               zorder=2, label="Yellow cones")
    ax.scatter(BIG_ORANGE[:, 0],   BIG_ORANGE[:, 1],
               c="darkorange", marker="s", s=100, alpha=0.7,
               zorder=2, label="Start gate")


def draw_car(ax, pos: np.ndarray, heading: float) -> None:
    ax.scatter(pos[0], pos[1], c="red", s=160, zorder=7, label="Car")
    ax.arrow(pos[0], pos[1],
             2.2 * np.cos(heading), 2.2 * np.sin(heading),
             head_width=0.8, fc="red", ec="red", zorder=8)


def setup_ax(ax, subtitle: str = "") -> None:
    ax.set_xlim(-28, 28)
    ax.set_ylim(-22, 22)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    if subtitle:
        ax.set_title(subtitle, fontsize=10)


# ── Abstract Base ─────────────────────────────────────────────────────────────
class Bot:
    def __init__(self):
        self.pos     = CAR_START_POS.copy()   # (2,) float64
        self.heading = CAR_START_HEADING      # radians

    def data_association(self, measurements, current_map):
        raise NotImplementedError

    def localization(self, velocity, steering):
        raise NotImplementedError

    def mapping(self, measurements):
        raise NotImplementedError


# ──  Solution ──────────────────────────────────────────────────────────
class Solution(Bot):
    def __init__(self):
        super().__init__()
        self.learned_map  = []                    # list of np.ndarray (2,)
        # Internal state exposed for visualisation
        self._global_meas   = np.zeros((0, 2))
        self._assoc         = np.array([], dtype=int)
        self._unmatched_meas = np.zeros((0, 2))   # NEW: store gated-out measurements

    # ------------------------------------------------------------------
    def data_association(self, measurements, current_map):
        """
        Improved data association using:
          1. Mahalanobis-distance gating  – rejects implausible matches before
             assignment, using chi-squared 95th-percentile threshold (2-DOF).
          2. Hungarian algorithm (linear_sum_assignment) – globally optimal
             1-to-1 matching instead of greedy nearest-neighbour, which can
             chain-cascade bad assignments when cones are close together.
          3. Outlier handling – unmatched measurements (outside the gate) are
             stored in self._unmatched_meas for visualisation and potential
             new-landmark initialisation.

        Parameters
        ----------
        measurements : np.ndarray (M, 2)  –  cone observations in LOCAL frame
        current_map  : np.ndarray (L, 2)  –  known cone positions in WORLD frame

        Returns
        -------
        assoc : np.ndarray (M,) int
            Index into current_map for each measurement.
            Unmatched measurements receive index -1.
        """
        # ── Reset visualisation state ──────────────────────────────────
        self._global_meas    = np.zeros((0, 2))
        self._assoc          = np.array([], dtype=int)
        self._unmatched_meas = np.zeros((0, 2))

        if len(measurements) == 0 or len(current_map) == 0:
            return self._assoc

        # ── 1. Transform measurements from local → global frame ────────
        gm = local_to_global(measurements, self.pos, self.heading)
        self._global_meas = gm                    # store for visualisation

        M = len(gm)
        L = len(current_map)

        # ── 2. Build Mahalanobis cost matrix (M × L) ──────────────────
        # current_map is an (L, 2) array here (MAP_CONES), so no 'cov' key.
        # We use the known sensor noise covariance as the innovation cov.
        cost = np.full((M, L), LARGE_COST)
        S_inv = np.linalg.inv(MEAS_COV)           # constant since no landmark cov

        for i, meas in enumerate(gm):
            diffs = current_map - meas             # (L, 2) broadcast
            # Vectorised Mahalanobis: d²ᵢⱼ = diffᵀ S⁻¹ diff
            maha2 = np.einsum("lj,jk,lk->l", diffs, S_inv, diffs)  # (L,)
            within_gate = maha2 < CHI2_GATE_95_2DOF
            cost[i, within_gate] = maha2[within_gate]

        # ── 3. Hungarian assignment on gated cost matrix ───────────────
        row_ind, col_ind = linear_sum_assignment(cost)

        # Build output array: -1 for unmatched measurements
        assoc_out = np.full(M, -1, dtype=int)
        matched   = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < LARGE_COST:            # valid (non-sentinel) pair
                assoc_out[r] = c
                matched.add(r)

        # ── 4. Store unmatched measurements for visualisation ──────────
        unmatched_idx = [i for i in range(M) if i not in matched]
        if unmatched_idx:
            self._unmatched_meas = gm[unmatched_idx]

        self._assoc = assoc_out
        return self._assoc


# ── Problem 1 – Data Association ──────────────────────────────────────────────
def make_problem1():
    """
    Visualise improved data association.

    Colour coding:
      cyan  dots  = measurements matched via Hungarian + Mahalanobis gating
      green dashes = association line to the assigned map cone
      red   dots  = measurements rejected by the Mahalanobis gate (outliers /
                    new-landmark candidates) – shown WITHOUT an association line
    """
    sol = Solution()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Problem 1 – Data Association  (Hungarian + Mahalanobis Gate)",
                 fontsize=13, fontweight="bold")

    def update(frame):
        ax.clear()
        steer = pure_pursuit(sol.pos, sol.heading, CENTERLINE)
        meas  = get_measurements(sol.pos, sol.heading)
        sol.data_association(meas, MAP_CONES)
        sol.pos, sol.heading = step_kinematic(sol.pos, sol.heading, SPEED, steer)

        draw_track(ax)

        # Draw matched measurements + association lines
        n_matched   = 0
        n_unmatched = 0
        if len(sol._global_meas) > 0:
            for i, (idx, gm) in enumerate(zip(sol._assoc, sol._global_meas)):
                if idx >= 0:
                    mc = MAP_CONES[idx]
                    ax.plot([gm[0], mc[0]], [gm[1], mc[1]],
                            "g--", lw=1.0, alpha=0.65, zorder=3)
                    n_matched += 1

            matched_mask = sol._assoc >= 0
            if matched_mask.any():
                ax.scatter(sol._global_meas[matched_mask, 0],
                           sol._global_meas[matched_mask, 1],
                           c="cyan", s=45, zorder=5,
                           label=f"Matched ({matched_mask.sum()})")

        # Draw unmatched / gated-out measurements in a distinct colour
        if len(sol._unmatched_meas) > 0:
            n_unmatched = len(sol._unmatched_meas)
            ax.scatter(sol._unmatched_meas[:, 0], sol._unmatched_meas[:, 1],
                       c="tomato", s=55, marker="x", zorder=6, linewidths=1.5,
                       label=f"Gated out / new LM ({n_unmatched})")

        draw_car(ax, sol.pos, sol.heading)
        setup_ax(ax, f"Frame {frame+1}/{N_FRAMES}  –  "
                     "green = Hungarian match  |  red × = outside Mahalanobis gate")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=100, repeat=True)
    return fig, ani

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Driverless Car Hackathon – SLAM Visualisation ===")
    print(f"  Blue cones   : {len(BLUE_CONES)}")
    print(f"  Yellow cones : {len(YELLOW_CONES)}")
    print(f"  Big orange   : {len(BIG_ORANGE)}")
    print(f"  Car start    : {CAR_START_POS}  "
          f"heading={np.degrees(CAR_START_HEADING):.1f}°")
    print(f"  Centerline   : {len(CENTERLINE)} waypoints (clockwise)")
    print("\nOpening 1 animation window …")

    # Keep references to prevent garbage collection of FuncAnimation objects.
    fig1, ani1 = make_problem1()

    plt.show()
