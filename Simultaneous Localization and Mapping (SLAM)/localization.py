import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance
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

# ── EKF Noise Parameters ──────────────────────────────────────────────────────
# Process noise covariance Q  (how much we distrust the motion model)
_Q = np.diag([0.05**2, 0.05**2, np.deg2rad(1.0)**2])

# Observation noise covariance R  (range std=0.3 m, bearing std=3°)
_R_OBS = np.diag([0.3**2, np.deg2rad(3.0)**2])


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
        self.learned_map = []   # list of np.ndarray (2,)

        # ── EKF state ────────────────────────────────────────────────
        # mu: [x, y, heading]  (initialised from CAR_START_*)
        self._mu = np.array([
            float(CAR_START_POS[0]),
            float(CAR_START_POS[1]),
            float(CAR_START_HEADING),
        ])
        # P: 3×3 pose covariance (small initial uncertainty)
        self._P  = np.diag([0.01, 0.01, np.deg2rad(1.0)**2])

        # Exposed for visualisation: dead-reckoning shadow trail
        self._dr_pos     = CAR_START_POS.copy()
        self._dr_heading = CAR_START_HEADING

    # ------------------------------------------------------------------
    def localization(self, velocity, steering,
                     landmark_obs=None, known_landmarks=None):
        """
        EKF localization fusing bicycle-model dead-reckoning with
        range-bearing observations to known landmarks.

        Improvements over the baseline dead-reckoning
        ----------------------------------------------
        Dead-reckoning is open-loop: heading error integrates without
        bound.  Even 0.5°/step bias drifts several metres per lap.
        The EKF corrects continuously whenever a landmark observation
        is available, keeping errors bounded while remaining real-time.

        Predict step  (bicycle kinematic model – same equations as baseline)
        ─────────────
          x'  = x + v·cos(θ)·dt
          y'  = y + v·sin(θ)·dt
          θ'  = θ + (v/L)·tan(δ)·dt
          P'  = F·P·Fᵀ + Q          (Jacobian F = ∂f/∂[x,y,θ])

        Update step  (range-bearing observation model)
        ───────────
          range   = √((lm_x−x)² + (lm_y−y)²)
          bearing = atan2(lm_y−y, lm_x−x) − θ
          K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
          μ ← μ + K·(z − ẑ)
          P ← (I − K·H)·P

        Parameters
        ----------
        velocity  : float – forward speed (m/s)
        steering  : float – steering angle (rad)
        landmark_obs    : list of (range, bearing, landmark_idx) – optional
        known_landmarks : (L, 2) array of landmark positions in world frame
        """
        x, y, th = self._mu

        # ── Also propagate a pure dead-reckoning shadow (for comparison) ──
        self._dr_pos[0]  += velocity * np.cos(self._dr_heading) * DT
        self._dr_pos[1]  += velocity * np.sin(self._dr_heading) * DT
        self._dr_heading  = angle_wrap(
            self._dr_heading + (velocity / WHEELBASE) * np.tan(steering) * DT
        )

        # ── EKF Predict ───────────────────────────────────────────────
        x_new  = x  + velocity * np.cos(th) * DT
        y_new  = y  + velocity * np.sin(th) * DT
        th_new = angle_wrap(
            th + (velocity / WHEELBASE) * np.tan(steering) * DT
        )

        # Jacobian of motion model w.r.t. state [x, y, θ]
        F = np.eye(3)
        F[0, 2] = -velocity * np.sin(th) * DT
        F[1, 2] =  velocity * np.cos(th) * DT

        self._mu = np.array([x_new, y_new, th_new])
        self._P  = F @ self._P @ F.T + _Q

        # ── EKF Update (one update per associated landmark) ───────────
        if landmark_obs and known_landmarks is not None:
            for (r_meas, b_meas, lm_idx) in landmark_obs:
                lm = known_landmarks[lm_idx]
                self._ekf_update(r_meas, b_meas, lm)

        # ── Sync pos / heading so the rest of the scaffold still works ─
        self.pos     = self._mu[:2].copy()
        self.heading = float(self._mu[2])

    # ------------------------------------------------------------------
    def _ekf_update(self, r_meas: float, b_meas: float,
                    lm_pos: np.ndarray) -> None:
        """Single EKF update for one range-bearing observation."""
        x, y, th = self._mu
        lm_x, lm_y = float(lm_pos[0]), float(lm_pos[1])

        dx = lm_x - x
        dy = lm_y - y
        q  = dx**2 + dy**2
        r  = np.sqrt(q)
        if r < 1e-6:
            return

        # Predicted observation
        z_pred = np.array([r, np.arctan2(dy, dx) - th])

        # Observation Jacobian H = ∂h/∂[x, y, θ]
        H = np.array([
            [-dx / r,  -dy / r,  0.0],
            [ dy / q,  -dx / q, -1.0],
        ])

        # Innovation (wrap bearing to [-π, π])
        innov    = np.array([r_meas, b_meas]) - z_pred
        innov[1] = angle_wrap(innov[1])

        # Kalman gain and state/covariance update
        S        = H @ self._P @ H.T + _R_OBS
        K        = self._P @ H.T @ np.linalg.inv(S)
        self._mu = self._mu + K @ innov
        self._mu[2] = angle_wrap(self._mu[2])
        self._P  = (np.eye(3) - K @ H) @ self._P

    # ------------------------------------------------------------------
    def _get_landmark_obs(self, measurements: np.ndarray) -> list:
        """
        Convert local-frame cone measurements into (range, bearing, idx)
        tuples by matching each measurement to the nearest MAP_CONE entry.
        Used internally to feed the EKF update without a full data-
        association pipeline.
        """
        if len(measurements) == 0:
            return []
        obs = []
        for local_pt in measurements:
            r = float(np.linalg.norm(local_pt))
            b = float(np.arctan2(local_pt[1], local_pt[0]))
            # Find nearest cone in world frame
            gm  = local_to_global(local_pt[None], self.pos, self.heading)[0]
            idx = int(np.argmin(np.linalg.norm(MAP_CONES - gm, axis=1)))
            obs.append((r, b, idx))
        return obs


# ── Problem 2 – Localization ───────────────────────────────────────────────────
def make_problem2():
    """
    Side-by-side comparison of dead-reckoning vs EKF localization.

    Magenta trail  = pure dead-reckoning shadow (open-loop, drifts over time)
    Green  trail   = EKF-corrected pose (fuses motion model + cone observations)

    The two paths start identically; divergence shows how much the EKF
    corrections reduce accumulated heading and position error.
    """
    sol    = Solution()
    # EKF path
    ekf_x  = [float(sol.pos[0])]
    ekf_y  = [float(sol.pos[1])]
    # Dead-reckoning shadow path
    dr_x   = [float(sol._dr_pos[0])]
    dr_y   = [float(sol._dr_pos[1])]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(
        "Problem 2 – Localization  (EKF: dead-reckoning + landmark observations)",
        fontsize=13, fontweight="bold"
    )

    def update(frame):
        ax.clear()
        steer = pure_pursuit(sol.pos, sol.heading, CENTERLINE)
        meas  = get_measurements(sol.pos, sol.heading)

        # Build range-bearing observations for the EKF update
        obs = sol._get_landmark_obs(meas)
        sol.localization(SPEED, steer,
                         landmark_obs=obs, known_landmarks=MAP_CONES)

        ekf_x.append(float(sol.pos[0]))
        ekf_y.append(float(sol.pos[1]))
        dr_x.append(float(sol._dr_pos[0]))
        dr_y.append(float(sol._dr_pos[1]))

        draw_track(ax)

        # Dead-reckoning shadow
        ax.plot(dr_x, dr_y, color="magenta", lw=1.5, alpha=0.6,
                zorder=4, label="Dead-reckoning (baseline)", linestyle="--")

        # EKF trajectory
        ax.plot(ekf_x, ekf_y, color="limegreen", lw=2.2,
                alpha=0.9, zorder=5, label="EKF trajectory (improved)")

        # Uncertainty ellipse (1-σ from covariance)
        _draw_cov_ellipse(ax, sol._mu[:2], sol._P[:2, :2])

        draw_car(ax, sol.pos, sol.heading)
        setup_ax(ax,
            f"Frame {frame+1}/{N_FRAMES}  –  "
            f"EKF pos=({sol.pos[0]:.1f}, {sol.pos[1]:.1f})  "
            f"ψ={np.degrees(sol.heading):.1f}°  "
            f"|obs|={len(obs)}")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=100, repeat=True)
    return fig, ani


def _draw_cov_ellipse(ax, mean: np.ndarray, cov2d: np.ndarray,
                      n_std: float = 1.0, **kwargs) -> None:
    """Draw a 1-σ uncertainty ellipse from a 2×2 covariance matrix."""
    try:
        vals, vecs = np.linalg.eigh(cov2d)
        vals = np.maximum(vals, 0)          # numerical safety
        w, h = 2 * n_std * np.sqrt(vals)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        from matplotlib.patches import Ellipse
        ell = Ellipse(xy=mean, width=w, height=h, angle=angle,
                      edgecolor="limegreen", facecolor="none",
                      lw=1.2, alpha=0.6, zorder=6)
        ax.add_patch(ell)
    except Exception:
        pass   # silently skip if cov is degenerate


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

    fig2, ani2 = make_problem2()

    plt.show()
