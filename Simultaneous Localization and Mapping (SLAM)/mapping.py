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
CAR_START_HEADING = float(_cs["direction"])

MAP_CONES = np.vstack([BLUE_CONES, YELLOW_CONES])

def _build_centerline():
    center = np.mean(MAP_CONES, axis=0)
    D      = distance.cdist(BLUE_CONES, YELLOW_CONES)
    mids   = np.array(
        [(BLUE_CONES[i] + YELLOW_CONES[np.argmin(D[i])]) / 2.0
         for i in range(len(BLUE_CONES))]
    )
    angles = np.arctan2(mids[:, 1] - center[1], mids[:, 0] - center[0])
    return mids[np.argsort(angles)[::-1]]

CENTERLINE = _build_centerline()

SENSOR_RANGE = 12.0
NOISE_STD    = 0.20
WHEELBASE    = 3.0
DT           = 0.1
SPEED        = 7.0
LOOKAHEAD    = 5.5
N_FRAMES     = 130

_OBS_COV       = np.eye(2) * (NOISE_STD ** 2)
_MIN_SIGHTINGS = 3
_MAX_MISSES    = 30
_MERGE_MAHA2   = 4.0


class _Landmark:
    _id_ctr = 0

    def __init__(self, x, y):
        _Landmark._id_ctr += 1
        self.id        = _Landmark._id_ctr
        self.mu        = np.array([x, y], dtype=float)
        self.P         = _OBS_COV.copy()
        self.sightings = 1
        self.misses    = 0

    def update(self, x, y):
        S     = self.P + _OBS_COV
        K     = self.P @ np.linalg.inv(S)
        innov = np.array([x, y]) - self.mu
        self.mu        = self.mu + K @ innov
        self.P         = (np.eye(2) - K) @ self.P
        self.sightings += 1
        self.misses     = 0

    @property
    def confirmed(self):
        return self.sightings >= _MIN_SIGHTINGS


def angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def pure_pursuit(pos, heading, path):
    dists   = np.linalg.norm(path - pos, axis=1)
    nearest = int(np.argmin(dists))
    n       = len(path)
    target  = path[(nearest + 5) % n]
    for k in range(nearest, nearest + n):
        pt = path[k % n]
        if np.linalg.norm(pt - pos) >= LOOKAHEAD:
            target = pt
            break
    alpha = angle_wrap(np.arctan2(target[1]-pos[1], target[0]-pos[0]) - heading)
    steer = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), LOOKAHEAD)
    return float(np.clip(steer, -0.6, 0.6))

def local_to_global(local_pts, pos, heading):
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, -s], [s, c]])
    return (R @ local_pts.T).T + pos

def get_measurements(pos, heading):
    dists   = np.linalg.norm(MAP_CONES - pos, axis=1)
    visible = MAP_CONES[dists < SENSOR_RANGE]
    if len(visible) == 0:
        return np.zeros((0, 2))
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, s], [-s, c]])
    local = (R @ (visible - pos).T).T
    return local + np.random.normal(0, NOISE_STD, local.shape)

def step_kinematic(pos, heading, velocity, steering):
    new_pos = pos.copy()
    new_pos[0] += velocity * np.cos(heading) * DT
    new_pos[1] += velocity * np.sin(heading) * DT
    new_heading = angle_wrap(heading + (velocity / WHEELBASE) * np.tan(steering) * DT)
    return new_pos, new_heading

def draw_track(ax, alpha_b=0.4, alpha_y=0.4):
    ax.scatter(BLUE_CONES[:,0], BLUE_CONES[:,1], c="royalblue", marker="^", s=65, alpha=alpha_b, zorder=2, label="Blue cones")
    ax.scatter(YELLOW_CONES[:,0], YELLOW_CONES[:,1], c="gold", marker="^", s=65, alpha=alpha_y, zorder=2, label="Yellow cones")
    ax.scatter(BIG_ORANGE[:,0], BIG_ORANGE[:,1], c="darkorange", marker="s", s=100, alpha=0.7, zorder=2, label="Start gate")

def draw_car(ax, pos, heading):
    ax.scatter(pos[0], pos[1], c="red", s=160, zorder=7, label="Car")
    ax.arrow(pos[0], pos[1], 2.2*np.cos(heading), 2.2*np.sin(heading), head_width=0.8, fc="red", ec="red", zorder=8)

def setup_ax(ax, subtitle=""):
    ax.set_xlim(-28, 28); ax.set_ylim(-22, 22); ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    if subtitle: ax.set_title(subtitle, fontsize=10)


class Bot:
    def __init__(self):
        self.pos     = CAR_START_POS.copy()
        self.heading = CAR_START_HEADING

    def data_association(self, measurements, current_map): raise NotImplementedError
    def localization(self, velocity, steering):            raise NotImplementedError
    def mapping(self, measurements):                       raise NotImplementedError


class Solution(Bot):
    def __init__(self):
        super().__init__()
        self.learned_map   = []           # confirmed landmark means (public API)
        self._landmarks    = []           # list of _Landmark (confirmed+tentative)
        self._tentative_pts = np.zeros((0, 2))   # for visualisation

    # ------------------------------------------------------------------
    def mapping(self, measurements):
        """
        Improved landmark mapping using:

        1. Recursive Bayesian update (EKF, H=I)
           Each new observation of an existing landmark is fused with the
           running estimate via a Kalman update, weighting contributions by
           covariance.  Position uncertainty narrows monotonically as
           sightings accumulate — raw-append ignores this entirely.

        2. Landmark lifecycle  (tentative → confirmed → pruned)
           A landmark must be seen >= MIN_SIGHTINGS (=3) times before it
           joins the authoritative map.  Tentative landmarks not re-observed
           within MAX_MISSES steps are pruned, preventing glitches from
           polluting the map permanently.

        3. Mahalanobis-gated nearest-neighbour matching
           Association uses the per-landmark covariance as the innovation
           matrix: a measurement must be statistically consistent with an
           existing landmark to update it; otherwise a new tentative
           landmark is spawned.

        4. Covariance-weighted merging of confirmed duplicates
           After each batch, any two confirmed landmarks whose Mahalanobis
           distance is below MERGE_MAHA2 are fused, handling the edge case
           where two tentative tracks for the same physical cone both reach
           confirmation.

        Parameters
        ----------
        measurements : np.ndarray (M, 2) – cone observations in LOCAL frame
        """
        if len(measurements) == 0:
            self._increment_misses()
            return

        gm = local_to_global(measurements, self.pos, self.heading)

        observed_ids = set()
        for pt in gm:
            lm = self._find_closest(pt)
            if lm is None:
                self._landmarks.append(_Landmark(pt[0], pt[1]))
            else:
                lm.update(pt[0], pt[1])
                observed_ids.add(lm.id)

        for lm in self._landmarks:
            if lm.id not in observed_ids:
                lm.misses += 1
        self._landmarks = [
            lm for lm in self._landmarks
            if lm.confirmed or lm.misses < _MAX_MISSES
        ]

        self._merge_confirmed()

        confirmed = [lm for lm in self._landmarks if lm.confirmed]
        self.learned_map = [lm.mu.copy() for lm in confirmed]

        tentative = [lm for lm in self._landmarks if not lm.confirmed]
        self._tentative_pts = (
            np.array([lm.mu for lm in tentative]) if tentative else np.zeros((0, 2))
        )

    def _find_closest(self, pt):
        best_lm, best_d2 = None, _MERGE_MAHA2 + 1.0
        for lm in self._landmarks:
            S = lm.P + _OBS_COV
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue
            diff = pt - lm.mu
            d2   = float(diff @ S_inv @ diff)
            if d2 < best_d2:
                best_d2, best_lm = d2, lm
        return best_lm

    def _merge_confirmed(self):
        merged = True
        while merged:
            merged = False
            conf = [lm for lm in self._landmarks if lm.confirmed]
            for i in range(len(conf)):
                for j in range(i + 1, len(conf)):
                    a, b = conf[i], conf[j]
                    try:
                        d2 = float((a.mu - b.mu) @ np.linalg.inv(a.P + b.P) @ (a.mu - b.mu))
                    except np.linalg.LinAlgError:
                        continue
                    if d2 < _MERGE_MAHA2:
                        wa, wb = np.linalg.inv(a.P), np.linalg.inv(b.P)
                        P_new  = np.linalg.inv(wa + wb)
                        a.mu   = P_new @ (wa @ a.mu + wb @ b.mu)
                        a.P    = P_new
                        a.sightings += b.sightings
                        if b in self._landmarks:
                            self._landmarks.remove(b)
                        merged = True
                        break
                if merged:
                    break

    def _increment_misses(self):
        for lm in self._landmarks:
            lm.misses += 1
        self._landmarks = [
            lm for lm in self._landmarks
            if lm.confirmed or lm.misses < _MAX_MISSES
        ]


# ── Problem 3 – Mapping ───────────────────────────────────────────────────────
def make_problem3():
    """
    Visualise improved incremental mapping.

    Colour coding:
      limegreen ×  = confirmed landmarks (seen >= MIN_SIGHTINGS, Bayesian-fused)
      orange    ·  = tentative landmarks (not yet confirmed)
      ellipses     = 1-sigma position uncertainty per confirmed landmark
    """
    sol = Solution()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(
        "Problem 3 – Mapping  (Bayesian Fusion + Lifecycle Management)",
        fontsize=13, fontweight="bold"
    )

    def update(frame):
        ax.clear()
        steer = pure_pursuit(sol.pos, sol.heading, CENTERLINE)
        meas  = get_measurements(sol.pos, sol.heading)
        sol.pos, sol.heading = step_kinematic(sol.pos, sol.heading, SPEED, steer)
        sol.mapping(meas)

        draw_track(ax, alpha_b=0.15, alpha_y=0.15)

        if sol.learned_map:
            lm_arr = np.array(sol.learned_map)
            ax.scatter(lm_arr[:, 0], lm_arr[:, 1],
                       c="limegreen", marker="x", s=90, linewidths=2.0,
                       zorder=5, label=f"Confirmed ({len(sol.learned_map)})")
            for lm in sol._landmarks:
                if lm.confirmed:
                    _draw_cov_ellipse(ax, lm.mu, lm.P)

        if len(sol._tentative_pts) > 0:
            ax.scatter(sol._tentative_pts[:, 0], sol._tentative_pts[:, 1],
                       c="orange", marker=".", s=40, alpha=0.6,
                       zorder=4, label=f"Tentative ({len(sol._tentative_pts)})")

        draw_car(ax, sol.pos, sol.heading)
        setup_ax(ax,
            f"Frame {frame+1}/{N_FRAMES}  –  "
            f"confirmed: {len(sol.learned_map)} / {len(MAP_CONES)} cones  |  "
            f"tentative: {len(sol._tentative_pts)}")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=100, repeat=True)
    return fig, ani


def _draw_cov_ellipse(ax, mean, cov2d, n_std=1.0):
    try:
        vals, vecs = np.linalg.eigh(cov2d)
        vals  = np.maximum(vals, 0)
        w, h  = 2 * n_std * np.sqrt(vals)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        from matplotlib.patches import Ellipse
        ell = Ellipse(xy=mean, width=w, height=h, angle=angle,
                      edgecolor="limegreen", facecolor="none",
                      lw=0.8, alpha=0.45, zorder=4)
        ax.add_patch(ell)
    except Exception:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Driverless Car Hackathon – SLAM Visualisation ===")
    print(f"  Blue cones   : {len(BLUE_CONES)}")
    print(f"  Yellow cones : {len(YELLOW_CONES)}")
    print(f"  Big orange   : {len(BIG_ORANGE)}")
    print(f"  Car start    : {CAR_START_POS}  heading={np.degrees(CAR_START_HEADING):.1f}°")
    print(f"  Centerline   : {len(CENTERLINE)} waypoints (clockwise)")
    print("\nOpening 1 animation window …")

    fig3, ani3 = make_problem3()
    plt.show()
