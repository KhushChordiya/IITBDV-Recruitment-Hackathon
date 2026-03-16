import numpy as np

# ── PID state (persistent across calls) ──────────────────────────────────────
_pid_integral   = 0.0
_pid_prev_error = 0.0

# ─── PURE PURSUIT STEERING ───────────────────────────────────────────────────
def steering(path: list[dict], state: dict) -> float:
    """
    Pure Pursuit controller.
    Finds a lookahead point on the path and computes the required steering angle.
    """
    if not path or len(path) < 2:
        return 0.0

    x, y, yaw = state["x"], state["y"], state["yaw"]
    vx = max(state["vx"], 1.0)  # avoid division by zero at standstill
    wheelbase = 2.6

    # Adaptive lookahead: faster = look further ahead
    lookahead = np.clip(0.6 * vx, 2.5, 12.0)

    # Find the best lookahead point on the path
    target = None
    min_dist_ahead = float("inf")

    for wp in path:
        dx = wp["x"] - x
        dy = wp["y"] - y
        dist = np.hypot(dx, dy)

        # Only consider points roughly ahead of the car
        point_angle = np.arctan2(dy, dx)
        angle_diff  = np.arctan2(np.sin(point_angle - yaw),
                                  np.cos(point_angle - yaw))

        if abs(angle_diff) < np.pi / 2 and abs(dist - lookahead) < min_dist_ahead:
            min_dist_ahead = abs(dist - lookahead)
            target = wp

    # Fallback: just use the last waypoint
    if target is None:
        target = path[-1]

    # Compute steering angle using Pure Pursuit geometry
    dx = target["x"] - x
    dy = target["y"] - y

    # Transform target into vehicle frame
    local_x =  np.cos(-yaw) * dx - np.sin(-yaw) * dy   # noqa (unused but good for debug)
    local_y =  np.sin(-yaw) * dx + np.cos(-yaw) * dy

    if abs(local_y) < 1e-6 and abs(local_x) < 1e-6:
        return 0.0

    dist_to_target = np.hypot(local_x, local_y)
    # Pure Pursuit formula: steer = arctan(2 * L * sin(alpha) / ld)
    alpha = np.arctan2(local_y, local_x)
    steer = np.arctan2(2.0 * wheelbase * np.sin(alpha), dist_to_target)

    return float(np.clip(steer, -0.5, 0.5))


# ─── CURVATURE-BASED TARGET SPEED ────────────────────────────────────────────
def compute_target_speed(path: list[dict], state: dict) -> float:
    """
    Look ahead on the path and slow down proportionally to upcoming curvature.
    High curvature (tight corner) → lower speed. Straight → max speed.
    """
    max_speed  = 16.0   # m/s (~58 km/h) on straights
    min_speed  =  4.0   # m/s on very tight corners
    lookahead_pts = 8   # how many waypoints to scan ahead

    x, y = state["x"], state["y"]

    # Find closest waypoint index
    dists = [np.hypot(wp["x"] - x, wp["y"] - y) for wp in path]
    closest = int(np.argmin(dists))

    # Grab the next N waypoints
    window = path[closest: closest + lookahead_pts + 2]
    if len(window) < 3:
        return min_speed

    # Estimate average curvature over the window
    curvatures = []
    for i in range(1, len(window) - 1):
        ax, ay = window[i-1]["x"], window[i-1]["y"]
        bx, by = window[i  ]["x"], window[i  ]["y"]
        cx, cy = window[i+1]["x"], window[i+1]["y"]

        # Menger curvature formula
        d1 = np.hypot(bx - ax, by - ay)
        d2 = np.hypot(cx - bx, cy - by)
        d3 = np.hypot(cx - ax, cy - ay)
        area = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
        denom = d1 * d2 * d3
        if denom < 1e-6:
            continue
        curvatures.append(area / denom)

    if not curvatures:
        return max_speed

    max_curv = max(curvatures)
    # Map curvature → speed  (high curv = slow)
    # Tune 0.3 / 1.5 to taste
    speed = np.interp(max_curv, [0.0, 0.3], [max_speed, min_speed])
    return float(np.clip(speed, min_speed, max_speed))


# ─── PID THROTTLE ────────────────────────────────────────────────────────────
def throttle_algorithm(target_speed: float,
                       current_speed: float,
                       dt: float) -> tuple[float, float]:
    global _pid_integral, _pid_prev_error

    # PID gains — tune these
    Kp, Ki, Kd = 0.6, 0.05, 0.08

    error = target_speed - current_speed
    _pid_integral   += error * dt
    _pid_integral    = np.clip(_pid_integral, -10.0, 10.0)  # anti-windup
    derivative       = (error - _pid_prev_error) / max(dt, 1e-6)
    _pid_prev_error  = error

    output = Kp * error + Ki * _pid_integral + Kd * derivative

    if output >= 0:
        return float(np.clip(output, 0.0, 1.0)), 0.0
    else:
        return 0.0, float(np.clip(-output * 0.5, 0.0, 1.0))  # gentle braking


# ─── MAIN CONTROL FUNCTION ───────────────────────────────────────────────────
def control(
    path: list[dict],
    state: dict,
    cmd_feedback: dict,
    step: int,
) -> tuple[float, float, float]:
    dt = 0.05  # 50ms timestep

    steer        = steering(path, state)
    target_speed = compute_target_speed(path, state)
    throttle, brake = throttle_algorithm(target_speed, state["vx"], dt)

    return throttle, steer, brake
