# Simultaneous Localization and Mapping (SLAM) – Problem Statement

You are given a simplified 2D driverless-racing simulation in [racing.py](racing.py).
Your task is to implement the SLAM pipeline inside the `Solution` class by completing:

- `data_association(measurements, current_map)`
- `localization(velocity, steering)`
- `mapping(measurements)`

The track map is loaded from [small_track.csv](small_track.csv) and the simulation already provides:

- car state: `self.pos`, `self.heading`
- sensor measurements: noisy cone positions in the **car local frame**
- helper transform: `local_to_global()`
- a driving controller for generating steering commands

---

## 1) Data Association

### Goal
Match each current sensor measurement to an existing landmark in the map.

### Inputs
- `measurements`: detected cone points in local frame, shape `(N, 2)`
- `current_map`: known map cone points in global frame, shape `(M, 2)`

### Expected behavior
1. Convert `measurements` from local frame to global frame using current pose.
2. For each transformed measurement, find its nearest neighbor in `current_map`.
3. Return an integer array of length `N`, where each value is the matched map index.
4. Handle empty inputs safely (no crash, return empty output).

### Suggested method
Use Euclidean distance (nearest-neighbor association).

---

## 2) Localization

### Goal
Update the vehicle pose estimate over time using a kinematic bicycle model.

### Inputs
- `velocity` (`v`)
- `steering` (`δ`)

### Expected behavior
Use the discrete update with timestep `DT` and wheelbase `WHEELBASE`:

$$
x_{t+1} = x_t + v \cos(\psi_t)\,DT
$$

$$
y_{t+1} = y_t + v \sin(\psi_t)\,DT
$$

$$
\psi_{t+1} = \psi_t + \frac{v}{L}\tan(\delta)\,DT
$$

Then normalize heading to $[-\pi, \pi]$.

---

## 3) Mapping

### Goal
Build a global map of cone landmarks from repeated local sensor observations.

### Inputs
- `measurements`: cone detections in local frame `(N, 2)`

### Expected behavior
1. Convert local measurements to global coordinates.
2. Add each point to `self.learned_map` only if it is not a duplicate.
3. Use a distance threshold (e.g., about `2.0 m`) to decide if a point is new.
4. Keep map data persistent across frames.

---

## Implementation Notes

- Keep implementations robust for zero measurements.
- Use NumPy for vector math and pairwise distances.
- Preserve function signatures and class structure.
- Do not change simulation constants unless necessary.

---

## Deliverable

Complete the three methods in `Solution` so that:

- Problem 1 visualization shows correct measurement-to-map matching.
- Problem 2 visualization shows smooth dead-reckoning trajectory updates.
- Problem 3 visualization shows incremental map growth with deduplication.

