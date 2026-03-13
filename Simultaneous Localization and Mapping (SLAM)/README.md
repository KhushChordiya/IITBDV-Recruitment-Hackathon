# SLAM Recruitment Hackathon

This challenge is intentionally **open-ended**.

You are given working templates and baseline implementations. Your goal is to push performance as far as you can and demonstrate engineering judgment, not just complete TODOs.

## Core Objective
Improve overall SLAM quality under noise and partial observability by making the best possible upgrades to:

- data association
- localization
- mapping

There is no single “correct” solution. We care about your reasoning, tradeoffs, and measurable improvements.

## Project Structure (keep responsibilities separate)

- `data_association.py` → improve `data_association(measurements, current_map)`
- `localization.py` → improve `localization(velocity, steering)`
- `mapping.py` → improve `mapping(measurements)`

Each file should remain independently runnable and focused on its own task.

## What is already provided in the templates

The starter code is already runnable and includes:

- track loading and preprocessing from `small_track.csv`
- sensor simulation with noise
- local↔global coordinate transforms
- visualization/animation scaffolding
- a baseline controller for generating steering commands

Baseline algorithm currently present in each target function:

- `data_association.py`:
	- `data_association(...)` uses nearest-neighbor matching after transforming measurements to global frame
- `localization.py`:
	- `localization(...)` uses a kinematic bicycle dead-reckoning update
- `mapping.py`:
	- `mapping(...)` builds a landmark map with distance-threshold deduplication

## What you should change (mandatory scope)

Participants should only modify one target function per file:

- `data_association(...)` in `data_association.py`
- `localization(...)` in `localization.py`
- `mapping(...)` in `mapping.py`

Do not modify unrelated scaffolding unless absolutely necessary for a justified improvement.

## Suggested Advanced Directions

You can go beyond nearest-neighbor and dead-reckoning. Examples:

### 1) Data Association
- Gated nearest-neighbor (Mahalanobis or adaptive distance thresholds)
- Global assignment (Hungarian algorithm) instead of greedy matching
- Probabilistic association (JPDA-style ideas)
- Outlier rejection / clutter handling
- Temporal consistency in landmark identities across frames

### 2) Localization
- Motion + observation fusion (EKF / UKF / Error-state filtering)
- Particle filter localization with resampling strategies
- Bias/noise modeling and covariance propagation
- Slip-aware or adaptive vehicle models
- Smoothing across time (fixed-lag smoothing)

### 3) Mapping
- Landmark state estimation with uncertainty (mean + covariance)
- Landmark update via recursive filtering instead of raw append
- Robust deduplication and landmark lifecycle (init / confirm / prune)
- Occupancy/grid or submap ideas where useful
- Loop-consistency logic when revisiting regions

### 4) End-to-End SLAM Extensions (optional)
- EKF-SLAM / FastSLAM style formulations
- Graph-SLAM concepts (pose graph + optimization)
- Joint optimization of trajectory and landmark map

## What to Submit

- Your improved code
- Brief notes on what changed and why
- Any assumptions and limitations
- Evidence of improvement (plots, metrics, or concise comparison vs baseline)

## Evaluation Focus

We evaluate both **results** and **engineering quality**:

- Accuracy and robustness of associations, trajectory, and map
- Stability under noisy measurements and edge cases
- Clarity and maintainability of code
- Soundness of design choices and tradeoff discussion
- Practicality: improvements that are effective without overcomplication

## Practical Guidance

- Start by establishing a baseline metric and visual behavior.
- Improve one component at a time, then re-evaluate end-to-end effects.
- Keep changes interpretable; explain why each change helps.
- Favor robust behavior over brittle “best-case” tuning.

Build boldly. We expect strong candidates to iterate, experiment, and justify decisions with evidence.
