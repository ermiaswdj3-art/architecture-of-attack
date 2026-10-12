"""
Full time-integrated pitch control (v2), following the structure published
in Spearman (2018) "Beyond Expected Goals" / the Fernandez & Bornn (2018)
pitch-control literature: every player contributes a time-dependent
probability of having individually established control, integrated as a
race between all attackers and all defenders over time. This is the
upgrade from pitch_control_simple.py's single-fastest-player comparison.

PROVENANCE NOTE: implemented from the published model STRUCTURE (arrival-
time race -> logistic-in-time control function -> integrated differential
control equation), not ported from a specific codebase I had open in this
session. Parameters (TTI_SIGMA, LAMBDA_CONTROL) are commonly-cited
literature defaults, not independently re-derived or fit to this data.
Known omissions vs. fuller published versions: no defensive-advantage
term, no goalkeeper-specific modeling.
"""
import math
import numpy as np
from tracking_utils import load_team, compute_velocities

REACTION_TIME = 0.7
MAX_PLAYER_SPEED = 5.0
TTI_SIGMA = 0.45
LAMBDA_CONTROL = 4.3
BALL_SPEED = 15.0
DT = 0.04
MAX_INTEGRATION_TIME = 10.0

def time_to_intercept(pos, vel, target):
    rx = pos[0] + vel[0] * REACTION_TIME
    ry = pos[1] + vel[1] * REACTION_TIME
    dist_remaining = math.hypot(target[0] - rx, target[1] - ry)
    return REACTION_TIME + dist_remaining / MAX_PLAYER_SPEED

def arrival_probability(t, T_i):
    return 1.0 / (1.0 + math.exp(-np.pi / (math.sqrt(3.0) * TTI_SIGMA) * (t - T_i)))

def integrated_pitch_control(home_pos, home_vel, away_pos, away_vel, ball_pos, target):
    ball_travel_time = math.hypot(target[0]-ball_pos[0], target[1]-ball_pos[1]) / BALL_SPEED
    home_T = {pid: time_to_intercept(home_pos[pid], home_vel[pid], target) for pid in home_pos}
    away_T = {pid: time_to_intercept(away_pos[pid], away_vel[pid], target) for pid in away_pos}
    PPCF_home = {pid: 0.0 for pid in home_pos}
    PPCF_away = {pid: 0.0 for pid in away_pos}
    t = ball_travel_time
    total = 0.0
    steps = 0
    while total < 0.99 and t < MAX_INTEGRATION_TIME:
        remaining = 1.0 - total
        for pid, T_i in home_T.items():
            PPCF_home[pid] += remaining * arrival_probability(t, T_i) * LAMBDA_CONTROL * DT
        for pid, T_i in away_T.items():
            PPCF_away[pid] += remaining * arrival_probability(t, T_i) * LAMBDA_CONTROL * DT
        total = sum(PPCF_home.values()) + sum(PPCF_away.values())
        t += DT
        steps += 1
        if steps > int(MAX_INTEGRATION_TIME / DT):
            break
    total_home, total_away = sum(PPCF_home.values()), sum(PPCF_away.values())
    norm = total_home + total_away
    return (0.5 if norm < 1e-9 else total_home / norm), steps


if __name__ == "__main__":
    home_frames = load_team("home_snippet.csv")
    away_frames = load_team("away_snippet.csv")
    FRAME_IDX = 4
    home_pos = home_frames[FRAME_IDX]['players']
    away_pos = away_frames[FRAME_IDX]['players']
    home_vel = compute_velocities(home_frames, FRAME_IDX)
    away_vel = compute_velocities(away_frames, FRAME_IDX)
    ball_pos = home_frames[FRAME_IDX]['ball']

    print(f"--- Full time-integrated (v2) model, frame {FRAME_IDX+1}, t={home_frames[FRAME_IDX]['time']}s ---")
    for x in [30.0, 52.5, 75.0]:
        probs = [integrated_pitch_control(home_pos, home_vel, away_pos, away_vel, ball_pos, (x, y))[0]
                 for y in [10, 20, 34, 48, 58]]
        print(f"  x={x:5.1f}m: avg P(home control)={sum(probs)/len(probs):.3f}  "
              f"[{', '.join(f'{p:.2f}' for p in probs)}]")
