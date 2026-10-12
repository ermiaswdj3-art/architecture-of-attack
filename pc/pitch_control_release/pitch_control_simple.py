"""
Simplified time-to-intercept pitch control (v1): compares each team's
FASTEST-arriving player at a target point via a static logistic on the
time gap. Legitimate first-pass approximation, but discards information
about numerical advantage near a location -- see pitch_control_full.py
for the upgraded version and README.md for a side-by-side comparison.
"""
import math
from tracking_utils import load_team, compute_velocities

REACTION_TIME = 0.7      # s, literature default
MAX_PLAYER_SPEED = 5.0   # m/s, literature default (controlled running, not sprint)
LOGISTIC_LAMBDA = 1.0

def time_to_intercept(pos, vel, target):
    rx = pos[0] + vel[0] * REACTION_TIME
    ry = pos[1] + vel[1] * REACTION_TIME
    dist_remaining = math.hypot(target[0] - rx, target[1] - ry)
    return REACTION_TIME + dist_remaining / MAX_PLAYER_SPEED

def team_min_arrival(positions, velocities, target):
    return min(time_to_intercept(positions[pid], velocities[pid], target) for pid in positions)

def control_probability(home_pos, home_vel, away_pos, away_vel, target):
    t_home = team_min_arrival(home_pos, home_vel, target)
    t_away = team_min_arrival(away_pos, away_vel, target)
    return 1.0 / (1.0 + math.exp(-LOGISTIC_LAMBDA * (t_away - t_home)))


if __name__ == "__main__":
    home_frames = load_team("home_snippet.csv")
    away_frames = load_team("away_snippet.csv")
    FRAME_IDX = 4
    home_pos = home_frames[FRAME_IDX]['players']
    away_pos = away_frames[FRAME_IDX]['players']
    home_vel = compute_velocities(home_frames, FRAME_IDX)
    away_vel = compute_velocities(away_frames, FRAME_IDX)

    print(f"--- Simplified (v1) model, frame {FRAME_IDX+1}, t={home_frames[FRAME_IDX]['time']}s ---")
    for x in [30.0, 52.5, 75.0]:
        probs = [control_probability(home_pos, home_vel, away_pos, away_vel, (x, y))
                 for y in [10, 20, 34, 48, 58]]
        print(f"  x={x:5.1f}m: avg P(home control)={sum(probs)/len(probs):.3f}  "
              f"[{', '.join(f'{p:.2f}' for p in probs)}]")
