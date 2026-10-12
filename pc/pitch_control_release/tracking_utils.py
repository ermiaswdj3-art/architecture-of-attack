"""
Shared loader and velocity utilities for Metrica Sports tracking-data CSVs
(Sample_Game_1 format: normalized [0,1] coordinates, 25Hz, one file per team).
"""
import csv

PITCH_LENGTH = 105.0  # meters
PITCH_WIDTH = 68.0    # meters

def load_team(path):
    """Load a Metrica-format team tracking CSV into a list of per-frame dicts:
    {'time': float, 'players': {player_id: (x_m, y_m)}, 'ball': (x_m, y_m)}
    Coordinates are converted from normalized [0,1] to meters
    ((0,0)=top-left corner, (1,1)=bottom-right, per Metrica's documented convention)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    player_cols = [c[:-1] for c in rows[0].keys() if c.endswith('x') and c != 'Ballx']
    frames = []
    for row in rows:
        t = float(row['Time'])
        players = {}
        for p in player_cols:
            xs, ys = row[p + 'x'], row[p + 'y']
            if xs == '' or xs.upper() == 'NAN':
                continue  # player not on pitch this frame (sub not yet entered)
            players[p] = (float(xs) * PITCH_LENGTH, float(ys) * PITCH_WIDTH)
        ball = (float(row['Ballx']) * PITCH_LENGTH, float(row['Bally']) * PITCH_WIDTH)
        frames.append({'time': t, 'players': players, 'ball': ball})
    return frames

def compute_velocities(frames, idx):
    """Finite-difference velocity at frame idx, in m/s.
    Central difference where possible; forward/backward at sequence ends.
    Uses REAL frame spacing from the data -- do not downsample before calling
    this, or velocity estimates will be smoothed/distorted (see README)."""
    n = len(frames)
    i0, i1 = max(0, idx - 1), min(n - 1, idx + 1)
    dt = frames[i1]['time'] - frames[i0]['time']
    vel = {}
    p0, p1 = frames[i0]['players'], frames[i1]['players']
    for pid in p1:
        if pid in p0 and dt > 0:
            vel[pid] = ((p1[pid][0]-p0[pid][0])/dt, (p1[pid][1]-p0[pid][1])/dt)
        else:
            vel[pid] = (0.0, 0.0)
    return vel
