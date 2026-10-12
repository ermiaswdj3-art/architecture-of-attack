import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PitchConfig:
    length: float = 105.0
    width: float = 68.0
    num_cells: int = 305
    side_length: float = 3.0

@dataclass
class Event:
    event_id: int
    start_timestamp: float   # FIX: was a single 'timestamp' before
    end_timestamp: float     # FIX: duration = end - start, not match clock
    player_id: str
    team_id: str
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    action_type: str

    @property
    def duration(self) -> float:
        return max(self.end_timestamp - self.start_timestamp, 0.1)  # floor to avoid div/0

class HexGridMapper:
    def __init__(self, config: PitchConfig):
        self.config = config
        self.hex_width = np.sqrt(3) * config.side_length
        self.hex_height = 2.0 * config.side_length

    def map_to_cell(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        x, y = pos
        q = int(x / (3/2 * self.config.side_length))
        r = int(y / (np.sqrt(3)/2 * self.config.side_length))
        return (q, r)

class MetricFootballEngine:
    def __init__(self, config: PitchConfig, weights=(0.4, 0.4, 0.2)):
        self.config = config
        self.mapper = HexGridMapper(config)
        self.w1, self.w2, self.w3 = weights

    def calculate_territorial_velocity(self, event: Event) -> float:
        """FIX (Bug 1): dt is now the actual duration of THIS action,
        not the match clock at which it occurred."""
        dx = event.end_pos[0] - event.start_pos[0]
        v_t = dx / event.duration
        return max(v_t, 0.0)

    def compute_compactness(self, cell_occupations):
        if not cell_occupations:
            return 0.0
        coords = np.array(cell_occupations)
        variance = np.var(coords[:, 0]) + np.var(coords[:, 1])
        return 1.0 / (1.0 + variance)

    @staticmethod
    def minmax_normalize(values: List[float]) -> List[float]:
        """FIX (Bug 2): put every AMI component on a common 0-1 scale
        BEFORE weighting, so the weights actually mean what they say."""
        arr = np.array(values, dtype=float)
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-9:
            return [0.5] * len(arr)  # no spread in this sample -> neutral
        return list((arr - lo) / (hi - lo))

    def evaluate_ami(self, occupation_time_norm, avg_velocity_norm, compactness_norm):
        """All three inputs are expected pre-normalized to [0,1]."""
        return float((self.w1*occupation_time_norm) + (self.w2*avg_velocity_norm) + (self.w3*compactness_norm))


if __name__ == "__main__":
    config = PitchConfig()
    engine = MetricFootballEngine(config)

    # Same three actions as before, but now with realistic action durations
    # (a pass/carry takes ~1-3 real seconds, NOT 12-18 seconds of match clock)
    events = [
        Event(1, 610.0, 611.8, "P1", "TeamA", (20.0, 30.0), (35.0, 32.0), "carry"),
        Event(2, 612.0, 613.5, "P2", "TeamA", (35.0, 32.0), (55.0, 40.0), "pass"),
        Event(3, 613.7, 614.6, "P3", "TeamA", (55.0, 40.0), (75.0, 35.0), "pass"),
    ]

    print("--- Per-event Territorial Velocity (now using real action duration) ---")
    velocities = []
    cells = []
    for ev in events:
        vt = engine.calculate_territorial_velocity(ev)
        velocities.append(vt)
        cells.append(engine.mapper.map_to_cell(ev.end_pos))
        dx = ev.end_pos[0]-ev.start_pos[0]
        print(f"Event {ev.event_id}: dx={dx:.1f}m over {ev.duration:.1f}s -> Vt={vt:.2f} m/s")

    # NOTE: with a single sequence there's only one T_occ/C_compact value,
    # so normalization needs a comparison set. We build one across a few
    # synthetic possession sequences to show the weights actually matter now.
    def synth_sequence(seed):
        rng = np.random.default_rng(seed)
        n = rng.integers(3, 7)
        evs = []
        t = 0.0
        x, y = rng.uniform(0, 40), rng.uniform(10, 58)
        for i in range(n):
            dur = rng.uniform(0.8, 2.5)
            dx = rng.uniform(-5, 20)
            dy = rng.uniform(-8, 8)
            nx, ny = np.clip(x+dx, 0, 105), np.clip(y+dy, 0, 68)
            evs.append(Event(i, t, t+dur, f"P{i}", "TeamA", (x, y), (nx, ny), "pass"))
            t += dur
            x, y = nx, ny
        return evs

    sequences = [synth_sequence(s) for s in range(30)]

    raw_t_occ, raw_vt, raw_compact = [], [], []
    for seq in sequences:
        v = [engine.calculate_territorial_velocity(e) for e in seq]
        c = [engine.mapper.map_to_cell(e.end_pos) for e in seq]
        raw_t_occ.append(len(set(c)) / config.num_cells)
        raw_vt.append(np.mean(v))
        raw_compact.append(engine.compute_compactness(c))

    norm_t_occ = engine.minmax_normalize(raw_t_occ)
    norm_vt = engine.minmax_normalize(raw_vt)
    norm_compact = engine.minmax_normalize(raw_compact)

    amis = [engine.evaluate_ami(t, v, c) for t, v, c in zip(norm_t_occ, norm_vt, norm_compact)]

    print(f"\n--- AMI across {len(sequences)} synthetic possession sequences (normalized inputs) ---")
    print(f"AMI range: {min(amis):.3f} to {max(amis):.3f}, mean {np.mean(amis):.3f}")

    # Show contribution shares now that inputs are on comparable scales
    i = int(np.argmax(amis))
    contrib_t = engine.w1*norm_t_occ[i]
    contrib_v = engine.w2*norm_vt[i]
    contrib_c = engine.w3*norm_compact[i]
    total = contrib_t + contrib_v + contrib_c
    print(f"\nHighest-AMI sequence (#{i}): AMI={amis[i]:.3f}")
    print(f"  Contribution shares -> T_occ={contrib_t/total*100:.1f}%  Vt={contrib_v/total*100:.1f}%  Compact={contrib_c/total*100:.1f}%")
