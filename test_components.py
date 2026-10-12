"""Do interpretable features predict shots once possession length is controlled? Stratify by n actions."""
import numpy as np, collections
from load import load_matches, load_events
from validate_outcomes import auc  # reuses definitions (re-runs its prints on import)
