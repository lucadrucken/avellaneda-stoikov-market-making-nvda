from pathlib import Path

# --- Pfade ---
EVENTS_PATH = Path("../data/processed/nvda_202606_events.parquet")
FEATURES_PATH = Path("../data/processed/nvda_202606_features.parquet")

# --- Markt und Order ---
TICK_SIZE = 0.01
ORDER_SIZE = 20.0
PARTICIPATION_RATE = 0.05
FILL_RULE = "conservative"

K_MIN = -10
K_MAX = 20

# --- Session ---
SESSION_START_UTC = "13:30"
SESSION_END_UTC = "20:00"
MINUTES_PER_SESSION = 390


def _minutes_since_midnight(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


_derived_minutes_per_session = (
    _minutes_since_midnight(SESSION_END_UTC) - _minutes_since_midnight(SESSION_START_UTC)
)

if _derived_minutes_per_session != MINUTES_PER_SESSION:
    raise ValueError(
        f"MINUTES_PER_SESSION ({MINUTES_PER_SESSION}) stimmt nicht mit der aus "
        f"SESSION_START_UTC/SESSION_END_UTC abgeleiteten Sessionlaenge "
        f"({_derived_minutes_per_session}) ueberein. SESSION_START_UTC/END_UTC "
        f"geaendert, ohne MINUTES_PER_SESSION anzupassen (oder umgekehrt)."
    )


def ladder_column_name(label: str, k: int) -> str:
    return f"{label}_ladder_{k:+03d}"


# --- Volatilitaet ---
HALF_LIFE_MINUTES = 60.0

# --- k-Bestimmung ---
K_MODE = "estimate"
K_ROBUSTNESS_VALUES = (50.0, 100.0, 200.0)
K_SENSITIVITY_VALUES = K_ROBUSTNESS_VALUES

K_ESTIMATION_CRITERIA = {
    "min_r_squared": 0.80,
    "min_support_points": 5,
    "min_fills_per_point": 30,
    "k_plausible_range": (10.0, 1000.0),
    "require_positive_k": True,
    "require_ci_excludes_zero": True,
    "max_curvature_pvalue": 0.05,
}

# --- Fixierte Kalibrierungsparameter (Stand 08.08.2026) ---
K_MAIN = 21.4454
GAMMA_SWEEP = (1.0e-05, 3.0e-05, 1.0e-04)
K_ROBUSTNESS_GAMMA = 3.0e-05
NAIVE_HALF_SPREAD = 0.05
Q_MAX = 300.0