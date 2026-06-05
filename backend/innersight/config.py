"""Central configuration for InnerSight UEBA.

Reads all tuneable values from environment variables or falls back to
sensible defaults.  Import individual constants from here; never hard-code
paths or thresholds elsewhere in the codebase.
"""

import logging
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# Bundled synthetic demo dataset (repo-root data/synthetic_demo). Used as a
# fallback so `docker-compose up` with no env vars boots into a working demo.
# config.py is backend/innersight/config.py → three dirnames reach the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO_DATA_DIR = os.environ.get(
    'INNERSIGHT_DEMO_DATA_DIR', os.path.join(_REPO_ROOT, 'data', 'synthetic_demo'))

# Real data directory if INNERSIGHT_DATA_DIR is set; otherwise the synthetic demo.
DATA_DIR  = os.environ.get('INNERSIGHT_DATA_DIR', '') or DEMO_DATA_DIR
MODEL_DIR = os.environ.get('INNERSIGHT_MODEL_DIR',  'checkpoints')

ALERTS_FILE       = os.path.join(MODEL_DIR, 'alerts.json')
CORRECTIONS_FILE  = os.path.join(MODEL_DIR, 'corrections.json')
BLOCK_LOG_FILE    = os.path.join(MODEL_DIR, 'block_log.json')
BEST_MODEL_PT_FILE  = os.path.join(MODEL_DIR, 'best_model.pt')    # PyTorch checkpoint
STANDARDIZER_FILE   = os.path.join(MODEL_DIR, 'standardizer.pt')  # PyTorch standardizer
# LDAP: loaded from DATA_DIR/LDAP/ directory (monthly snapshots). See Phase 1 pipeline.

# ── Default temporal split dates (override via config YAML per version) ───────
TRAIN_END_DATE = '2010-09-30'
VAL_END_DATE   = '2010-11-30'

# ── Feature engineering ───────────────────────────────────────────────────────
BUSINESS_HOURS_START   = 7
BUSINESS_HOURS_END     = 19
LARGE_ATTACHMENT_BYTES = 1_048_576
JOB_KEYWORDS           = ('job', 'career', 'linkedin', 'indeed')
CLOUD_KEYWORDS         = ('dropbox', 'wikileaks', 'pastebin')
INTERNAL_DOMAIN        = 'dtaa.com'

FEATURE_COLS = [
    'logon_count', 'logoff_count', 'after_hours_logons', 'weekend_logons', 'unique_pcs_used',
    'usb_connect_count', 'usb_disconnect_count', 'after_hours_usb',
    'file_count', 'file_to_removable_count', 'unique_filenames',
    'email_sent_count', 'email_to_external_count', 'large_attachment_count', 'total_email_size',
    'http_request_count', 'job_search_visits', 'cloud_upload_visits',
]

# ── CERT Version Families ────────────────────────────────────────────────────
# Maps user-facing version string to its adapter family.
VERSION_FAMILIES: dict[str, str] = {
    'r1':   'r1',
    'r2':   'r2',
    'r3.1': 'r3x', 'r3.2': 'r3x',
    'r4.1': 'r4x', 'r4.2': 'r4x',
    'r5.1': 'r5x', 'r5.2': 'r5x',
    'r6.1': 'r6x', 'r6.2': 'r6x',
}

# ── Job search keywords (from CERT readme + actual URLs in r4.2) ─────────────
# Overrides the incomplete JOB_KEYWORDS above for the universal pipeline.
# Keep old JOB_KEYWORDS for backward compat; new code should use CERT_JOB_DOMAINS.
CERT_JOB_DOMAINS: tuple[str, ...] = (
    'monster.com', 'careerbuilder.com', 'craigslist.org',
    'jobhuntersbible.com', 'aol.com/jobs',
    'job', 'career', 'linkedin', 'indeed',
)

CERT_CLOUD_DOMAINS: tuple[str, ...] = (
    'wikileaks.org', 'dropbox',
)

CERT_KEYLOGGER_DOMAINS: tuple[str, ...] = (
    'refog.com', 'softactivity.com', 'keylogger',
)

# ── Chunked reading ──────────────────────────────────────────────────────────
CSV_CHUNK_SIZE: int = 50_000  # rows per chunk for streaming large CSVs (r6.2 http = 85M rows)

# ── Date range ───────────────────────────────────────────────────────────────
CERT_DATE_START = '2010-01-01'
CERT_DATE_END   = '2011-06-01'

# ── Training ──────────────────────────────────────────────────────────────────
DEFAULT_TRAINING_CONFIG = {
    'epochs':      50,
    'batch_size':  64,
    'lr':          0.001,
    'layer_sizes': [18, 64, 32, 1],
    'pos_weight':  50.0,
    'patience':    5,
    'seed':        42,
}

CORRECTION_LR = 0.0001

# ── Per-user baseline (Module 1) ──────────────────────────────────────────────
# EMA-smoothed baseline of each user's normal daily behaviour. The deviation of
# a day from this baseline (z-score) is the model input for Module 1.
DEFAULT_BASELINE_CONFIG = {
    'ema_alpha':        0.05,   # EMA smoothing factor; half-life ≈ 13.5 days
    'min_history_days': 14,     # days of history before the baseline is reliable
    'std_floor_ratio':  0.1,    # std floor = this × global median std per feature
    'variance_eps':     1e-6,   # prevent division by zero in EMA variance
}

# ── Sliding windows (Module 1 training labels) ────────────────────────────────
# Days are grouped into overlapping fixed-length windows; each window is labelled
# by how much it overlaps a known attack period.
DEFAULT_WINDOW_CONFIG = {
    'window_size':                28,    # days per sliding window
    'window_stride':              7,     # slide by this many days
    'overlap_positive_threshold': 0.5,   # window is positive if ≥50% overlaps an attack
    # Windows with 1–49% overlap are EXCLUDED from training (ambiguous);
    # windows with 0% overlap are negative.
}


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
