"""Central configuration for InnerSight UEBA.

Reads all tuneable values from environment variables or falls back to
sensible defaults.  Import individual constants from here; never hard-code
paths or thresholds elsewhere in the codebase.
"""

import logging
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = os.environ.get('INNERSIGHT_DATA_DIR',   '')
MODEL_DIR = os.environ.get('INNERSIGHT_MODEL_DIR',  'innersight/data')

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

# ── Training ──────────────────────────────────────────────────────────────────
DEFAULT_TRAINING_CONFIG = {
    'epochs':      50,
    'batch_size':  64,
    'lr':          0.001,
    'layer_sizes': [18, 64, 32, 1],
    'pos_weight':  50.0,
    'patience':    5,
}

CORRECTION_LR = 0.0001


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
