"""Heterogeneous graph schema for the InnerSight UEBA system.

The graph is built from CERT event logs and represents user behaviour
as a network of entities and interactions. A single graph covers one time
window (e.g. one week); nodes accumulate feature vectors from the raw logs
and edges carry the temporal/contextual attributes of each event.

Node types
----------
user  — an employee (user_id from CERT logs)
pc    — a workstation or server (pc field in logon/device/file logs)
url   — a URL visited via the HTTP proxy log (deduplicated by domain or full URL)
file  — a file path that was copied to removable media (from the file log)

Edge types
----------
(user, logon,        pc)    — user authenticated on a machine
(user, usb_connect,  pc)    — user plugged in a USB device at a machine
(user, email_to,     user)  — user sent an email to another user
(user, http_request, url)   — user visited a URL
(user, file_copy,    file)  — user copied a file to removable media

Email is modelled as a direct user→user edge rather than through a dedicated
email-message node. This keeps the schema simple; an EMAIL_MSG node type can
be added later if per-message attributes are needed for attention scoring.

Reverse edges are included for every forward edge so that bidirectional
message passing works out of the box with PyG's HeteroConv layers.

Feature dimensions
------------------
Node features are behavioural aggregates computed over the time window.
Edge features are per-event contextual attributes captured at event time.
The exact lists are finalised in Tasks 4-5; the constants here serve as the
single authoritative source referenced by the builder and model code.
"""

# ── Node type constants ───────────────────────────────────────────────────────

NODE_USER = 'user'
"""An employee identified by their CERT user_id."""

NODE_PC = 'pc'
"""A workstation or server identified by its pc field in the log."""

NODE_URL = 'url'
"""A URL (or deduplicated domain) visited via the HTTP proxy."""

NODE_FILE = 'file'
"""A file path that was copied to removable media."""

# ── Forward edge type constants ───────────────────────────────────────────────
# Convention: (source_node_type, relation_name, destination_node_type)

EDGE_LOGON = (NODE_USER, 'logon', NODE_PC)
"""User authenticated on a machine (logon.csv, activity == Logon)."""

EDGE_USB = (NODE_USER, 'usb_connect', NODE_PC)
"""User connected a USB device at a machine (device.csv, activity == Connect)."""

EDGE_EMAIL = (NODE_USER, 'email_to', NODE_USER)
"""User sent an email to another internal user (email.csv, internal recipient)."""

EDGE_HTTP = (NODE_USER, 'http_request', NODE_URL)
"""User visited a URL via the HTTP proxy (http.csv)."""

EDGE_FILE_COPY = (NODE_USER, 'file_copy', NODE_FILE)
"""User copied a file to removable media (file.csv, to_removable_media == True)."""

# ── Reverse edge type constants ───────────────────────────────────────────────
# PyG's HeteroConv requires explicit reverse edges for bidirectional message
# passing. The relation name is prefixed with 'rev_' by convention.

REV_EDGE_LOGON = (NODE_PC, 'rev_logon', NODE_USER)
"""Reverse of EDGE_LOGON: machine → user for inbound message passing."""

REV_EDGE_USB = (NODE_PC, 'rev_usb_connect', NODE_USER)
"""Reverse of EDGE_USB: machine → user."""

REV_EDGE_EMAIL = (NODE_USER, 'rev_email_to', NODE_USER)
"""Reverse of EDGE_EMAIL: recipient → sender (same node type, reverse direction)."""

REV_EDGE_HTTP = (NODE_URL, 'rev_http_request', NODE_USER)
"""Reverse of EDGE_HTTP: URL → user."""

REV_EDGE_FILE_COPY = (NODE_FILE, 'rev_file_copy', NODE_USER)
"""Reverse of EDGE_FILE_COPY: file → user."""

# ── Collections ───────────────────────────────────────────────────────────────

ALL_NODE_TYPES: list[str] = [NODE_USER, NODE_PC, NODE_URL, NODE_FILE]
"""All node types in canonical order (user first — it is the prediction target)."""

ALL_EDGE_TYPES: list[tuple[str, str, str]] = [
    EDGE_LOGON,
    EDGE_USB,
    EDGE_EMAIL,
    EDGE_HTTP,
    EDGE_FILE_COPY,
]
"""Forward edge types only."""

ALL_REV_EDGE_TYPES: list[tuple[str, str, str]] = [
    REV_EDGE_LOGON,
    REV_EDGE_USB,
    REV_EDGE_EMAIL,
    REV_EDGE_HTTP,
    REV_EDGE_FILE_COPY,
]
"""Reverse edge types only, parallel to ALL_EDGE_TYPES."""

ALL_EDGES: list[tuple[str, str, str]] = ALL_EDGE_TYPES + ALL_REV_EDGE_TYPES
"""All edge types (forward + reverse) as a single flat list."""

# ── Node feature dimensions ───────────────────────────────────────────────────
# These are placeholder values finalised in Tasks 4-5.
# All downstream model code should read from NODE_FEATURE_DIMS rather than
# hard-coding numbers so that a single change here propagates everywhere.

USER_FEATURE_DIM = 16
"""Behavioural aggregates per time window: logon count, after-hours rate,
USB events, email volume, external email ratio, HTTP volume, job-search visits,
cloud-upload visits, unique PCs, unique URLs, unique recipients, file copies,
large attachments, weekend activity, avg session length, risk score lag."""

PC_FEATURE_DIM = 8
"""Machine characteristics: num_distinct_users, total_logon_events, is_shared,
after_hours_usage_ratio, usb_event_count, mean_session_count,
weekend_usage_ratio, unique_days_active."""

URL_FEATURE_DIM = 8
"""Domain visit patterns: total_visits, unique_visitors, is_job_related,
is_cloud_related, mean_daily_visits, visitor_concentration,
weekend_visit_ratio, after_hours_visit_ratio."""

FILE_FEATURE_DIM = 6
"""Removable-media copy patterns: total_copies, unique_users, file_type_code,
avg_copy_hour, is_after_hours_copy, content_topic_count."""

NODE_FEATURE_DIMS: dict[str, int] = {
    NODE_USER: USER_FEATURE_DIM,
    NODE_PC:   PC_FEATURE_DIM,
    NODE_URL:  URL_FEATURE_DIM,
    NODE_FILE: FILE_FEATURE_DIM,
}
"""Map from node type string → feature vector length."""

# ── Edge feature dimensions ───────────────────────────────────────────────────

LOGON_EDGE_DIM = 4
"""Per-logon event features: hour_of_day, is_weekend, is_after_hours, duration_hours."""

USB_EDGE_DIM = 3
"""Per-USB-connect event features: hour_of_day, is_weekend, is_after_hours."""

EMAIL_EDGE_DIM = 5
"""Per-email event features: log1p_size_bytes, attachment_count,
is_external_recipient, hour_of_day, is_after_hours."""

HTTP_EDGE_DIM = 3
"""Per-HTTP-request event features: hour_of_day, is_weekend, is_after_hours."""

FILE_EDGE_DIM = 4
"""Per-file-copy event features: hour_of_day, is_weekend, is_after_hours,
file_type_code (0=doc, 1=archive, 2=executable, 3=other)."""

EDGE_FEATURE_DIMS: dict[tuple[str, str, str], int] = {
    EDGE_LOGON:     LOGON_EDGE_DIM,
    EDGE_USB:       USB_EDGE_DIM,
    EDGE_EMAIL:     EMAIL_EDGE_DIM,
    EDGE_HTTP:      HTTP_EDGE_DIM,
    EDGE_FILE_COPY: FILE_EDGE_DIM,
}
"""Map from forward edge type tuple → edge feature vector length.
Reverse edges carry the same features as their forward counterpart."""


# ── Schema summary ────────────────────────────────────────────────────────────

def print_schema() -> None:
    """Print a formatted summary of all node types, edge types, and dimensions."""
    W = 70
    print('=' * W)
    print('InnerSight Heterogeneous Graph Schema')
    print('=' * W)

    print('\nNode types')
    print('-' * W)
    print(f"  {'type':<12} {'feature_dim':>11}   description")
    print(f"  {'':<12} {'':>11}   -----------")
    descs = {
        NODE_USER: 'behavioural aggregates per time window',
        NODE_PC:   'machine characteristics',
        NODE_URL:  'URL visit patterns',
        NODE_FILE: 'removable-media copy patterns',
    }
    for ntype in ALL_NODE_TYPES:
        dim = NODE_FEATURE_DIMS[ntype]
        print(f"  {ntype:<12} {dim:>11}   {descs[ntype]}")

    print('\nForward edge types')
    print('-' * W)
    print(f"  {'(src, relation, dst)':<38} {'edge_dim':>8}")
    print(f"  {'':<38} {'--------':>8}")
    for etype in ALL_EDGE_TYPES:
        label = str(etype)
        dim   = EDGE_FEATURE_DIMS[etype]
        print(f"  {label:<38} {dim:>8}")

    print('\nReverse edge types  (same edge features as forward)')
    print('-' * W)
    for etype in ALL_REV_EDGE_TYPES:
        print(f"  {str(etype)}")

    print('\nTotals')
    print('-' * W)
    total_node_feats = sum(NODE_FEATURE_DIMS.values())
    total_edge_feats = sum(EDGE_FEATURE_DIMS.values())
    print(f"  node types        : {len(ALL_NODE_TYPES)}")
    print(f"  forward edges     : {len(ALL_EDGE_TYPES)}")
    print(f"  total edge types  : {len(ALL_EDGES)}  (forward + reverse)")
    print(f"  total node feats  : {total_node_feats}  (sum across all node types)")
    print(f"  total edge feats  : {total_edge_feats}  (sum across forward edge types)")
    print('=' * W)


if __name__ == '__main__':
    print_schema()
