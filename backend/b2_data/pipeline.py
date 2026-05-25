import logging
import os
import pandas as pd

from innersight.backend.config import DATA_DIR, TRAIN_END_DATE, VAL_END_DATE

logger = logging.getLogger(__name__)

LOG_FILES = ['logon', 'device', 'file', 'email', 'http']

# Minimum columns required by feature engineering for each log type.
_REQUIRED_COLS = {
    'logon':  {'date', 'user', 'activity', 'pc'},
    'device': {'date', 'user', 'activity'},
    'file':   {'date', 'user', 'filename'},
    'email':  {'date', 'user', 'to', 'size', 'id'},
    'http':   {'date', 'user', 'url'},
}

# Empty-frame column sets returned when a CSV is absent, so callers always
# get a DataFrame with the expected schema rather than a bare empty frame.
_EMPTY_COLS = {
    'logon':  ['id', 'date', 'user', 'pc', 'activity'],
    'device': ['id', 'date', 'user', 'pc', 'activity'],
    'file':   ['id', 'date', 'user', 'filename', 'to_removable_media'],
    'email':  ['id', 'date', 'user', 'to', 'cc', 'bcc', 'size', 'attachments'],
    'http':   ['id', 'date', 'user', 'url', 'content'],
}


def load_raw_logs(data_dir=DATA_DIR):
    dfs = {}
    for name in LOG_FILES:
        path = os.path.join(data_dir, f'{name}.csv')
        try:
            df = pd.read_csv(path, parse_dates=['date'])
        except FileNotFoundError:
            logger.warning('load_raw_logs | %s not found: %s — using empty DataFrame', name, path)
            dfs[name] = pd.DataFrame(columns=_EMPTY_COLS[name])
            continue
        except Exception as exc:
            logger.error('load_raw_logs | failed to parse %s: %s', path, exc)
            raise

        missing = _REQUIRED_COLS[name] - set(df.columns)
        if missing:
            raise ValueError(
                f'{path} is missing required columns: {sorted(missing)}. '
                f'Found: {sorted(df.columns.tolist())}'
            )

        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        dfs[name] = df
        logger.info('load_raw_logs | %s: %d rows', name, len(df))
    return dfs


def load_labels(answers_dir):
    if not os.path.exists(answers_dir):
        logger.warning('load_labels | answers directory not found: %s — no labels loaded', answers_dir)
        return set()

    malicious = set()
    for fname in os.listdir(answers_dir):
        if not fname.endswith('.csv'):
            continue
        path = os.path.join(answers_dir, fname)
        try:
            df = pd.read_csv(path, parse_dates=['date'])
        except Exception as exc:
            logger.warning('load_labels | skipping malformed answer file %s: %s', path, exc)
            continue

        missing = {'date', 'user'} - set(df.columns)
        if missing:
            logger.warning('load_labels | %s missing columns %s — skipping', fname, sorted(missing))
            continue

        for _, row in df.iterrows():
            malicious.add((row['user'], row['date'].date()))

    logger.info('load_labels | total malicious events: %d', len(malicious))
    return malicious


def time_split(logs_dict, train_end, val_end):
    t_train = pd.Timestamp(train_end)
    t_val   = pd.Timestamp(val_end)

    splits = {'train': {}, 'val': {}, 'test': {}}
    for name, df in logs_dict.items():
        if 'date' not in df.columns:
            if not df.empty:
                raise ValueError(
                    f"DataFrame '{name}' is missing a 'date' column required for time_split. "
                    f"Found columns: {sorted(df.columns.tolist())}"
                )
            # Columnless empty frame: propagate as empty splits without filtering.
            empty = pd.DataFrame(columns=df.columns)
            splits['train'][name] = splits['val'][name] = splits['test'][name] = empty
            continue
        splits['train'][name] = df[df['date'] <= t_train].reset_index(drop=True)
        splits['val'][name]   = df[(df['date'] > t_train) & (df['date'] <= t_val)].reset_index(drop=True)
        splits['test'][name]  = df[df['date'] > t_val].reset_index(drop=True)
        logger.info(
            'time_split | %s: train=%d val=%d test=%d',
            name,
            len(splits['train'][name]),
            len(splits['val'][name]),
            len(splits['test'][name]),
        )
    return splits


def load_data(data_dir=DATA_DIR):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f'Data directory not found: {data_dir}. '
            'Set the INNERSIGHT_DATA_DIR environment variable to the correct path.'
        )

    logs   = load_raw_logs(data_dir)
    labels = load_labels(os.path.join(data_dir, 'answers'))
    splits = time_split(logs, train_end=TRAIN_END_DATE, val_end=VAL_END_DATE)
    return {'splits': splits, 'labels': labels}
