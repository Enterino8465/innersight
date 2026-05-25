import logging
import innersight.backend.config as cfg


def test_path_attributes_are_strings():
    for attr in ('DATA_DIR', 'MODEL_DIR', 'ALERTS_FILE', 'CORRECTIONS_FILE',
                 'BLOCK_LOG_FILE', 'PREPROCESSOR_FILE', 'BEST_MODEL_FILE',
                 'BEST_MODEL_PT_FILE', 'STANDARDIZER_FILE', 'LDAP_FILE'):
        assert isinstance(getattr(cfg, attr), str), f'{attr} must be a str'


def test_default_training_config_keys():
    required = {'epochs', 'batch_size', 'lr', 'layer_sizes', 'pos_weight', 'patience'}
    assert required == set(cfg.DEFAULT_TRAINING_CONFIG.keys())


def test_default_training_config_values():
    c = cfg.DEFAULT_TRAINING_CONFIG
    assert isinstance(c['epochs'], int) and c['epochs'] > 0
    assert isinstance(c['batch_size'], int) and c['batch_size'] > 0
    assert isinstance(c['lr'], float) and c['lr'] > 0
    assert isinstance(c['pos_weight'], float) and c['pos_weight'] > 0
    assert isinstance(c['patience'], int) and c['patience'] > 0
    assert isinstance(c['layer_sizes'], list) and c['layer_sizes'][-1] == 1


def test_feature_cols_nonempty():
    assert len(cfg.FEATURE_COLS) > 0
    assert all(isinstance(c, str) for c in cfg.FEATURE_COLS)


def test_business_hours_ordering():
    assert 0 <= cfg.BUSINESS_HOURS_START < cfg.BUSINESS_HOURS_END <= 24


def test_setup_logging_does_not_raise():
    cfg.setup_logging(logging.WARNING)
