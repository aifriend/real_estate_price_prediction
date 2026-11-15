"""Tests for TrainService.pre_process_for and scaler persistence, on a tiny synthetic frame."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.service.TrainService import TrainService

DROPPED = ['street', 'neighbourhood_cleansed',
           'neighbourhood_group_cleansed', 'city']
# Column positions (after the drop above) that pre_process_for removes.
REMOVED_POSITIONS = [1, 7, 11, 12, 13, 15, 17, 18, 19, 20, 23, 27, 28, 29,
                     30, 31, 34, 36, 37, 38, 39, 40, 41, 42, 43, 47, 49, 50, 51]
N_FEATURE_COLUMNS = 56


def _synthetic_listings(n_rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {c: ['s'] * n_rows for c in DROPPED}
    for i in range(N_FEATURE_COLUMNS):
        data[f'f{i}'] = rng.normal(size=n_rows)
    # one text column, which must be label-encoded
    data['f3'] = ['a', 'b', 'c', 'd'] * (n_rows // 4)
    data['price'] = rng.uniform(30, 120, size=n_rows)
    # keep price outside the removed positions: put it at index 2
    cols = [c for c in data if c not in DROPPED + ['price']]
    cols.insert(2, 'price')
    return pd.DataFrame({c: data[c] for c in DROPPED + cols})


def test_pre_process_for_shapes_and_split():
    df = _synthetic_listings(20)
    n_features = N_FEATURE_COLUMNS + 1 - len(REMOVED_POSITIONS) - 1  # minus price

    Xtr, Xts, ytr, yts = TrainService.pre_process_for(df)

    assert Xtr.shape == (14, n_features)  # 70% of 20 rows
    assert Xts.shape == (6, n_features)
    assert len(ytr) == 14 and len(yts) == 6
    assert np.issubdtype(Xtr.dtype, np.number)  # text was encoded


def test_pre_process_for_is_deterministic():
    first = TrainService.pre_process_for(_synthetic_listings(20))
    second = TrainService.pre_process_for(_synthetic_listings(20))
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_pre_process_for_requires_expected_columns():
    df = _synthetic_listings(20).drop(columns=['street'])
    with pytest.raises(KeyError):
        TrainService.pre_process_for(df)


def test_train_persists_scaler_and_models(tmp_path, monkeypatch):
    """train() must save the fitted scaler next to the models it trains."""
    saved = {}

    def fake_save(model, project_dir, parent=0, apex='default'):
        saved[apex] = model

    monkeypatch.setattr(
        'src.service.TrainService.DataService.save_model', staticmethod(fake_save))
    monkeypatch.setattr('src.service.TrainService.plt.show', lambda: None)
    monkeypatch.setattr(
        'src.service.TrainService.GradientBoostingRegressor',
        lambda **kw: __import__('sklearn.ensemble', fromlist=['x']).GradientBoostingRegressor(n_estimators=3))
    monkeypatch.setattr(
        'src.service.TrainService.RandomForestRegressor',
        lambda **kw: __import__('sklearn.ensemble', fromlist=['x']).RandomForestRegressor(n_estimators=3, max_depth=3, random_state=0))
    real_xgb = __import__('xgboost').XGBRegressor
    monkeypatch.setattr(
        'src.service.TrainService.XGBRegressor',
        lambda **kw: real_xgb(n_estimators=3, max_depth=2, n_jobs=1))

    rng = np.random.default_rng(1)
    X = rng.normal(loc=5, scale=3, size=(40, 4))
    y = X[:, 0] * 2 + 10
    xgb, rescaled = TrainService.train(X[:30], X[30:], y[:30], y[30:], tmp_path)

    scaler = saved['price_scaler']
    assert isinstance(scaler, StandardScaler)
    np.testing.assert_allclose(scaler.transform(X[:30]), rescaled)
    assert abs(rescaled.mean()) < 1e-9
    assert xgb is not None
