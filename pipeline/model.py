"""Walk-forward Random Forest training, plus same-day prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from .config import Config


@dataclass
class FoldResult:
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    n_train:     int
    n_test:      int
    test_r2:     float
    feat_imp:    pd.Series


RNG_SEED = 42


def walk_forward_rf(stock_panel: pd.DataFrame,
                    feature_cols: Iterable[str],
                    target_col: str,
                    cfg: Config,
                    verbose: bool = False) -> tuple[pd.DataFrame, list[FoldResult]]:
    """Train an RF in expanding-window walk-forward mode.
    Returns: (predictions_df[date,ticker,y_true,y_pred], list_of_FoldResult)."""
    feature_cols = list(feature_cols)
    df = stock_panel.dropna(subset=feature_cols + [target_col]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    dates = df["date"].drop_duplicates().sort_values().to_numpy()
    train_min_days = (cfg.train_years + cfg.val_years) * 252
    if len(dates) <= train_min_days + cfg.refit_every_days:
        raise ValueError(
            f"Not enough history: {len(dates)} trading days, need >"
            f"{train_min_days + cfg.refit_every_days}.")

    fold_starts = list(range(train_min_days, len(dates), cfg.refit_every_days))
    preds, folds = [], []

    for k, start_idx in enumerate(fold_starts):
        train_dates = dates[:start_idx]
        end_idx = min(start_idx + cfg.refit_every_days, len(dates))
        test_dates = dates[start_idx:end_idx]

        train = df[df["date"].isin(train_dates)]
        test  = df[df["date"].isin(test_dates)]

        rf = RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            max_features=cfg.max_features,
            n_jobs=-1, random_state=RNG_SEED + k,
        )
        rf.fit(train[feature_cols].astype("float32").to_numpy(),
               train[target_col].astype("float32").to_numpy())
        y_pred = rf.predict(test[feature_cols].astype("float32").to_numpy())

        out = test[["date", "ticker", target_col]].rename(columns={target_col: "y_true"}).copy()
        out["y_pred"] = y_pred
        preds.append(out)

        y_test = test[target_col].astype("float32").to_numpy()
        fr = FoldResult(
            train_start=pd.Timestamp(train_dates[0]),
            train_end=pd.Timestamp(train_dates[-1]),
            test_start=pd.Timestamp(test_dates[0]),
            test_end=pd.Timestamp(test_dates[-1]),
            n_train=len(train), n_test=len(test),
            test_r2=r2_score(y_test, y_pred) if len(y_test) > 0 else float("nan"),
            feat_imp=pd.Series(rf.feature_importances_, index=feature_cols),
        )
        folds.append(fr)
        if verbose:
            print(f"Fold {k:>2}: train {fr.train_start.date()}..{fr.train_end.date()} "
                  f"({fr.n_train:,}) | test {fr.test_start.date()}..{fr.test_end.date()} "
                  f"({fr.n_test:,}) | R^2 = {fr.test_r2:+.4%}")

    return pd.concat(preds, ignore_index=True), folds


def fit_full_model(stock_panel: pd.DataFrame,
                   feature_cols: Iterable[str],
                   target_col: str,
                   cfg: Config) -> RandomForestRegressor:
    """Fit a single RF on ALL available history. Use this for next-day predictions."""
    feature_cols = list(feature_cols)
    train = stock_panel.dropna(subset=feature_cols + [target_col])
    rf = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features,
        n_jobs=-1, random_state=RNG_SEED,
    )
    rf.fit(train[feature_cols].astype("float32").to_numpy(),
           train[target_col].astype("float32").to_numpy())
    return rf


def predict_latest(model: RandomForestRegressor,
                   ranked_panel: pd.DataFrame,
                   feature_cols: Iterable[str]) -> pd.DataFrame:
    """Return predictions for the latest date present in `ranked_panel`."""
    feature_cols = list(feature_cols)
    last_date = ranked_panel["date"].max()
    g = ranked_panel[ranked_panel["date"] == last_date].copy()
    g = g.dropna(subset=feature_cols)
    g["y_pred"] = model.predict(g[feature_cols].astype("float32").to_numpy())
    return g[["date", "ticker", "y_pred"]].sort_values("y_pred", ascending=False).reset_index(drop=True)
