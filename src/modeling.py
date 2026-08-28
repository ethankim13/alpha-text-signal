"""
Turns model_dataset.csv into a feature matrix, defines the models being
compared, and runs a walk-forward backtest across all of them. This is
the module that actually produces the project's finding.
"""
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

FEATURE_COLUMNS = ["drift_score_mda", "drift_score_risk_factors"]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Converts the raw CSV columns into a numeric X/y pair scikit-learn
    models can consume directly."""
    df = df.copy()

    # risk_factors drift is NaN for boilerplate quarters (Phase 1b's
    # "no material change" filings). Filling with the column mean keeps
    # the row usable via drift_score_mda instead of losing it outright.
    df["drift_score_risk_factors"] = df["drift_score_risk_factors"].fillna(
        df["drift_score_risk_factors"].mean()
    )

    # sector is text ("Technology", "Airlines", ...) — one-hot encoding
    # turns it into one 0/1 column per sector, since a linear/tree model
    # can't use a category directly, only numbers.
    sector_dummies = pd.get_dummies(df["sector"], prefix="sector")

    X = pd.concat([df[FEATURE_COLUMNS], sector_dummies], axis=1)
    y = df["forward_return"]
    return X, y


def get_models() -> dict:
    """One of each model type being compared. Defaults are light-touch,
    not hyperparameter-tuned — with ~160 rows, an aggressive tuning search
    risks overfitting the search itself rather than just the model."""
    return {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42),
    }


def walk_forward_splits(df: pd.DataFrame, n_folds: int = 5, min_train_fraction: float = 0.5) -> list[tuple[list, list]]:
    """
    df must already be sorted chronologically by filing_date, index reset.

    Uses an EXPANDING window: fold 1 trains on the first
    min_train_fraction of rows (in time order) and tests on the next
    slice; fold 2 trains on everything through fold 1's test slice (the
    training set grows), and so on. This guarantees a model never trains
    on data from after whatever it's predicting — the whole point of a
    walk-forward evaluation over a single train/test split.
    """
    n = len(df)
    min_train_size = int(n * min_train_fraction)
    remaining = n - min_train_size
    fold_size = remaining // n_folds

    splits = []
    train_end = min_train_size
    for fold in range(n_folds):
        test_start = train_end
        test_end = test_start + fold_size if fold < n_folds - 1 else n  # last fold takes any leftover rows
        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))
        if test_idx:
            splits.append((train_idx, test_idx))
        train_end = test_end

    return splits


def run_walk_forward_backtest(df: pd.DataFrame) -> dict:
    """Runs every model across every walk-forward fold, pools all
    out-of-sample predictions together, and reports R^2 and correlation
    per model — the actual finding this project is testing for."""
    df = df.sort_values("filing_date").reset_index(drop=True)
    X, y = prepare_features(df)
    splits = walk_forward_splits(df)

    results = {}
    for name, model in get_models().items():
        all_preds = []
        all_actuals = []

        for train_idx, test_idx in splits:
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            all_preds.extend(preds)
            all_actuals.extend(y_test.tolist())

        r2 = r2_score(all_actuals, all_preds)
        corr, _ = pearsonr(all_actuals, all_preds)

        results[name] = {
            "r2": r2,
            "correlation": corr,
            "n_predictions": len(all_preds),
        }

    return results