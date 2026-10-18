"""
Turns model_dataset.csv into a feature matrix, defines the models being
compared, and runs a walk-forward out-of-sample evaluation across all of them.

This measures predictive accuracy only (R^2 and correlation). It does NOT
simulate positions, transaction costs, portfolio returns, or drawdowns.
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

FEATURE_COLUMNS = ["drift_score_mda", "drift_score_risk_factors"]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Converts the CSV columns into a numeric X/y pair. Missing values
    are left as NaN here; they are filled per fold inside the backtest."""
    df = df.copy()

    # One-hot encode sector: one 0.0/1.0 column per sector.
    sector_dummies = pd.get_dummies(df["sector"], prefix="sector", dtype=float)

    X = pd.concat([df[FEATURE_COLUMNS], sector_dummies], axis=1)
    y = df["forward_return"]
    return X, y


def get_models() -> dict:
    """One of each model type. Light-touch defaults, no tuning."""
    return {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42),
    }


def walk_forward_splits(df: pd.DataFrame, n_folds: int = 5, min_train_fraction: float = 0.5):
    """
    df must be sorted by filing_date with the index reset.

    Expanding window: the first fold starts with the earliest
    min_train_fraction of rows, and each later fold adds the previous
    test rows to training.

    Purging: a training row is kept only if its 5-day outcome
    (target_end_date) finished BEFORE the first test filing.
    """
    n = len(df)
    min_train_size = int(n * min_train_fraction)
    fold_size = (n - min_train_size) // n_folds

    filing_dates = pd.to_datetime(df["filing_date"])
    outcome_dates = pd.to_datetime(df["target_end_date"])

    splits = []
    train_end = min_train_size
    for fold in range(n_folds):
        test_start = train_end
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = list(range(test_start, test_end))
        if not test_idx:
            continue

        first_test_date = filing_dates.iloc[test_idx[0]]
        train_idx = [i for i in range(train_end) if outcome_dates.iloc[i] < first_test_date]
        if not train_idx:
            raise ValueError(f"No eligible training rows in fold {fold + 1}")

        splits.append((train_idx, test_idx))
        train_end = test_end

    return splits


def run_walk_forward_backtest(df: pd.DataFrame) -> dict:
    """Walk-forward out-of-sample evaluation (not a trading backtest).
    Pools all out-of-sample predictions per model and reports R^2 and correlation."""
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

            # Learn column averages from TRAINING rows only, then fill both sets.
            imputer = SimpleImputer(strategy="mean")
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            model.fit(X_train_imp, y_train)
            preds = model.predict(X_test_imp)

            all_preds.extend(preds)
            all_actuals.extend(y_test.tolist())

        r2 = r2_score(all_actuals, all_preds)
        corr, _ = pearsonr(all_actuals, all_preds)

        results[name] = {"r2": r2, "correlation": corr, "n_predictions": len(all_preds)}

    return results