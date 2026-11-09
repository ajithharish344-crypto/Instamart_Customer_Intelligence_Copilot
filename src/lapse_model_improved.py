"""Part B: Leakage-safe customer lapse prediction with oversampling."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from src.customer_models import load, rfm_and_segments

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def build_lapse_model():
    # 1. Define the historical period and prediction period
    customers, orders = load()
    cutoff = pd.Timestamp("2025-09-01")
    end = pd.Timestamp("2025-10-01")

    hist = orders[orders.order_ts < cutoff].copy()
    completed = hist[hist.status.eq("completed")].copy()

    # 2. Build customer features using historical data only
    cohort = completed.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean")
    ).reset_index()

    cohort["recency"] = (cutoff - cohort["last_order"]).dt.days
    cohort = cohort.drop(columns=["last_order"])

    final_orders = hist[hist.status.isin(["completed", "cancelled"])]
    cancellation = final_orders.groupby("customer_id").status.agg(
        total="count",
        cancelled=lambda x: (x == "cancelled").sum()
    ).reset_index()

    cohort = cohort.merge(cancellation, on="customer_id", how="left")
    cohort["cancellation_rate"] = (
        cohort["cancelled"] / cohort["total"]
    )

    # Favourite product category from historical completed orders
    products = pd.read_csv(ROOT / "data/clean/products.csv")

    category_counts = completed.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left"
    ).groupby(["customer_id", "category"]).size().rename("count").reset_index()

    favourite = (
        category_counts
        .sort_values(
            ["customer_id", "count", "category"],
            ascending=[True, False, True]
        )
        .drop_duplicates("customer_id")
        .rename(columns={"category": "favourite_category"})
        [["customer_id", "favourite_category"]]
    )

    cohort = cohort.merge(favourite, on="customer_id", how="left")
    cohort = cohort.merge(customers, on="customer_id", how="left")

    cohort["signup_date"] = pd.to_datetime(cohort["signup_date"])
    cohort["tenure_days"] = (
        cutoff - cohort["signup_date"]
    ).dt.days

    # 3. Create the lapse label using September outcomes
    september_completed = orders[
        (orders.order_ts >= cutoff)
        & (orders.order_ts < end)
        & (orders.status == "completed")
    ].groupby("customer_id").size()

    cohort["lapsed"] = (
        ~cohort["customer_id"].isin(september_completed.index)
    ).astype(int)

    # 4. Select features; remove identifiers and post-period information
    numeric = [
        "recency", "frequency", "monetary", "avg_discount",
        "cancellation_rate", "avg_delivery", "age", "tenure_days"
    ]
    categorical = ["favourite_category", "city", "gender"]
    features = numeric + categorical

    X = cohort.set_index("customer_id")[features].copy()
    y = cohort.set_index("customer_id")["lapsed"].copy()

    # 5. Split BEFORE fitting preprocessing or oversampling
    train_ids, test_ids = train_test_split(
        X.index.to_numpy(),
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train, X_test = X.loc[train_ids], X.loc[test_ids]
    y_train, y_test = y.loc[train_ids], y.loc[test_ids]

    # 6. Preprocessing is fitted inside each model pipeline
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        ))
    ])

    prep = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

    # 7. Compare original and oversampled models
    model_configs = [
        (
            "logistic_original",
            Pipeline([
                ("prep", prep),
                ("model", LogisticRegression(max_iter=1000))
            ]),
            {"model__C": [0.01, 0.1, 1, 10, 100]}
        ),
        (
            "forest_original",
            Pipeline([
                ("prep", prep),
                ("model", RandomForestClassifier(
                    n_estimators=100, random_state=42
                ))
            ]),
            {
                "model__max_depth": [3, 5, 8],
                "model__min_samples_leaf": [5, 10, 20]
            }
        ),
        (
            "logistic_oversampled",
            ImbPipeline([
                ("prep", prep),
                ("sampler", RandomOverSampler(random_state=42)),
                ("model", LogisticRegression(max_iter=1000))
            ]),
            {"model__C": [0.01, 0.1, 1, 10, 100]}
        ),
        (
            "forest_oversampled",
            ImbPipeline([
                ("prep", prep),
                ("sampler", RandomOverSampler(random_state=42)),
                ("model", RandomForestClassifier(
                    n_estimators=100, random_state=42
                ))
            ]),
            {
                "model__max_depth": [3, 5, 8],
                "model__min_samples_leaf": [5, 10, 20]
            }
        )
    ]

    results = {}
    comparison = []

    for name, pipeline, parameters in model_configs:
        print(f"\nTraining: {name}")

        search = GridSearchCV(
            pipeline,
            parameters,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )
        search.fit(X_train, y_train)

        predictions = search.predict(X_test)

        metrics = {
            "best_params": search.best_params_,
            "cv_f1": float(search.best_score_),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(
                y_test, predictions, zero_division=0
            )),
            "recall": float(recall_score(
                y_test, predictions, zero_division=0
            )),
            "f1": float(f1_score(
                y_test, predictions, zero_division=0
            )),
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            ).tolist()
        }

        results[name] = {
            "search": search,
            "metrics": metrics
        }

        comparison.append({
            "model": name,
            "cv_f1": metrics["cv_f1"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "best_params": str(metrics["best_params"])
        })

    # 8. Select the model using training cross-validation F1,
    # not the test-set results.
    selected_name = max(
        results,
        key=lambda name: results[name]["metrics"]["cv_f1"]
    )
    selected = results[selected_name]["search"]

    OUT.mkdir(exist_ok=True)

    joblib.dump(selected, OUT / "lapse_model.joblib")

    # Verify saved model gives identical test predictions
    reloaded = joblib.load(OUT / "lapse_model.joblib")
    assert np.array_equal(
        selected.predict(X_test),
        reloaded.predict(X_test)
    )

    # 9. Create the top-10 Premium retention list
    segments, _, _, _ = rfm_and_segments(k=4)
    premium_ids = set(
        segments.loc[segments.is_premium, "customer_id"]
    )

    probabilities = reloaded.predict_proba(X_test)[:, 1]

    retention = pd.DataFrame({
        "customer_id": X_test.index,
        "lapse_probability": probabilities,
        "actually_lapsed": y_test.loc[X_test.index].values
    })

    retention = (
        retention[retention.customer_id.isin(premium_ids)]
        .sort_values("lapse_probability", ascending=False)
        .head(10)
    )

    retention.to_csv(OUT / "retention_list.csv", index=False)

    # 10. Save comparison and a readable evaluation report
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(
        OUT / "lapse_model_comparison.csv", index=False
    )

    baseline_accuracy = accuracy_score(
        y_test, np.zeros(len(y_test), dtype=int)
    )

    report = {
        "cohort_rows": int(len(cohort)),
        "lapsed_customers": int(y.sum()),
        "lapse_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "baseline_accuracy": float(baseline_accuracy),
        "selected_model": selected_name,
        "selected_model_metrics": results[selected_name]["metrics"],
        "all_models": {
            name: value["metrics"]
            for name, value in results.items()
        }
    }

    with open(OUT / "lapse_model_report.json", "w") as file:
        json.dump(report, file, indent=2)

    print("\n========== LAPSE MODEL SUMMARY ==========")
    print("Cohort customers:", len(cohort))
    print("Lapsed customers:", int(y.sum()))
    print("Lapse rate:", round(y.mean() * 100, 2), "%")
    print("Baseline accuracy:", round(baseline_accuracy, 4))
    print("\nMODEL COMPARISON:")
    print(comparison_df.to_string(index=False))
    print("\nSELECTED MODEL:", selected_name)
    print("\nSELECTED MODEL CONFUSION MATRIX:")
    print(results[selected_name]["metrics"]["confusion_matrix"])
    print("\nRETENTION LIST:")
    print(retention.to_string(index=False))
    print("\nSaved: outputs/lapse_model.joblib")
    print("Saved: outputs/retention_list.csv")
    print("Saved: outputs/lapse_model_comparison.csv")
    print("Saved: outputs/lapse_model_report.json")

    assert len(cohort) == 1156, "Unexpected lapse cohort size"
    assert int(y.sum()) == 443, "Unexpected lapse count"
    assert len(retention) <= 10
    print("\nPART B LAPSE MODEL COMPLETED SUCCESSFULLY")

    return report