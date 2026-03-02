# scripts/sklearn_baseline.py
# Single-node baseline using scikit-learn (8GB-safe, robust)
# Uses SGDClassifier (logistic regression via SGD) + custom AUC (avoids sklearn roc_auc crash)

import time
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

TRAIN_PATH = "data/processed/train"
TEST_PATH  = "data/processed/test"

SAMPLE_FRACTION = 0.02
RANDOM_SEED = 42


def build_spark():
    return (
        SparkSession.builder
        .appName("HIGGS_Sklearn_Baseline_SGD")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


@F.udf(returnType=ArrayType(DoubleType()))
def vec_to_list(v):
    if v is None:
        return None
    return v.toArray().tolist()


def to_numpy_xy(sdf):
    sdf2 = sdf.select(
        F.col("label").cast("int").alias("label"),
        vec_to_list(F.col("scaledFeatures")).alias("x")
    )
    pdf = sdf2.toPandas()

    X = np.asarray(pdf["x"].tolist(), dtype=np.float32)
    y = np.asarray(pdf["label"].values, dtype=np.int32)

    # Clean + clip to prevent numeric explosion
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -10.0, 10.0)

    return X, y


def auc_rank(y_true, y_score):
    """
    Robust ROC-AUC using rank statistic (Mann–Whitney U).
    Works even if sklearn's roc_auc_score crashes on this environment.
    Handles ties by average ranks.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    # Make scores finite
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)

    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC is undefined: need both positive and negative samples.")

    # Compute ranks with tie handling (average ranks)
    order = np.argsort(y_score, kind="mergesort")
    scores_sorted = y_score[order]
    y_sorted = y_true[order]

    ranks = np.empty_like(scores_sorted, dtype=np.float64)
    i = 0
    n = scores_sorted.size
    while i < n:
        j = i
        while j + 1 < n and scores_sorted[j + 1] == scores_sorted[i]:
            j += 1
        # average rank for ties; ranks are 1..n
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        ranks[i:j + 1] = avg_rank
        i = j + 1

    sum_ranks_pos = ranks[y_sorted == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def main():
    spark = build_spark()

    print("\nLoading sampled data from Spark Parquet...")
    train_sdf = spark.read.parquet(TRAIN_PATH).sample(False, SAMPLE_FRACTION, seed=RANDOM_SEED)
    test_sdf  = spark.read.parquet(TEST_PATH).sample(False, SAMPLE_FRACTION, seed=RANDOM_SEED)

    print("Train sample count:", train_sdf.count())
    print("Test sample count :", test_sdf.count())

    X_train, y_train = to_numpy_xy(train_sdf)
    X_test, y_test = to_numpy_xy(test_sdf)

    spark.stop()

    print("\nTraining scikit-learn SGDClassifier (logistic regression baseline)...")
    start = time.time()

    clf = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("sgd", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-3,
            random_state=RANDOM_SEED
        ))
    ])

    clf.fit(X_train, y_train)

    end = time.time()

    y_pred = clf.predict(X_test)
    y_score = clf.decision_function(X_test)

    # Make score finite before AUC
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)

    roc = auc_rank(y_test, y_score)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n===== Sklearn Baseline Results (sampled) =====")
    print("ROC-AUC :", roc)
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Training time (seconds):", round(end - start, 2))


if __name__ == "__main__":
    main()
