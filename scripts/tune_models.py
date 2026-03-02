# scripts/tune_models.py
# 8GB-safe hyperparameter tuning for HIGGS (Spark MLlib)
# - Tunes RandomForest on 5% sample
# - Tunes GBT on 2% sample (faster)
# - Uses small grids + low parallelism to avoid OOM on 8GB laptops

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

TRAIN_PATH = "data/processed/train"
TEST_PATH = "data/processed/test"

# Sampling for tuning (resource-aware)
RF_SAMPLE = 0.05      # ~440k rows from train
GBT_SAMPLE = 0.02     # ~176k rows from train (faster)

def build_spark():
    return (
        SparkSession.builder
        .appName("HIGGS_Hyperparam_Tuning_8GB")
        # 8GB-safe settings
        .config("spark.driver.memory", "3g")
        .config("spark.executor.memory", "3g")
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.executor.memoryOverhead", "512m")
        .config("spark.driver.memoryOverhead", "512m")
        # reduce shuffle pressure
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )

def main():
    spark = build_spark()

    # Load full datasets (do NOT cache full datasets on 8GB)
    train_full = spark.read.parquet(TRAIN_PATH)
    test_full = spark.read.parquet(TEST_PATH)

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    # =========================
    # 1) RandomForest Tuning
    # =========================
    train_rf = train_full.sample(withReplacement=False, fraction=RF_SAMPLE, seed=42).repartition(64).persist()
    test_rf = test_full.sample(withReplacement=False, fraction=RF_SAMPLE, seed=42).repartition(64).persist()

    print("\n=== RF Tuning Sample Sizes ===")
    print("Train RF sample:", train_rf.count())
    print("Test RF sample :", test_rf.count())

    rf = RandomForestClassifier(
        featuresCol="scaledFeatures",
        labelCol="label",
        seed=42
    )

    # Small, valid grid for 8GB
    rf_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [20, 40])
        .addGrid(rf.maxDepth, [6, 10])
        .build()
    )

    rf_cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=rf_grid,
        evaluator=evaluator,
        numFolds=3,        # solid methodology
        parallelism=1,     # keep low for 8GB
        seed=42
    )

    print("\n=== Tuning RandomForest (3-fold CV) ===")
    rf_cv_model = rf_cv.fit(train_rf)
    rf_best = rf_cv_model.bestModel

    rf_pred = rf_best.transform(test_rf)
    rf_auc = evaluator.evaluate(rf_pred)

    print("\nRF Best Params:")
    print(rf_best.extractParamMap())
    print("RF ROC-AUC (sampled test):", rf_auc)

    # Free RF sample memory before GBT tuning
    train_rf.unpersist()
    test_rf.unpersist()

    # =========================
    # 2) GBT Tuning (faster)
    # =========================
    train_gbt = train_full.sample(withReplacement=False, fraction=GBT_SAMPLE, seed=42).repartition(64).persist()
    test_gbt = test_full.sample(withReplacement=False, fraction=GBT_SAMPLE, seed=42).repartition(64).persist()

    print("\n=== GBT Tuning Sample Sizes ===")
    print("Train GBT sample:", train_gbt.count())
    print("Test GBT sample :", test_gbt.count())

    gbt = GBTClassifier(
        featuresCol="scaledFeatures",
        labelCol="label",
        seed=42
    )

    # Very small grid for 8GB: only 2 combinations total
    gbt_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [3, 5])
        .addGrid(gbt.maxIter, [15])  # fixed to reduce runtime
        .build()
    )

    gbt_cv = CrossValidator(
        estimator=gbt,
        estimatorParamMaps=gbt_grid,
        evaluator=evaluator,
        numFolds=2,        # faster, still acceptable under constraints
        parallelism=1,
        seed=42
    )

    print("\n=== Tuning GBT (2-fold CV, small grid) ===")
    gbt_cv_model = gbt_cv.fit(train_gbt)
    gbt_best = gbt_cv_model.bestModel

    gbt_pred = gbt_best.transform(test_gbt)
    gbt_auc = evaluator.evaluate(gbt_pred)

    print("\nGBT Best Params:")
    print(gbt_best.extractParamMap())
    print("GBT ROC-AUC (sampled test):", gbt_auc)

    train_gbt.unpersist()
    test_gbt.unpersist()

    spark.stop()

if __name__ == "__main__":
    main()
