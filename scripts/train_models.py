from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
    LinearSVC
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
import os

TRAIN_PATH = "data/processed/train"
TEST_PATH = "data/processed/test"
MODELS_DIR = "models"

def build_spark():
    return (
        SparkSession.builder
        .appName("HIGGS_ModelTraining")
        .master("local[*]")
        # Fix macOS bind issue
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")

        # 8GB-friendly settings
        .config("spark.sql.shuffle.partitions", "80")
        .config("spark.default.parallelism", "80")

        # Keep driver memory realistic for 8GB machine
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "1g")

        # Reduce memory pressure in shuffles
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

        # Use disk for shuffle spill
        .config("spark.local.dir", "spark_tmp")
        .getOrCreate()
    )

def eval_binary(pred_df, label_col="label", score_col="rawPrediction"):
    auc_eval = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol=score_col, metricName="areaUnderROC"
    )
    auc = auc_eval.evaluate(pred_df)

    acc_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
    f1_eval  = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="f1")
    prec_eval= MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedPrecision")
    rec_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedRecall")

    return {
        "ROC_AUC": auc,
        "Accuracy": acc_eval.evaluate(pred_df),
        "F1": f1_eval.evaluate(pred_df),
        "Precision": prec_eval.evaluate(pred_df),
        "Recall": rec_eval.evaluate(pred_df),
    }

def confusion_matrix(pred_df):
    return pred_df.groupBy("label", "prediction").count().orderBy("label", "prediction")

def save_model(fitted_model, model_name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, model_name)
    fitted_model.write().overwrite().save(path)
    print(f"✅ Saved model -> {path}")

def main():
    spark = build_spark()

    # IMPORTANT: Do NOT cache or count full dataset on 8GB machine
    train = spark.read.parquet(TRAIN_PATH)
    test  = spark.read.parquet(TEST_PATH)

    feat_col = "scaledFeatures"

    # Train smaller set of models (GBT + LR). RF/SVM can be heavy.
    models = {
        "lr_model": LogisticRegression(
            featuresCol=feat_col, labelCol="label",
            maxIter=20, regParam=0.0, elasticNetParam=0.0
        ),
        "gbt_model": GBTClassifier(
            featuresCol=feat_col, labelCol="label",
            maxIter=30, maxDepth=5, seed=42
        ),
    }

    results = []

    for model_key, model in models.items():
        print("\n==============================")
        print(f"Training: {model_key}")
        print("==============================")

        fitted = model.fit(train)
        save_model(fitted, model_key)

        preds = fitted.transform(test)

        metrics = eval_binary(preds, score_col="rawPrediction")
        results.append((
            model_key,
            float(metrics["ROC_AUC"]),
            float(metrics["Accuracy"]),
            float(metrics["F1"]),
            float(metrics["Precision"]),
            float(metrics["Recall"])
        ))

        print("Metrics:", metrics)
        print("Confusion Matrix:")
        confusion_matrix(preds).show(10, truncate=False)

    print("\n===== Summary (Test Set) =====")
    summary_df = spark.createDataFrame(results, ["Model", "ROC_AUC", "Accuracy", "F1", "Precision", "Recall"])
    summary_df.orderBy(F.desc("ROC_AUC")).show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()