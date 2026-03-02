import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.functions import vector_to_array

TEST_PATH = "data/processed/test"
MODEL_PATH = "models/gbt_model"
OUT_DIR = "tableau"
OUT_FILE = "higgs_tableau_export.csv"

def build_spark():
    return (
        SparkSession.builder
        .appName("HIGGS_Tableau_Export")
        .master("local[*]")
        # macOS bind fix
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        # 8GB friendly
        .config("spark.sql.shuffle.partitions", "80")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )

def main():
    spark = build_spark()

    print("Loading test dataset...")
    test = spark.read.parquet(TEST_PATH)

    print("Loading trained GBT model...")
    model = GBTClassificationModel.load(MODEL_PATH)

    print("Generating predictions...")
    preds = model.transform(test)

    # Convert probability vector -> array so we can access index 1
    preds = preds.withColumn("prob_arr", vector_to_array(F.col("probability")))
    preds = preds.withColumn("prob_class1", F.col("prob_arr")[1])

    # Keep columns useful for Tableau
    feature_cols = [c for c in preds.columns if c.startswith("f")]
    out_cols = ["label"] + feature_cols + ["prediction", "prob_class1"]

    export_df = preds.select(*out_cols)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_FILE)

    print(f"Writing CSV for Tableau -> {out_path}")
    # single CSV file (Tableau-friendly). This can take a bit on 2.2M rows.
    export_df.coalesce(1).write.mode("overwrite").option("header", True).csv(OUT_DIR)

    print("✅ Export complete!")
    print("NOTE: Spark writes CSV as a folder. Open the part-*.csv inside the tableau/ folder.")

    spark.stop()

if __name__ == "__main__":
    main()