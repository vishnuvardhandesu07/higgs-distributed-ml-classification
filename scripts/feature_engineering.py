from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import functions as F

IN_PATH = "data/processed/higgs_parquet_partitioned"
OUT_FE_PATH = "data/processed/higgs_features"
OUT_TRAIN = "data/processed/train"
OUT_TEST = "data/processed/test"

def build_spark():
    return (
        SparkSession.builder
        .appName("HIGGS_FeatureEngineering")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .getOrCreate()
    )

def main():
    spark = build_spark()

    df = spark.read.parquet(IN_PATH)

    # Ensure label is double and clean
    df = df.withColumn("label", F.col("label").cast("double"))

    feature_cols = [f"f{i}" for i in range(1, 29)]

    # 1) Assemble features vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_vec = assembler.transform(df).select("label", "features")

    # 2) Scale features (recommended for LR/SVM)
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
    scaler_model = scaler.fit(df_vec)
    df_scaled = scaler_model.transform(df_vec).select("label", "scaledFeatures")

    # Save full feature dataset
    df_scaled.write.mode("overwrite").parquet(OUT_FE_PATH)
    print(f"Saved scaled feature dataset to: {OUT_FE_PATH}")

    # 3) Train/Test split (reproducible)
    train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)

    train_df.write.mode("overwrite").parquet(OUT_TRAIN)
    test_df.write.mode("overwrite").parquet(OUT_TEST)
    print(f"Saved train to: {OUT_TRAIN}")
    print(f"Saved test to: {OUT_TEST}")

    print("Train/Test counts:")
    print("Train:", train_df.count())
    print("Test :", test_df.count())

    spark.stop()

if __name__ == "__main__":
    main()
