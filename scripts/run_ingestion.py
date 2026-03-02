from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql import functions as F
from pyspark import StorageLevel

RAW_CSV = "data/raw/HIGGS.csv"
PARQUET_OUT = "data/processed/higgs_parquet"
PARQUET_PART_OUT = "data/processed/higgs_parquet_partitioned"

def build_spark():
    spark = (
        SparkSession.builder
        .appName("HIGGS_Ingestion")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .getOrCreate()
    )
    return spark

def main():
    spark = build_spark()

    schema = StructType(
        [StructField("label", DoubleType(), True)] +
        [StructField(f"f{i}", DoubleType(), True) for i in range(1, 29)]
    )

    df = (
        spark.read
        .schema(schema)
        .option("sep", ",")
        .csv(RAW_CSV)
    )

    print("Schema:")
    df.printSchema()

    print("Row count:")
    print(df.count())

    print("Label distribution:")
    df.groupBy("label").count().show()

    print("Null counts per column:")
    null_counts = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns])
    null_counts.show(truncate=False)

    df_repart = df.repartition(200)
    df_repart.write.mode("overwrite").parquet(PARQUET_OUT)
    print(f"Wrote Parquet to: {PARQUET_OUT}")

    df.write.mode("overwrite").partitionBy("label").parquet(PARQUET_PART_OUT)
    print(f"Wrote Partitioned Parquet to: {PARQUET_PART_OUT}")

    df_part = spark.read.parquet(PARQUET_PART_OUT)
    df_part = df_part.persist(StorageLevel.MEMORY_AND_DISK)
    df_part.count()  # materialize cache
    print("Cached partitioned dataset (MEMORY_AND_DISK).")

    df_part.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()
