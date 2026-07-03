"""
================================================================================
SMART AGRICULTURE ANALYTICS AND CROP PREDICTION SYSTEM
Using Apache Spark
================================================================================
This is a beginner-friendly, complete solution for the case study.
Each section is clearly marked with the Question Number (Q1-Q7).

HOW TO RUN:
    1. Install requirements: pip install -r requirements.txt
    2. Run this script: python src/main_solution.py
    3. Or open in Jupyter Notebook: jupyter notebook notebooks/main_solution.ipynb

REQUIREMENTS:
    - Python 3.8+
    - Apache Spark 3.x
    - Java 8 or 11 (required for Spark)
"""

# =============================================================================
# STEP 0: IMPORT LIBRARIES
# =============================================================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("SMART AGRICULTURE ANALYTICS PLATFORM - APACHE SPARK")
print("=" * 80)

# =============================================================================
# Q1. SPARK INITIALIZATION AND DATA LOADING (2 Marks)
# =============================================================================
print("\n" + "=" * 80)
print("Q1. SPARK INITIALIZATION AND DATA LOADING")
print("=" * 80)

# Initialize Spark Session (this is the entry point for all Spark operations)
spark = SparkSession.builder \
    .appName("SmartAgricultureAnalytics") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Spark Context (older API, but sometimes needed for RDD operations)
sc = spark.sparkContext

print(f"Spark Version: {spark.version}")
print(f"App Name: {spark.sparkContext.appName}")
print(f"Master: {spark.sparkContext.master}")

# Load agricultural datasets
print("\nLoading datasets...")

weather_df = spark.read.csv("data/weather_data.csv", header=True, inferSchema=True)
soil_df = spark.read.csv("data/soil_data.csv", header=True, inferSchema=True)
crop_df = spark.read.csv("data/crop_data.csv", header=True, inferSchema=True)
iot_df = spark.read.csv("data/iot_sensor_data.csv", header=True, inferSchema=True)

print("\nWEATHER DATA - Schema:")
weather_df.printSchema()
print("\nSample Data (5 rows):")
weather_df.show(5, truncate=False)

print("\nSOIL DATA - Schema:")
soil_df.printSchema()
print("\nSample Data (5 rows):")
soil_df.show(5, truncate=False)

print("\nCROP DATA - Schema:")
crop_df.printSchema()
print("\nSample Data (5 rows):")
crop_df.show(5, truncate=False)

print("\nIOT SENSOR DATA - Schema:")
iot_df.printSchema()
print("\nSample Data (5 rows):")
iot_df.show(5, truncate=False)

# =============================================================================
# Q2. RDD IMPLEMENTATION (3 Marks)
# =============================================================================
print("\n" + "=" * 80)
print("Q2. RDD IMPLEMENTATION - Transformations & Actions")
print("=" * 80)

# Convert DataFrames to RDDs (Resilient Distributed Datasets)
weather_rdd = weather_df.rdd
soil_rdd = soil_df.rdd
crop_rdd = crop_df.rdd
iot_rdd = iot_df.rdd

print("\nTRANSFORMATION OPERATIONS:")

# 1. map() - Transform each element
print("\n1. map() - Convert temperature from Celsius to Fahrenheit:")
temp_f = weather_rdd.map(lambda row: (row.station_id, row.temperature * 9/5 + 32))
print("   Sample (first 5):")
for item in temp_f.take(5):
    print(f"   Station {item[0]}: {item[1]:.2f} F")

# 2. filter() - Keep only rows matching condition
print("\n2. filter() - Regions with high rainfall (>150mm):")
high_rain = weather_rdd.filter(lambda row: row.rainfall > 150)
print(f"   Count of high rainfall records: {high_rain.count()}")

# 3. flatMap() - One to many mapping
print("\n3. flatMap() - Split region names into characters:")
region_chars = weather_rdd.flatMap(lambda row: list(row.region))
print(f"   First 10 characters from region names: {region_chars.take(10)}")

# 4. union() - Combine RDDs
print("\n4. union() - Combine weather and IoT temperature data:")
weather_temp = weather_rdd.map(lambda row: ("weather", row.temperature))
iot_temp = iot_rdd.map(lambda row: ("iot", row.temperature))
combined_temp = weather_temp.union(iot_temp)
print(f"   Total temperature records combined: {combined_temp.count()}")

print("\nACTION OPERATIONS:")

# 1. collect() - Get all data to driver (USE WITH CAUTION on large data!)
print("\n1. collect() - Get all regions:")
regions = weather_rdd.map(lambda row: row.region).collect()
print(f"   All regions: {set(regions)}")

# 2. count() - Count elements
print(f"\n2. count() - Total weather records: {weather_rdd.count()}")

# 3. reduce() - Aggregate elements
print("\n3. reduce() - Sum of all rainfall:")
total_rainfall = weather_rdd.map(lambda row: row.rainfall).reduce(lambda a, b: a + b)
print(f"   Total rainfall across all stations: {total_rainfall:.2f} mm")

# 4. first() - Get first element
print(f"\n4. first() - First weather record: {weather_rdd.first()}")

# 5. take() - Get n elements
print("\n5. take() - Top 3 highest temperatures:")
highest_temps = weather_rdd.takeOrdered(3, key=lambda row: -row.temperature)
for row in highest_temps:
    print(f"   Station {row.station_id}: {row.temperature} C")

# 6. distinct() - Get unique values
print(f"\n6. distinct() - Unique soil types: {soil_rdd.map(lambda row: row.soil_type).distinct().collect()}")

# =============================================================================
# Q3. KEY-VALUE OPERATIONS AND PERSISTENCE (2 Marks)
# =============================================================================
print("\n" + "=" * 80)
print("Q3. KEY-VALUE PAIR OPERATIONS & PERSISTENCE")
print("=" * 80)

# Create Key-Value RDDs (region as key, data as value)
print("\nCreating Key-Value RDDs...")

# Weather: (region, temperature)
weather_kv = weather_rdd.map(lambda row: (row.region, row.temperature))
print("\nWeather Key-Value (Region, Temperature) - Sample:")
for item in weather_kv.take(5):
    print(f"   {item}")

# Crop: (region, yield)
crop_kv = crop_rdd.map(lambda row: (row.region, row.yield_tons))
print("\nCrop Key-Value (Region, Yield) - Sample:")
for item in crop_kv.take(5):
    print(f"   {item}")

print("\nSHUFFLE OPERATIONS:")

# 1. reduceByKey() - Group by key and reduce
print("\n1. reduceByKey() - Average temperature by region:")
region_temp = weather_kv.mapValues(lambda x: (x, 1)) \
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
    .mapValues(lambda x: x[0] / x[1])
print("   Average temperature by region:")
for region, temp in region_temp.collect():
    print(f"   {region}: {temp:.2f} C")

# 2. groupByKey() - Group all values by key
print("\n2. groupByKey() - All temperatures grouped by region:")
region_temps_grouped = weather_kv.groupByKey()
for region, temps in region_temps_grouped.collect():
    print(f"   {region}: {list(temps)[:3]}... (showing first 3)")

# 3. join() - Join two RDDs by key
print("\n3. join() - Join weather and crop data by region:")
crop_by_region = crop_kv.mapValues(lambda x: (x, 1)) \
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
    .mapValues(lambda x: x[0] / x[1])
joined = region_temp.join(crop_by_region)
print("   (Region, (Avg Temp, Avg Yield)):")
for item in joined.take(5):
    print(f"   {item[0]}: Temp={item[1][0]:.2f}C, Yield={item[1][1]:.2f} tons")

# 4. sortByKey() - Sort by key
print("\n4. sortByKey() - Regions sorted alphabetically:")
for item in region_temp.sortByKey().collect():
    print(f"   {item[0]}: {item[1]:.2f}C")

print("\nRDD PERSISTENCE TECHNIQUES:")

# Persist RDD in memory (useful when reused multiple times)
weather_kv.persist()
print("\n1. persist() - Weather KV RDD cached in memory")

# Check storage level
print(f"   Storage Level: {weather_kv.getStorageLevel()}")

# Unpersist when done
weather_kv.unpersist()
print("\n2. unpersist() - Removed from cache")

# Different persistence levels
from pyspark import StorageLevel

# MEMORY_ONLY (default)
weather_kv.persist(StorageLevel.MEMORY_ONLY)
print("\n3. MEMORY_ONLY - Stored in RAM only")
weather_kv.unpersist()

# MEMORY_AND_DISK - Spill to disk if memory full
weather_kv.persist(StorageLevel.MEMORY_AND_DISK)
print("\n4. MEMORY_AND_DISK - Stored in RAM + Disk if needed")
weather_kv.unpersist()

# DISK_ONLY - Store only on disk
weather_kv.persist(StorageLevel.DISK_ONLY)
print("\n5. DISK_ONLY - Stored on disk only")
weather_kv.unpersist()

print("\nRDD Persistence demonstrated successfully!")

# =============================================================================
# Q4. SPARK DATAFRAME OPERATIONS (3 Marks)
# =============================================================================
print("\n" + "=" * 80)
print("Q4. SPARK DATAFRAME OPERATIONS")
print("=" * 80)

print("\nSELECTION OPERATIONS:")

# 1. select() - Choose specific columns
print("\n1. select() - Select only station_id, region, and temperature:")
weather_selected = weather_df.select("station_id", "region", "temperature")
weather_selected.show(5)

# 2. selectExpr() - SQL-like expressions
print("\n2. selectExpr() - Convert temperature to Fahrenheit in query:")
weather_df.selectExpr("station_id", "region", "temperature * 9/5 + 32 as temp_fahrenheit").show(5)

print("\nFILTERING OPERATIONS:")

# 3. filter() / where() - Row filtering
print("\n3. filter() - High temperature regions (>35C):")
weather_df.filter(col("temperature") > 35).show(5)

print("\n4. Multiple conditions - North region with rainfall > 100:")
weather_df.filter((col("region") == "North") & (col("rainfall") > 100)).show(5)

print("\nGROUPING AND AGGREGATION:")

# 5. groupBy() with aggregations
print("\n5. groupBy() - Average rainfall by region:")
weather_df.groupBy("region").agg(
    avg("rainfall").alias("avg_rainfall"),
    max("temperature").alias("max_temp"),
    min("temperature").alias("min_temp"),
    count("*").alias("record_count")
).show()

print("\n6. Crop yield statistics by crop name:")
crop_df.groupBy("crop_name").agg(
    avg("yield_tons").alias("avg_yield"),
    sum("yield_tons").alias("total_yield"),
    max("yield_tons").alias("max_yield"),
    avg("area_hectares").alias("avg_area")
).orderBy(desc("avg_yield")).show()

print("\nJOIN OPERATIONS:")

# 7. Join weather and crop data on region
print("\n7. inner join() - Weather + Crop on region:")
joined_df = weather_df.join(crop_df, weather_df.region == crop_df.region, "inner")
joined_df.select("station_id", weather_df.region, "temperature", "rainfall", "crop_name", "yield_tons").show(10)

print("\n8. left join() - All weather records with matching crops:")
weather_df.join(crop_df, "region", "left").select("region", "temperature", "crop_name", "yield_tons").show(10)

print("\nADVANCED OPERATIONS:")

# 9. withColumn() - Add new column
print("\n9. withColumn() - Add yield_per_hectare:")
crop_with_efficiency = crop_df.withColumn("yield_per_hectare", round(col("yield_tons") / col("area_hectares"), 2))
crop_with_efficiency.select("crop_name", "yield_tons", "area_hectares", "yield_per_hectare").show(5)

# 10. drop() - Remove column
print("\n10. drop() - Remove irrigation_type column:")
crop_df.drop("irrigation_type").printSchema()

# 11. distinct() - Unique values
print(f"\n11. distinct() - Unique regions in crop data: {crop_df.select('region').distinct().count()}")

# 12. orderBy() / sort() - Sorting
print("\n12. orderBy() - Top 5 yields:")
crop_df.orderBy(desc("yield_tons")).show(5)

# 13. limit() - Limit rows
print("\n13. limit() - First 3 records:")
weather_df.limit(3).show()

# 14. describe() - Statistical summary
print("\n14. describe() - Weather statistics:")
weather_df.describe().show()

# =============================================================================
# Q5. EXPLORATORY DATA ANALYSIS AND SPARK SQL (5 Marks)
# =============================================================================
print("\n" + "=" * 80)
print("Q5. EXPLORATORY DATA ANALYSIS (EDA) AND SPARK SQL")
print("=" * 80)

# Register DataFrames as temporary views (like virtual tables) for SQL queries
weather_df.createOrReplaceTempView("weather")
soil_df.createOrReplaceTempView("soil")
crop_df.createOrReplaceTempView("crops")
iot_df.createOrReplaceTempView("iot_sensors")

print("\n1. CROP-WISE PRODUCTION TRENDS:")
print("-" * 50)
query1 = spark.sql("""
    SELECT 
        crop_name,
        year,
        season,
        ROUND(AVG(yield_tons), 2) as avg_yield,
        ROUND(SUM(yield_tons), 2) as total_yield,
        COUNT(*) as record_count
    FROM crops
    GROUP BY crop_name, year, season
    ORDER BY crop_name, year, season
""")
query1.show(20)

print("\n2. RAINFALL AND TEMPERATURE PATTERNS:")
print("-" * 50)
query2 = spark.sql("""
    SELECT 
        region,
        ROUND(AVG(rainfall), 2) as avg_rainfall,
        ROUND(AVG(temperature), 2) as avg_temperature,
        ROUND(MAX(temperature), 2) as max_temperature,
        ROUND(MIN(temperature), 2) as min_temperature,
        ROUND(AVG(humidity), 2) as avg_humidity
    FROM weather
    GROUP BY region
    ORDER BY avg_rainfall DESC
""")
query2.show()

print("\n3. REGION-WISE CROP YIELDS:")
print("-" * 50)
query3 = spark.sql("""
    SELECT 
        c.region,
        c.crop_name,
        ROUND(AVG(c.yield_tons), 2) as avg_yield,
        ROUND(SUM(c.yield_tons), 2) as total_yield,
        ROUND(AVG(c.area_hectares), 2) as avg_area
    FROM crops c
    GROUP BY c.region, c.crop_name
    ORDER BY c.region, avg_yield DESC
""")
query3.show(20)

print("\n4. TOP PERFORMING AGRICULTURAL REGIONS:")
print("-" * 50)
query4 = spark.sql("""
    SELECT 
        region,
        ROUND(SUM(yield_tons), 2) as total_yield,
        ROUND(AVG(yield_tons), 2) as avg_yield,
        COUNT(DISTINCT crop_name) as crop_variety,
        ROUND(SUM(area_hectares), 2) as total_area
    FROM crops
    GROUP BY region
    ORDER BY total_yield DESC
""")
query4.show()

print("\n5. SEASONAL CROP PRODUCTION:")
print("-" * 50)
query5 = spark.sql("""
    SELECT 
        season,
        crop_name,
        ROUND(AVG(yield_tons), 2) as avg_yield,
        ROUND(SUM(yield_tons), 2) as total_yield,
        COUNT(*) as record_count
    FROM crops
    GROUP BY season, crop_name
    ORDER BY season, total_yield DESC
""")
query5.show(20)

print("\n6. COMPREHENSIVE ANALYSIS - Weather impact on crops:")
print("-" * 50)
query6 = spark.sql("""
    SELECT 
        w.region,
        ROUND(AVG(w.rainfall), 2) as avg_rainfall,
        ROUND(AVG(w.temperature), 2) as avg_temp,
        ROUND(AVG(c.yield_tons), 2) as avg_crop_yield,
        COUNT(c.crop_id) as total_crops
    FROM weather w
    JOIN crops c ON w.region = c.region
    GROUP BY w.region
    ORDER BY avg_crop_yield DESC
""")
query6.show()

print("\n7. SOIL QUALITY ANALYSIS:")
print("-" * 50)
query7 = spark.sql("""
    SELECT 
        soil_type,
        ROUND(AVG(ph_level), 2) as avg_ph,
        ROUND(AVG(nitrogen), 2) as avg_nitrogen,
        ROUND(AVG(phosphorus), 2) as avg_phosphorus,
        ROUND(AVG(potassium), 2) as avg_potassium,
        COUNT(*) as sample_count
    FROM soil
    GROUP BY soil_type
    ORDER BY sample_count DESC
""")
query7.show()

# Save EDA outputs
print("\nSaving EDA results to output folder...")
query1.write.mode("overwrite").csv("output/eda_crop_trends", header=True)
query2.write.mode("overwrite").csv("output/eda_weather_patterns", header=True)
query3.write.mode("overwrite").csv("output/eda_region_yields", header=True)
query4.write.mode("overwrite").csv("output/eda_top_regions", header=True)
query5.write.mode("overwrite").csv("output/eda_seasonal_production", header=True)

print("EDA outputs saved to output/ folder")

# =============================================================================
# Q6. ETL PIPELINE DEVELOPMENT (3 Marks)
# =============================================================================
print("\n" + "=" * 80)
print("Q6. ETL PIPELINE DEVELOPMENT")
print("=" * 80)

print("\nETL PIPELINE: Extract -> Transform -> Load")
print("-" * 50)

# ETL Step 1: EXTRACT - Read raw data from multiple sources
print("\nSTEP 1: EXTRACT")
print("   Sources: IoT devices, Weather stations, Agricultural databases")

raw_iot = spark.read.csv("data/iot_sensor_data.csv", header=True, inferSchema=True)
raw_weather = spark.read.csv("data/weather_data.csv", header=True, inferSchema=True)
raw_soil = spark.read.csv("data/soil_data.csv", header=True, inferSchema=True)
raw_crops = spark.read.csv("data/crop_data.csv", header=True, inferSchema=True)

print(f"   - IoT records extracted: {raw_iot.count()}")
print(f"   - Weather records extracted: {raw_weather.count()}")
print(f"   - Soil records extracted: {raw_soil.count()}")
print(f"   - Crop records extracted: {raw_crops.count()}")

# ETL Step 2: TRANSFORM - Clean, enrich, and aggregate data
print("\nSTEP 2: TRANSFORM")

# 2a. Data Cleaning - Handle nulls, standardize formats
print("   2a. Cleaning IoT data...")
clean_iot = raw_iot \
    .dropna(subset=["soil_moisture", "temperature"]) \
    .withColumn("timestamp", to_timestamp(col("timestamp"))) \
    .withColumn("date", to_date(col("timestamp")))

# 2b. Standardize region names (upper case, trim)
print("   2b. Standardizing region names...")
clean_weather = raw_weather.withColumn("region", upper(trim(col("region"))))
clean_soil = raw_soil.withColumn("region", upper(trim(col("region"))))
clean_crops = raw_crops.withColumn("region", upper(trim(col("region"))))

# 2c. Enrich data - Add calculated columns
print("   2c. Enriching crop data with efficiency metrics...")
enriched_crops = clean_crops \
    .withColumn("yield_per_hectare", round(col("yield_tons") / col("area_hectares"), 2)) \
    .withColumn("fertilizer_per_hectare", round(col("fertilizer_used") / col("area_hectares"), 2))

# 2d. Aggregate IoT readings by region and date
print("   2d. Aggregating IoT sensor readings...")
agg_iot = clean_iot.groupBy("region", "date").agg(
    avg("soil_moisture").alias("avg_soil_moisture"),
    avg("temperature").alias("avg_sensor_temp"),
    avg("light_intensity").alias("avg_light"),
    min("battery_level").alias("min_battery")
)

# 2e. Join all datasets into a centralized fact table
print("   2e. Joining all datasets into centralized warehouse table...")

# Join crop with weather on region
etl_step1 = enriched_crops.join(
    clean_weather.select("region", "temperature", "rainfall", "humidity"),
    "region",
    "left"
)

# Join with soil data
etl_step2 = etl_step1.join(
    clean_soil.select("region", "ph_level", "soil_type", "nitrogen", "phosphorus", "potassium"),
    "region",
    "left"
)

# Final centralized data
print("\nTransformed Data Schema:")
etl_step2.printSchema()

# ETL Step 3: LOAD - Save to centralized storage
print("\nSTEP 3: LOAD")

# Load to Parquet format (optimized for analytics)
print("   Loading to Parquet format (optimized)...")
etl_step2.write.mode("overwrite").parquet("output/centralized_agriculture_data")

# Also save as CSV for compatibility
print("   Loading to CSV format...")
etl_step2.write.mode("overwrite").option("header", "true").csv("output/centralized_agriculture_csv")

# Save aggregated IoT data
agg_iot.write.mode("overwrite").parquet("output/aggregated_iot_data")

print("\nETL Pipeline completed successfully!")
print("   Output locations:")
print("   - output/centralized_agriculture_data (Parquet)")
print("   - output/centralized_agriculture_csv (CSV)")
print("   - output/aggregated_iot_data (Parquet)")

# =============================================================================
# Q7. MACHINE LEARNING - CROP YIELD PREDICTION (2 Marks)
# =============================================================================
print("\n" + "=" * 80)
print("Q7. MACHINE LEARNING - CROP YIELD PREDICTION")
print("=" * 80)

print("\nBuilding ML Pipeline to Predict Crop Yield...")

# Prepare ML dataset from the joined ETL data
ml_data = etl_step2.select(
    "crop_name",
    "region",
    "season",
    "area_hectares",
    "fertilizer_used",
    "temperature",
    "rainfall",
    "humidity",
    "ph_level",
    "nitrogen",
    "phosphorus",
    "potassium",
    "yield_tons"
).dropna()

print(f"\nML Dataset: {ml_data.count()} records after removing nulls")

# Step 1: Convert categorical variables to numeric using StringIndexer
print("\nStep 1: Encoding categorical variables...")
crop_indexer = StringIndexer(inputCol="crop_name", outputCol="crop_index", handleInvalid="keep")
region_indexer = StringIndexer(inputCol="region", outputCol="region_index", handleInvalid="keep")
season_indexer = StringIndexer(inputCol="season", outputCol="season_index", handleInvalid="keep")

# Step 2: One-hot encode the indexed categories
print("   Step 2: One-hot encoding...")
crop_encoder = OneHotEncoder(inputCol="crop_index", outputCol="crop_encoded")
region_encoder = OneHotEncoder(inputCol="region_index", outputCol="region_encoded")
season_encoder = OneHotEncoder(inputCol="season_index", outputCol="season_encoded")

# Step 3: Assemble features into a single vector
print("   Step 3: Assembling feature vector...")
feature_cols = [
    "crop_encoded", "region_encoded", "season_encoded",
    "area_hectares", "fertilizer_used", "temperature",
    "rainfall", "humidity", "ph_level",
    "nitrogen", "phosphorus", "potassium"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

# Step 4: Linear Regression Model for Yield Prediction
print("   Step 4: Creating Linear Regression model...")
lr = LinearRegression(
    featuresCol="features",
    labelCol="yield_tons",
    predictionCol="predicted_yield",
    maxIter=50,
    regParam=0.1
)

# Step 5: Build Pipeline
print("\nBuilding ML Pipeline...")
pipeline = Pipeline(stages=[
    crop_indexer, region_indexer, season_indexer,
    crop_encoder, region_encoder, season_encoder,
    assembler, lr
])

# Step 6: Split data into training and testing sets
print("\nSplitting data (80% train, 20% test)...")
train_data, test_data = ml_data.randomSplit([0.8, 0.2], seed=42)
print(f"   Training records: {train_data.count()}")
print(f"   Testing records: {test_data.count()}")

# Step 7: Train the model
print("\nTraining the model...")
model = pipeline.fit(train_data)

# Step 8: Make predictions
print("\nMaking predictions...")
predictions = model.transform(test_data)

print("\nSample Predictions:")
predictions.select(
    "crop_name", "region", "season", "area_hectares",
    "yield_tons", "predicted_yield"
).show(10)

# Step 9: Evaluate the model
print("\nMODEL EVALUATION:")
print("-" * 50)

# Regression Evaluator
evaluator_rmse = RegressionEvaluator(
    labelCol="yield_tons",
    predictionCol="predicted_yield",
    metricName="rmse"
)
evaluator_mae = RegressionEvaluator(
    labelCol="yield_tons",
    predictionCol="predicted_yield",
    metricName="mae"
)
evaluator_r2 = RegressionEvaluator(
    labelCol="yield_tons",
    predictionCol="predicted_yield",
    metricName="r2"
)

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print(f"   RMSE (Root Mean Square Error): {rmse:.4f}")
print(f"   MAE (Mean Absolute Error): {mae:.4f}")
print(f"   R2 (Coefficient of Determination): {r2:.4f}")

# Save model evaluation report
print("\nSaving model evaluation report...")
eval_report = f"""
================================================================================
MODEL EVALUATION REPORT - Crop Yield Prediction
================================================================================
Model Type: Linear Regression (Spark MLlib)
Training Records: {train_data.count()}
Testing Records: {test_data.count()}

METRICS:
--------
RMSE (Root Mean Square Error): {rmse:.4f}
MAE (Mean Absolute Error): {mae:.4f}
R2 (Coefficient of Determination): {r2:.4f}

INTERPRETATION:
---------------
- RMSE: Average prediction error is {rmse:.2f} tons
- MAE: Average absolute error is {mae:.2f} tons
- R2: Model explains {r2*100:.2f}% of yield variance

FEATURES USED:
--------------
{', '.join(feature_cols)}

CONCLUSION:
-----------
The Linear Regression model provides a baseline for crop yield prediction.
For production use, consider:
1. Random Forest Regressor
2. Gradient Boosted Trees
3. Deep Learning (Neural Networks)
4. Feature engineering (interaction terms, polynomial features)

================================================================================
"""

with open("output/model_evaluation_report.txt", "w") as f:
    f.write(eval_report)

print("Model evaluation report saved to: output/model_evaluation_report.txt")

# Save the trained model
print("\nSaving trained model...")
model.write().overwrite().save("output/crop_yield_model")
print("Model saved to: output/crop_yield_model")

# =============================================================================
# VISUALIZATION (Bonus - Using Pandas + Matplotlib)
# =============================================================================
print("\n" + "=" * 80)
print("BONUS: DATA VISUALIZATION")
print("=" * 80)

print("\nGenerating visualization charts...")

# Convert to Pandas for visualization
weather_pd = weather_df.toPandas()
crop_pd = crop_df.toPandas()

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Smart Agriculture Analytics Dashboard', fontsize=16, fontweight='bold')

# Chart 1: Average yield by crop
ax1 = axes[0, 0]
crop_yield = crop_pd.groupby('crop_name')['yield_tons'].mean().sort_values(ascending=True)
crop_yield.plot(kind='barh', ax=ax1, color='green', alpha=0.7)
ax1.set_title('Average Yield by Crop Type')
ax1.set_xlabel('Yield (tons)')

# Chart 2: Temperature by region
ax2 = axes[0, 1]
weather_pd.boxplot(column='temperature', by='region', ax=ax2)
ax2.set_title('Temperature Distribution by Region')
ax2.set_xlabel('Region')
ax2.set_ylabel('Temperature (C)')

# Chart 3: Rainfall vs Temperature scatter
ax3 = axes[1, 0]
ax3.scatter(weather_pd['temperature'], weather_pd['rainfall'], c='blue', alpha=0.6)
ax3.set_title('Rainfall vs Temperature')
ax3.set_xlabel('Temperature (C)')
ax3.set_ylabel('Rainfall (mm)')

# Chart 4: Crop yield by season
ax4 = axes[1, 1]
season_yield = crop_pd.groupby('season')['yield_tons'].sum()
ax4.pie(season_yield, labels=season_yield.index, autopct='%1.1f%%', startangle=90)
ax4.set_title('Total Yield Distribution by Season')

plt.tight_layout()
plt.savefig('output/agriculture_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

print("Dashboard saved to: output/agriculture_dashboard.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("""
SUMMARY OF COMPLETED QUESTIONS:
-------------------------------
Q1. Spark Initialization & Data Loading
Q2. RDD Implementation (Transformations & Actions)
Q3. Key-Value Operations & Persistence
Q4. Spark DataFrame Operations (Select, Filter, Join, Aggregate)
Q5. EDA & Spark SQL (Crop trends, Weather patterns, Regional yields)
Q6. ETL Pipeline (Extract, Transform, Load)
Q7. Machine Learning (Crop Yield Prediction with Evaluation)

OUTPUT FILES GENERATED:
-----------------------
output/eda_crop_trends/           - Crop-wise production trends
output/eda_weather_patterns/        - Rainfall & temperature analysis
output/eda_region_yields/           - Region-wise crop yields
output/eda_top_regions/             - Top performing regions
output/eda_seasonal_production/      - Seasonal production data
output/centralized_agriculture_data/ - ETL output (Parquet)
output/centralized_agriculture_csv/ - ETL output (CSV)
output/aggregated_iot_data/        - Aggregated IoT data
output/crop_yield_model/            - Saved ML model
output/model_evaluation_report.txt  - Model evaluation metrics
output/agriculture_dashboard.png    - Visualization dashboard
""")

# Stop Spark Session
spark.stop()
print("\nSpark Session stopped. Goodbye!")
