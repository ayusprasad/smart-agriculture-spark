# Smart Agriculture Analytics - Project Documentation

## 1. Project Overview

This project implements a **Smart Agriculture Analytics and Crop Prediction System** using Apache Spark. It processes agricultural data from multiple sources (weather stations, IoT sensors, soil labs, and crop records) to provide insights and predictions.

## 2. Architecture

```
Data Sources:
  - Weather Stations (temperature, rainfall, humidity)
  - IoT Sensors (soil moisture, light, battery)
  - Soil Labs (pH, NPK levels, soil type)
  - Crop Records (yield, area, season, fertilizer)

Processing Layer (Apache Spark):
  - RDD Operations (Q2, Q3)
  - DataFrame Operations (Q4)
  - Spark SQL (Q5)
  - ETL Pipeline (Q6)
  - ML Pipeline (Q7)

Storage:
  - Raw Data: CSV files
  - Processed: Parquet format
  - Output: CSV/Parquet + Visualizations

Deployment:
  - Docker Containerization
  - Kubernetes Orchestration
  - CI/CD via GitHub Actions
```

## 3. Question-by-Question Explanation

### Q1: Spark Initialization & Data Loading (2 Marks)
- **SparkSession**: Entry point for Spark operations
- **Data Loading**: `spark.read.csv()` with `header=True` and `inferSchema=True`
- **Schema Display**: `printSchema()` shows column names and types
- **Sample Data**: `show(n)` displays first n rows

### Q2: RDD Implementation (3 Marks)
**Transformations** (lazy operations that create new RDDs):
| Operation | Description | Example |
|-----------|-------------|---------|
| `map()` | Transform each element | Convert Celsius to Fahrenheit |
| `filter()` | Keep matching elements | Rainfall > 150mm |
| `flatMap()` | One-to-many mapping | Split strings to characters |
| `union()` | Combine two RDDs | Merge weather and IoT data |

**Actions** (trigger computation):
| Operation | Description |
|-----------|-------------|
| `collect()` | Get all data to driver |
| `count()` | Count elements |
| `reduce()` | Aggregate with function |
| `first()` | Get first element |
| `take(n)` | Get n elements |
| `distinct()` | Get unique values |

### Q3: Key-Value Operations & Persistence (2 Marks)
**Key-Value RDDs**: `rdd.map(lambda x: (key, value))`

**Shuffle Operations**:
| Operation | Description |
|-----------|-------------|
| `reduceByKey()` | Aggregate values by key |
| `groupByKey()` | Group all values by key |
| `join()` | Join two KV RDDs by key |
| `sortByKey()` | Sort by key |

**Persistence Levels**:
| Level | Description |
|-------|-------------|
| `MEMORY_ONLY` | Store in RAM only |
| `MEMORY_AND_DISK` | RAM + spill to disk |
| `DISK_ONLY` | Store on disk only |

Use `persist()` to cache and `unpersist()` to release.

### Q4: Spark DataFrame Operations (3 Marks)
| Operation | Purpose | Example |
|-----------|---------|---------|
| `select()` | Choose columns | `df.select("col1", "col2")` |
| `filter()` | Row filtering | `df.filter(col("age") > 18)` |
| `groupBy().agg()` | Aggregation | `df.groupBy("city").agg(avg("salary"))` |
| `join()` | Combine DataFrames | `df1.join(df2, "key")` |
| `withColumn()` | Add column | `df.withColumn("new", col("a")+col("b"))` |
| `orderBy()` | Sorting | `df.orderBy(desc("score"))` |
| `describe()` | Statistics | `df.describe()` |

### Q5: EDA & Spark SQL (5 Marks)
1. **Create Temp View**: `df.createOrReplaceTempView("table_name")`
2. **Run SQL**: `spark.sql("SELECT * FROM table_name")`
3. **Queries Implemented**:
   - Crop-wise production trends
   - Rainfall & temperature patterns
   - Region-wise crop yields
   - Top performing regions
   - Seasonal crop production

### Q6: ETL Pipeline (3 Marks)
```
EXTRACT     -> Read from CSV files (IoT, Weather, Soil, Crops)
TRANSFORM   -> Clean nulls, standardize, enrich, aggregate
LOAD        -> Save to Parquet (optimized) and CSV (compatible)
```

### Q7: Machine Learning (2 Marks)
**Pipeline Steps**:
1. `StringIndexer` - Convert text to numbers
2. `OneHotEncoder` - Encode categories as vectors
3. `VectorAssembler` - Combine features into one vector
4. `LinearRegression` - Train prediction model
5. `RegressionEvaluator` - Calculate RMSE, MAE, R²

## 4. DevOps Components

### Docker
- Base image: `apache/spark:3.4.1`
- Includes all dependencies
- Run with: `docker build -t smart-agriculture . && docker run smart-agriculture`

### Kubernetes
- **Deployment**: 2 replicas with resource limits
- **Service**: LoadBalancer exposing port 8080
- **PersistentVolumeClaim**: 10GB for data storage

### CI/CD (GitHub Actions)
1. Checkout code
2. Set up Java 11 + Python 3.9
3. Install dependencies
4. Run tests
5. Build Docker image
6. Push to Docker Hub
7. Deploy to Kubernetes

## 5. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the solution
python src/main_solution.py

# OR use Jupyter
jupyter notebook notebooks/main_solution.ipynb

# OR use Spark Submit
spark-submit src/main_solution.py
```

## 6. Output Files

| File/Folder | Description |
|-------------|-------------|
| `output/eda_crop_trends/` | Crop production analysis |
| `output/eda_weather_patterns/` | Weather pattern analysis |
| `output/eda_region_yields/` | Regional yield analysis |
| `output/eda_top_regions/` | Top region rankings |
| `output/eda_seasonal_production/` | Seasonal analysis |
| `output/centralized_agriculture_data/` | ETL output (Parquet) |
| `output/centralized_agriculture_csv/` | ETL output (CSV) |
| `output/aggregated_iot_data/` | IoT aggregation |
| `output/crop_yield_model/` | Saved ML model |
| `output/model_evaluation_report.txt` | Evaluation metrics |
| `output/agriculture_dashboard.png` | Visual dashboard |

## 7. Troubleshooting

| Issue | Solution |
|-------|----------|
| `Java not found` | Install Java 8 or 11, set JAVA_HOME |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `OutOfMemoryError` | Increase Spark memory: `.config("spark.driver.memory", "4g")` |
| `Port already in use` | Change port in Spark config |

## 8. Future Enhancements

- Implement Random Forest / Gradient Boosting models
- Add real-time streaming with Spark Streaming
- Integrate satellite imagery analysis
- Build interactive dashboard with Streamlit
- Add anomaly detection for crop diseases
