# Smart Agriculture Analytics - Docker Container
FROM bitnami/spark:3.4.1

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY src/ ./src/
COPY data/ ./data/
COPY notebooks/ ./notebooks/

# Create output directory
RUN mkdir -p output

# Set environment variables
ENV SPARK_HOME=/opt/bitnami/spark
ENV PYTHONPATH=/app

# Default command - run the main solution
CMD ["spark-submit", "src/main_solution.py"]
