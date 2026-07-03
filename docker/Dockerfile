# Smart Agriculture Analytics - Docker Container
FROM apache/spark:3.4.1

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
USER root
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY --chown=spark:spark src/ ./src/
COPY --chown=spark:spark data/ ./data/
COPY --chown=spark:spark notebooks/ ./notebooks/

# Create output directory
RUN mkdir -p output && chown -R spark:spark /app

# Set environment variables
ENV SPARK_HOME=/opt/spark
ENV PYTHONPATH=/app
ENV MPLCONFIGDIR=/tmp/matplotlib

USER spark

# Default command - run the main solution
CMD ["/opt/spark/bin/spark-submit", "src/main_solution.py"]
