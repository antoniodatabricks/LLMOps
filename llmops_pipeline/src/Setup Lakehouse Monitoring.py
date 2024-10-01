# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk mlflow textstat tiktoken evaluate

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

# Inference table details

inference_processed_table = dbutils.widgets.get("inference_processed_table")

# Lakehouse monitoring info

lakehouse_monitoring_schema = dbutils.widgets.get("lakehouse_monitoring_schema")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Monitoring Table
# MAGIC - Here we are just creating an empty monitoring table. This table will be populated by the script "Compute Inference Table Metrics"

# COMMAND ----------

spark.sql(f"""
          
          CREATE TABLE IF NOT EXISTS {inference_processed_table} (
          execution_time_ms BIGINT,
          input STRING,
          output STRING,
          timestamp DOUBLE,
          model_id STRING,
          `toxicity(input)` DOUBLE,
          `perplexity(input)` DOUBLE,
          `token_count(input)` INT,
          `flesch_kincaid_grade(input)` DOUBLE,
          `automated_readability_index(input)` DOUBLE,
          `toxicity(output)` DOUBLE,
          `perplexity(output)` DOUBLE,
          `token_count(output)` INT,
          `flesch_kincaid_grade(output)` DOUBLE,
          `automated_readability_index(output)` DOUBLE)
           USING delta
           TBLPROPERTIES (
           'delta.columnMapping.mode' = 'name',
           'delta.enableChangeDataFeed' = 'true',
           'delta.enableDeletionVectors' = 'true',
           'delta.feature.changeDataFeed' = 'supported',
           'delta.feature.columnMapping' = 'supported',
           'delta.feature.deletionVectors' = 'supported',
           'delta.minReaderVersion' = '3',
           'delta.minWriterVersion' = '7')
          
          """)

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup Lakehouse monitoring for the inference table metrics

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorTimeSeries, MonitorInfoStatus
import os
import time

# COMMAND ----------

# Create monitor using databricks-sdk's `quality_monitors` client
w = WorkspaceClient()

monitor_exists = True

try:
    w.quality_monitors.get(table_name=inference_processed_table)
except:
    monitor_exists = False

# COMMAND ----------

if not monitor_exists:
  
  lhm_monitor = w.quality_monitors.create(
      table_name=inference_processed_table, # Always use 3-level namespace
      time_series = MonitorTimeSeries(
        timestamp_col = "timestamp",
        granularities = ["5 minutes"],
     ),
      assets_dir = os.getcwd(),
      slicing_exprs = ["model_id"],
      output_schema_name=f"{lakehouse_monitoring_schema}"
    )

  # Wait for monitor to be created
  while lhm_monitor.status ==  MonitorInfoStatus.MONITOR_STATUS_PENDING:
    lhm_monitor = w.quality_monitors.get(table_name=inference_processed_table)
    time.sleep(10)

  assert lhm_monitor.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE, "Error creating monitor"
