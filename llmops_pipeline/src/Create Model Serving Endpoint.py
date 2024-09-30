# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./helper

# COMMAND ----------

# Endpoint details
model_name = dbutils.widgets.get("model_name")
model_version = get_latest_model_version(model_name)
endpoint_name = dbutils.widgets.get("endpoint_name")
endpoint_host = dbutils.widgets.get("endpoint_host")
endpoint_token_scope = dbutils.widgets.get("endpoint_token_scope")
endpoint_token_secret = dbutils.widgets.get("endpoint_token_secret")
endpoint_token = dbutils.secrets.get(scope=endpoint_token_scope, key=endpoint_token_secret)
endpoint_workload_type = "GPU_SMALL" 
endpoint_workload_size = "Small" 
endpoint_scale_to_zero = False

# Connectivity to dependencies (vector search, guardrails, etc)
dependencies_host = dbutils.widgets.get("dependencies_host")
dependencies_token_scope = dbutils.widgets.get("dependencies_token_scope")
dependencies_token_secret = dbutils.widgets.get("dependencies_token_secret")
dependencies_token = dbutils.secrets.get(scope=dependencies_token_scope, key=dependencies_token_secret)

# Traching table details
tracking_table_catalog = dbutils.widgets.get("tracking_table_catalog")
tracking_table_schema = dbutils.widgets.get("tracking_table_schema")
tracking_table_name = dbutils.widgets.get("tracking_table_name")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Serving Endpoint

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput

endpoint_config_dict = { 
                        "served_entities": [
                                {
                                    "entity_name": model_name,
                                    "entity_version": model_version,
                                    "workload_size": endpoint_workload_size,
                                    "scale_to_zero_enabled": endpoint_scale_to_zero,
                                    "workload_type": endpoint_workload_type,
                                    "environment_vars": {
                                        "DATABRICKS_HOST": f"{dependencies_host}",
                                        "DATABRICKS_TOKEN": f"{dependencies_token}"
                                        }
                                }
                        ]
                    }

if tracking_table_catalog and tracking_table_schema and tracking_table_name:
    endpoint_config_dict["auto_capture_config"] = {
        "catalog_name": f"{tracking_table_catalog}",
        "schema_name": f"{tracking_table_schema}",
        "table_name_prefix": f"{tracking_table_name}"
                            }
endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

print(endpoint_config)

# COMMAND ----------

deploy_model_serving_endpoint(endpoint_name, endpoint_config, endpoint_host)

# COMMAND ----------

import time

# Make the process sleep for 15 seconds
time.sleep(15)

# COMMAND ----------

test_serving_endpoint(endpoint_name, endpoint_host, endpoint_token)
