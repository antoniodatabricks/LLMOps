# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -U --quiet langchain==0.2.16 langchain-community==0.2.17 databricks-vectorsearch pydantic==1.10.9 mlflow==2.16.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

model_to_promote = "llmops_dev.model_schema.basic_rag_demo_foundation_model"
catalog_qa = "llmops_qa"
schema_qa = "model_schema"
model_name_qa = "basic_rag_demo_foundation_model"

# COMMAND ----------

# MAGIC %md
# MAGIC # Get latest model version

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")

def get_latest_model_version(model_name_in:str = None):
    """
    Get latest version of registered model
    """
    client = MlflowClient()
    model_version_infos = client.search_model_versions("name = '%s'" % model_name_in)

    if model_version_infos:
      return max([int(model_version_info.version) for model_version_info in model_version_infos])
    else:
      return None

# COMMAND ----------

latest_version = get_latest_model_version(model_to_promote)
model_to_promote_uri = f"models:/{model_to_promote}/{latest_version}"
print(model_to_promote_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy to PROD

# COMMAND ----------

prod_model = f"{catalog_qa}.{schema_qa}.{model_name_qa}"
print(prod_model)

# COMMAND ----------

copied_model_version = mlflow.register_model(
    model_uri=model_to_promote_uri,
    name=prod_model
)

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias(name=f"{catalog_qa}.{schema_qa}.{model_name_qa}", alias="Champion", version=copied_model_version.version)
