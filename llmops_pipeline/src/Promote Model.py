# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %pip install -U --quiet langchain==0.2.16 langchain-community==0.2.17 databricks-vectorsearch pydantic==1.10.9 mlflow==2.16.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

source_model = dbutils.widgets.get("source_model")
source_model_version = dbutils.widgets.get("source_model_version")
target_model = dbutils.widgets.get("target_model")

# COMMAND ----------

# MAGIC %md
# MAGIC # Promote Model
# MAGIC
# MAGIC This step will only be executed if the target model name is different from the source model name and the model is not within "system.ai"

# COMMAND ----------

import mlflow

if target_model != source_model and not source_model.startswith("system.ai"):
    copied_model_version = mlflow.register_model(
        model_uri = f"models:/{source_model}/{source_model_version}",
        name = target_model
        )
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(name=target_model, alias="Champion", version=copied_model_version.version)
