# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %pip install databricks-agents langchain==0.2.16 langchain-community==0.2.17 databricks-vectorsearch pydantic==1.10.9 mlflow==2.16.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./helper

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

model_name = dbutils.widgets.get("model_name")
host = dbutils.widgets.get("host")
endpoint_token = dbutils.secrets.get(scope="creds", key="pat")

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy app
# MAGIC - It can take a few minutes to complete

# COMMAND ----------

from databricks.agents import deploy

latest_version = get_latest_model_version(model_name)

deploy(model_name, 
       latest_version, 
       environment_vars={
           "DATABRICKS_HOST": f"{host}",
           "DATABRICKS_TOKEN": f"{endpoint_token}"
           })
