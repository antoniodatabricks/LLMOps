# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %pip install mlflow databricks-sdk evaluate rouge_score databricks-agents

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./helper

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

# Information about the endpoint that we are evaluating
challenger_endpoint_name = dbutils.widgets.get("challenger_endpoint_name")
challenger_endpoint_host = dbutils.widgets.get("challenger_endpoint_host")
challenger_endpoint_token_scope = dbutils.widgets.get("challenger_endpoint_token_scope")
challenger_endpoint_token_secret = dbutils.widgets.get("challenger_endpoint_token_secret")
challenger_endpoint_token = dbutils.secrets.get(scope=challenger_endpoint_token_scope, key=challenger_endpoint_token_secret)

# Information about the endpoint currently in production
champion_endpoint_name = dbutils.widgets.get("champion_endpoint_name")
champion_endpoint_host = dbutils.widgets.get("champion_endpoint_host")
champion_endpoint_token_scope = dbutils.widgets.get("champion_endpoint_token_scope")
champion_endpoint_token_secret = dbutils.widgets.get("champion_endpoint_token_secret")
champion_endpoint_token = dbutils.secrets.get(scope=champion_endpoint_token_scope, key=champion_endpoint_token_secret)

# Evaludation dataset
evaluation_dataset = dbutils.widgets.get("evaluation_dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC # Build data set for evaluation

# COMMAND ----------

# DBTITLE 1,Evaluation dataset
df_eval = spark.table(evaluation_dataset).limit(10) # Selecting only 10 records for test purposes

# COMMAND ----------

# DBTITLE 1,Build question / answer from DEV and PROD
import mlflow
import pandas as pd

dev_ds_for_eval = []
prod_ds_for_eval = []
for row in df_eval.collect():

    question = row["request"]
    expected_answer = row["expected_response"]

    answer_dev = score_model(question, challenger_endpoint_host, challenger_endpoint_name, challenger_endpoint_token)
    answer_prod = score_model(question, champion_endpoint_host, champion_endpoint_name, champion_endpoint_token)

    dev_ds_for_eval.append({"request": f"{question}", "response": f"{answer_dev}", "retrieved_context": [], "expected_response": f"{expected_answer}"})
    prod_ds_for_eval.append({"request": f"{question}", "response": f"{answer_prod}", "retrieved_context": [], "expected_response": f"{expected_answer}"})

pd_dev_data = pd.DataFrame(dev_ds_for_eval)
pd_prod_data = pd.DataFrame(prod_ds_for_eval)

# COMMAND ----------

# MAGIC %md
# MAGIC # Example: Databricks Mosaic Agent (LLM as a judge)

# COMMAND ----------

result = mlflow.evaluate(
    data=pd_dev_data,
    model_type="databricks-agent",
)

display(result.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Example: MLflow evaluators (LLM as a judge)

# COMMAND ----------

pd_dev_data_mlflow = pd_dev_data.rename(columns={'request': 'inputs', 'expected_response': 'ground_truth', 'response': 'predictions'})
pd_dev_data_mlflow = pd_dev_data_mlflow[['inputs', 'ground_truth', 'predictions']]

pd_prod_data_mlflow = pd_prod_data.rename(columns={'request': 'inputs', 'expected_response': 'ground_truth', 'response': 'predictions'})
pd_prod_data_mlflow = pd_prod_data_mlflow[['inputs', 'ground_truth', 'predictions']]

# COMMAND ----------

with mlflow.start_run() as run:
    results = mlflow.evaluate(
        data=pd_dev_data_mlflow,
        targets="ground_truth",
        predictions="predictions",
        model_type="question-answering"
    )
    
    results.tables["eval_results_table"].display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaludate the model in DEV and PROD

# COMMAND ----------

with mlflow.start_run() as run:
    dev_results = mlflow.evaluate(
        data=pd_dev_data_mlflow,
        targets="ground_truth",
        predictions="predictions",
        model_type="text-summarization"
    )
    
    dev_results.tables["eval_results_table"].display()

# COMMAND ----------

with mlflow.start_run() as run:
    prod_results = mlflow.evaluate(
        data=pd_prod_data_mlflow,
        targets="ground_truth",
        predictions="predictions",
        model_type="text-summarization"
    )
    
    prod_results.tables["eval_results_table"].display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Compare DEV metrics to PROD metrics

# COMMAND ----------

assert float(dev_results.metrics["rouge1/v1/mean"]) >= float(prod_results.metrics["rouge1/v1/mean"]) , "rouge1/v1/mean is not greater than prod"
assert float(dev_results.metrics["rouge1/v1/variance"]) >= float(prod_results.metrics["rouge1/v1/variance"]), "rouge1/v1/variance is not greater than prod"
assert float(dev_results.metrics["rouge1/v1/p90"]) >= float(prod_results.metrics["rouge1/v1/p90"]), "rouge1/v1/p90 is not greater than prod"
assert float(dev_results.metrics["rouge2/v1/mean"]) >= float(prod_results.metrics["rouge2/v1/mean"]), "rouge2/v1/mean is not greater than prod"
assert float(dev_results.metrics["rouge2/v1/variance"]) >= float(prod_results.metrics["rouge2/v1/variance"]), "rouge2/v1/variance is not greater than prod"
assert float(dev_results.metrics["rouge2/v1/p90"]) >= float(prod_results.metrics["rouge2/v1/p90"]), "rouge2/v1/p90 is not greater than prod"
assert float(dev_results.metrics["rougeL/v1/mean"]) >= float(prod_results.metrics["rougeL/v1/mean"]), "rougeL/v1/mean is not greater than prod"
assert float(dev_results.metrics["rougeL/v1/variance"]) >= float(prod_results.metrics["rougeL/v1/variance"]), "rougeL/v1/variance is not greater than prod"
assert float(dev_results.metrics["rougeL/v1/p90"]) >= float(prod_results.metrics["rougeL/v1/p90"]), "rougeL/v1/p90 is not greater than prod"
assert float(dev_results.metrics["rougeLsum/v1/mean"]) >= float(prod_results.metrics["rougeLsum/v1/mean"]), "rougeLsum/v1/mean is not greater than prod"
assert float(dev_results.metrics["rougeLsum/v1/variance"]) >= float(prod_results.metrics["rougeLsum/v1/variance"]), "rougeLsum/v1/variance is not greater than prod"
assert float(dev_results.metrics["rougeLsum/v1/p90"]) >= float(prod_results.metrics["rougeLsum/v1/p90"]), "rougeLsum/v1/p90 is not greater than prod"
