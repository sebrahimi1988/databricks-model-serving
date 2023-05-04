# Databricks notebook source
# MAGIC %md # Serve a Scikit learn model at a REST API endpoint 
# MAGIC
# MAGIC This notebook demonstrates how to serve Scikit learn models at REST API endpoints with Serverless Real-Time Inference ([AWS](https://docs.databricks.com/applications/mlflow/serverless-real-time-inference.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/serverless-real-time-inference)).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import libraries

# COMMAND ----------

import mlflow
from src.databricks.model_serving.client import EndpointClient

# get API URL and token 
databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

mlflow_client = mlflow.MlflowClient()
client = EndpointClient(databricks_url, databricks_token)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an endpoint with your model 

# COMMAND ----------

model_name = "Diabetes_srti_demo"
endpoint_name = "Diabetes_ep_srti_demo"

model_version = mlflow_client.get_latest_versions(model_name, stages=["Production"])[
    -1
].version
models = [
    {
        "model_name": model_name,
        "model_version": model_version,
        "workload_size": "Small",
        "scale_to_zero_enabled": False,
    }
]
client.create_inference_endpoint(endpoint_name, models)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Query the endpoint
# MAGIC
# MAGIC Now that our endpoint is ready, we can query it

# COMMAND ----------

endpoint_name = "Diabetes_ep_srti_demo"
input_data = {
    "dataframe_records": [
        {
            "age": -0.0127796318808497,
            "sex": -0.044641636506989,
            "bmi": -0.0654856181992578,
            "bp": -0.0699375301828207,
            "s1": 0.00118294589619092,
            "s2": 0.0168487333575743,
            "s3": -0.0029028298070691,
            "s4": -0.00702039650329191,
            "s5": -0.0307512098645563,
            "s6": -0.0507829804784829,
        },
        {
            "age": -0.107225631607358,
            "sex": -0.044641636506989,
            "bmi": -0.0115950145052127,
            "bp": -0.0400993174922969,
            "s1": 0.0493412959332305,
            "s2": 0.0644472995495832,
            "s3": -0.0139477432193303,
            "s4": 0.0343088588777263,
            "s5": 0.00702686254915195,
            "s6": -0.0300724459043093,
        },
        {
            "age": 0.030810829531385,
            "sex": 0.0506801187398187,
            "bmi": 0.0466068374843559,
            "bp": -0.015999222636143,
            "s1": 0.0204462859110067,
            "s2": 0.0506687672308438,
            "s3": -0.0581273968683752,
            "s4": 0.0712099797536354,
            "s5": 0.0062093156165054,
            "s6": 0.00720651632920303,
        },
    ]
}
client.query_inference_endpoint(endpoint_name, input_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Debug an endpoint
# MAGIC logs/metrics/events are still available in the endpoint-centric workflow (there will be a UI soon).

# COMMAND ----------

client.get_inference_endpoint_events(endpoint_name)

# COMMAND ----------

# MAGIC %md ### Retrieve the logs associated with building the model's environment for a given serving endpoint's served model.

# COMMAND ----------

model_name = "Diabetes_srti_demo"


model_version = mlflow_client.get_latest_versions(model_name, stages=["Production"])[
    -1
].version
served_model_name = f"{model_name}-{model_version}"

client.get_served_model_build_logs(endpoint_name, served_model_name)

# COMMAND ----------

# MAGIC %md ### Retrieve the most recent log lines associated with a given serving endpoint's served model

# COMMAND ----------

client.get_served_model_server_logs(endpoint_name, served_model_name)
