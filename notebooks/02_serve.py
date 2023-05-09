# Databricks notebook source
# MAGIC %md # Serve a Scikit learn model at a REST API endpoint 
# MAGIC
# MAGIC This notebook demonstrates how to serve Scikit learn models at REST API endpoints with Serverless Real-Time Inference ([AWS](https://docs.databricks.com/applications/mlflow/serverless-real-time-inference.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/serverless-real-time-inference)).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import libraries

# COMMAND ----------

# MAGIC %pip install ../ -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks.model_serving.client import EndpointClient

# get API URL and token 
databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

mlflow_client = mlflow.MlflowClient()
client = EndpointClient(databricks_url, databricks_token)

# COMMAND ----------

# MAGIC %md
# MAGIC ## List existing endpoints

# COMMAND ----------

client = EndpointClient(databricks_url, databricks_token)
client.list_inference_endpoints()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create an endpoint with your model
# MAGIC
# MAGIC In the endpoint object returned by the create call, we can see that our endpoint’s update state is `IN_PROGRESS` and our served model is in a `CREATING` state. The `pending_config` field shows the details of the update in progress.

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

client.list_inference_endpoints()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Check the state of your endpoint
# MAGIC
# MAGIC We can check on the status of our endpoint to see if it is ready to receive traffic. Note that when the update is complete and the endpoint is ready to be queried, the `pending_config` is no longer populated.

# COMMAND ----------

import time

endpoint = client.get_inference_endpoint(endpoint_name)

while endpoint['state']['config_update'] == "IN_PROGRESS":
  time.sleep(5)
  endpoint = client.get_inference_endpoint(endpoint_name)
  print(endpoint["name"], endpoint["state"])

# COMMAND ----------

client.get_inference_endpoint(endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Query the endpoint
# MAGIC
# MAGIC Now that our endpoint is ready, we can query it

# COMMAND ----------

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
# MAGIC
# MAGIC ## Update the compute config of the endpoint's served model
# MAGIC
# MAGIC Say we want the endpoint to scale to zero to cut down the cost while the endpoint is not called overnight. To do this, we can perform an update on our endpoint. 

# COMMAND ----------

models = [{
  "model_name": model_name,
  "model_version": model_version,
  "workload_size": "Small",
  "scale_to_zero_enabled": True,
}]
client.update_served_models(endpoint_name, models)

# COMMAND ----------

import time

endpoint = client.get_inference_endpoint(endpoint_name)

while endpoint['state']['config_update'] == "IN_PROGRESS":
  time.sleep(5)
  endpoint = client.get_inference_endpoint(endpoint_name)
  print(endpoint["name"], endpoint["state"])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Update the endpoint's served model
# MAGIC
# MAGIC We can also use the update API to change the underlying model (e.g we trained a newer version of our model and would like the endpoint to serve the newer version of the model instead) and define how much of the traffic to be routed to that endpoint by setting the `traffic_percentage` variable. If only a single model is served at an endpoint the `traffic_percentage` is set to 100% by default.
# MAGIC
# MAGIC In this example we are going to use the `Staging` version of our Diabetes model as a second model behind our endpoint and we will route 25% of the traffic to this new model.

# COMMAND ----------

model_name_staging = "Diabetes_srti_demo_staging"
model_version_staging = mlflow_client.get_latest_versions(
    model_name, stages=["Staging"]
)[-1].version
models = [
    {
        "model_name": model_name,
        "model_version": model_version,
        "workload_size": "Small",
        "scale_to_zero_enabled": True,
    },
    {
        "model_name": model_name,
        "model_version": model_version_staging,
        "workload_size": "Small",
        "scale_to_zero_enabled": True,
    },
]
traffic_config = {
    "routes": [
        {
            "served_model_name": f"{model_name}-{model_version}",
            "traffic_percentage": "50",
        },
        {
            "served_model_name": f"{model_name}-{model_version_staging}",
            "traffic_percentage": "50",
        },
    ]
}
client.update_served_models(endpoint_name, models, traffic_config)

# COMMAND ----------

# MAGIC %md 
# MAGIC While the update is in progress, we can continue to query the endpoint (serving the original Production model). Once the update is complete, the endpoint will start to return responses from the Production model(50% of the traffic) as well as the Staging (50% of the traffic) one. 

# COMMAND ----------

client.query_inference_endpoint(endpoint_name, input_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that while there is an update in progress, another update cannot be made.

# COMMAND ----------

client.get_inference_endpoint(endpoint_name)

# COMMAND ----------

import time

endpoint = client.get_inference_endpoint(endpoint_name)

while endpoint['state']['config_update'] == "IN_PROGRESS":
  time.sleep(5)
  endpoint = client.get_inference_endpoint(endpoint_name)
  print(endpoint["name"], endpoint["state"])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Delete an endpoint
# MAGIC
# MAGIC Lastly, if we no longer need an endpoint, we can delete it

# COMMAND ----------

client.delete_inference_endpoint(endpoint_name)

# COMMAND ----------

client.list_inference_endpoints()

# COMMAND ----------


