# Databricks notebook source
# MAGIC %md # Train and register a Scikit learn model for model serving
# MAGIC 
# MAGIC This notebook trains an ElasticNet and a LassoCV model using the diabetes dataset from scikit learn. Databricks autologging is also used to both log metrics and to register the trained models to the Databricks Model Registry.
# MAGIC 
# MAGIC After running the code in this notebook, you have a registered model ready for model serving with Serverless Real-Time Inference ([AWS](https://docs.databricks.com/applications/mlflow/serverless-real-time-inference.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/serverless-real-time-inference)).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import libraries

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LassoCV
from sklearn import datasets

# Import mlflow
import mlflow
import mlflow.sklearn

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data

# COMMAND ----------

# Load Diabetes datasets
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data
diabetes_y = diabetes.target

# Create pandas DataFrame for sklearn ElasticNet linear_model
diabetes_Y = np.array([diabetes_y]).transpose()
d = np.concatenate((diabetes_X, diabetes_Y), axis=1)
cols = diabetes.feature_names + ['progression']
diabetes_data = pd.DataFrame(d, columns=cols)
# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(diabetes_data)

# The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
train_x = train.drop(["progression"], axis=1)
test_x = test.drop(["progression"], axis=1)
train_y = train[["progression"]]
test_y = test[["progression"]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and register models
# MAGIC The following cells train two different models to solve the same problem. The first one is an [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet) model which will be used as a `Production` model and the second one is a [LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV) model wich will be used in `Staging`.
# MAGIC 
# MAGIC The code  automattically logs the trained model. By specifying `registered_model_name` in the autologging configuration, the model trained is automatically registered to the Databricks Model Registry in `None` stage. We then transition the new models to `Production/Staging` stages.

# COMMAND ----------

# Archive any existing versions of the model
model_name = "Diabetes_srti_demo"

mlflow_client = mlflow.MlflowClient()
for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
    if mv.current_stage in ["Production", "Staging", None]:
        mlflow_client.transition_model_version_stage(
            name=model_name, version=mv.version, stage="Archived"
        )

# COMMAND ----------

mlflow.sklearn.autolog(log_input_examples=True, registered_model_name=model_name)

# Run ElasticNet
lr_ElasticNet = ElasticNet(alpha=0.05, l1_ratio=0.05, random_state=42)
lr_ElasticNet.fit(train_x, train_y)

# Get the version of the new ElasticNet model
latest_model_version = mlflow_client.get_latest_versions(model_name, stages=["None"])[
    -1
].version

# Move the new ElasticNet model from "None" to "Production" stage
mlflow_client.transition_model_version_stage(
    name=model_name,
    version=latest_model_version,
    stage="Production",
)

# COMMAND ----------

mlflow.sklearn.autolog(log_input_examples=True, registered_model_name=model_name)

# Run LassoCV
lr_LassoCV = LassoCV(cv=5, random_state=42)
lr_LassoCV.fit(train_x, train_y)

latest_model_version = mlflow_client.get_latest_versions(
    model_name, stages=["None"]
)[-1].version

# Move the new model to Production in the model registry
mlflow_client.transition_model_version_stage(
    name=model_name,
    version=latest_model_version,
    stage="Staging",
)
