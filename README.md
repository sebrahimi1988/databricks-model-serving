# Databricks Model Serving Endpoints Python Client

<hr />

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![ci](https://github.com/sebrahimi1988/databricks-srti-demo/actions/workflows/ci.yml/badge.svg?style=for-the-badge)

<hr/>

## Introduction

* This is a Python framework which wraps [Databricks Model Serving Endpoints API](https://www.databricks.com/blog/2023/03/07/announcing-general-availability-databricks-model-serving.html#:~:text=Databricks%20Model%20Serving%20is%20the,reducing%20mistakes%20through%20integrated%20tools) functionality.
* With a few lines of code, you can:
  * Deploy realtime models
  * Distribute traffic across two or more models running under the same endpoint (e.g. for A/B testing)
  * Inspect model build and server logs
  
## Getting Started

To get started, simply install the package from this repo:

```bash
pip install https://github.com/sebrahimi1988/databricks-model-serving
```

Once the package is installed, you can leverage different functions in the `EndpointClient` class to `list`, `create` and `update` endpoints, amongst others. For instance, to list all model serving endpoints from a particular workspace:

```python
from databricks.model_serving.client import EndpointClient

client = EndpointClient(databricks_url, databricks_token)
client.list_inference_endpoints()
```

## Examples

In the `notebooks` folder you can find an example use case, where we train a model, register it in Model Registry and deploy it using the framework.

* [01_train](https://github.com/sebrahimi1988/databricks-model-serving/tree/main/notebooks/01_train.py) trains two models and registers them in model registry prior to serving.

* [02_serve](https://github.com/sebrahimi1988/databricks-model-serving/tree/main/notebooks/02_serve.py) demonstrates how to serve models using Model Serving.

* [03_debug](https://github.com/sebrahimi1988/databricks-model-serving/tree/main/notebooks/03_debug.py) shows you how to investigate logs of the endpoint for debugging purposes.
