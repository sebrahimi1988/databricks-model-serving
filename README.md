# databricks-srti-demo

This repo demonstrates the capabilities of the Serverless Real Time Inference (SRTI) on Databricks. It consists of 4 notebooks:

* [00_srti_client](./Notebooks/00_srti_client) is a generic client to interact with the SRTI REST API.

* [01_train](./Notebooks/01_train) trains two models and registers them in model registry prior to serving.

* [02_serve](./Notebooks/02_serve) demonstrates how to serve models using SRTI.

* [03_debug](./Notebooks/03_debug) shows you how to investigate logs of the endpoint for debugging purposes.