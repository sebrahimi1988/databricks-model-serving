# Databricks Model Serving Endpoints Python Client

<hr />

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![ci](https://github.com/sebrahimi1988/databricks-srti-demo/actions/workflows/ci.yml/badge.svg?style=for-the-badge)

<hr/>


This repo demonstrates the capabilities of the Serverless Real Time Inference (SRTI) on Databricks. It consists of 4 notebooks:

## Examples

* [01_train](./Notebooks/01_train) trains two models and registers them in model registry prior to serving.

* [02_serve](./Notebooks/02_serve) demonstrates how to serve models using SRTI.

* [03_debug](./Notebooks/03_debug) shows you how to investigate logs of the endpoint for debugging purposes.
