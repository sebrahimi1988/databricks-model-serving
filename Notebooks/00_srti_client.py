# Databricks notebook source
import urllib
import json


class EndpointApiClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.token = token

    # CRUD

    def create_inference_endpoint(self, endpoint_name, served_models):
        data = {"name": endpoint_name, "config": {"served_models": served_models}}
        return self._post("api/2.0/serving-endpoints", data)

    def get_inference_endpoint(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}")

    def list_inference_endpoints(self):
        return self._get("api/2.0/serving-endpoints")

    def update_served_models(self, endpoint_name, served_models, traffic_config=None):
        if traffic_config is None:
            data = data = {"served_models": served_models}
        else:
            data = {"served_models": served_models, "traffic_config": traffic_config}
        return self._put(f"api/2.0/serving-endpoints/{endpoint_name}/config", data)

    def delete_inference_endpoint(self, endpoint_name):
        return self._delete(f"api/2.0/serving-endpoints/{endpoint_name}")

    # Making predictions

    def query_inference_endpoint(self, endpoint_name, data):
        return self._post(f"realtime-inference/{endpoint_name}/invocations", data)

    # Debugging

    def get_served_model_build_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/build-logs"
        )

    def get_served_model_server_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/logs"
        )

    def get_inference_endpoint_events(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}/events")

    def _get(self, uri):
        url = f"{self.base_url}/{uri}"
        headers = {"Authorization": f"Bearer {self.token}"}

        req = urllib.request.Request(url, headers=headers)
        return self._make_request(req)

    def _post(self, uri, body):
        json_body = json.dumps(body)
        json_bytes = json_body.encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/{uri}"
        req = urllib.request.Request(url, data=json_bytes, headers=headers)
        return self._make_request(req)

    def _put(self, uri, body):
        json_body = json.dumps(body)
        json_bytes = json_body.encode("utf-8")
        headers = {"Authorization": f"Bearer {self.token}"}

        url = f"{self.base_url}/{uri}"
        req = urllib.request.Request(
            url, data=json_bytes, headers=headers, method="PUT"
        )
        return self._make_request(req)

    def _delete(self, uri):
        headers = {"Authorization": f"Bearer {self.token}"}
        url = f"{self.base_url}/{uri}"
        req = urllib.request.Request(url, headers=headers, method="DELETE")
        return self._make_request(req)

    def _make_request(self, req):
        try:
            response = urllib.request.urlopen(req)
            return json.load(response)
        except urllib.error.HTTPError as err:
            print(err)
            print(err.code)
            print(err.reason)
            print(err.headers)
