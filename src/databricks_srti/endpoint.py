
import urllib
import json
import logging
from typing import List, Dict


class EndpointApiClient:
    """
        Wrapper class around Databricks Serverless Realtime 
        Inference Endpoints.
    """

    def __init__(self, base_url: str, token: str):
        """
            Instantiates an EndpointApiClient. Parameters:
            base_url: A string pointing to your workspace URL.
            token: Access Token for interacting with your Databricks Workspace.
        """
        self.base_url = base_url
        self.token = token

    def create_inference_endpoint(
            self,
            endpoint_name: str,
            served_models: List[str]
        ):
        """
            Creates inference endpoints for models.
            endpoint_name: Serving endpoint name.
            served_models: List of model names that will be deployed.
        """

        data = {"name": endpoint_name, "config": {"served_models": served_models}}
        return self._post("api/2.0/serving-endpoints", data)

    def get_inference_endpoint(self, endpoint_name: str) -> Dict:
        """
            Gets info on the inference endpoint.

            endpoint_name: Serving endpoint name.
        """

        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}")

    def list_inference_endpoints(self) -> Dict:
        """Lists all running inference endpoints."""

        return self._get("api/2.0/serving-endpoints")

    def update_served_models(
        self,
        endpoint_name: str,
        served_models: List[str],
        traffic_config: Dict = None
    ):
        """
            Updates served models with the specified traffic_config.
            
            endpoint_name: Serving endpoint name.
            served_models: List of served models.
            traffic_config: New traffic split configuration.
        """

        if traffic_config is None:
            data = data = {"served_models": served_models}
        else:
            data = {"served_models": served_models, "traffic_config": traffic_config}
        return self._put(f"api/2.0/serving-endpoints/{endpoint_name}/config", data)

    def delete_inference_endpoint(self, endpoint_name: str) -> Dict:
        """
            Deletes an inference endpoint.
            
            endpoint_name: Serving endpoint name.
        """

        return self._delete(f"api/2.0/serving-endpoints/{endpoint_name}")

    def query_inference_endpoint(self, endpoint_name: str, data: Dict) -> Dict:
        """
            Makes HTTP requests to an inference endpoint.
            
            endpoint_name: Serving endpoint name.
            data: Payload containing the data expected by the model.
        """

        return self._post(f"realtime-inference/{endpoint_name}/invocations", data)

    # Debugging

    def get_served_model_build_logs(
        self,
        endpoint_name: str,
        served_model_name: str
    ) -> Dict:
        """
            Gets the build logs for the specified endpoint/model.
            
            endpoint_name: Serving endpoint name.
            served_model_name: Served model name.
        """

        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/build-logs"
        )

    def get_served_model_server_logs(
        self,
        endpoint_name: str,
        served_model_name: str
    ) -> Dict:
        """
            Gets the server logs for the specified endpoint/model.
            
            endpoint_name: Serving endpoint name.
            served_model_name: Served model name.
        """

        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/logs"
        )

    def get_inference_endpoint_events(self, endpoint_name: str) -> Dict:
        """
            Gets the build endpoint events for the specified endpoint.
            
            endpoint_name: Serving endpoint name.
        """

        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}/events")

    def _get(self, uri) -> Dict:
        url = f"{self.base_url}/{uri}"
        headers = {"Authorization": f"Bearer {self.token}"}

        req = urllib.request.Request(url, headers=headers)
        return self._make_request(req)

    def _post(self, uri, body) -> Dict:
        json_body = json.dumps(body)
        json_bytes = json_body.encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/{uri}"
        req = urllib.request.Request(url, data=json_bytes, headers=headers)
        return self._make_request(req)

    def _put(self, uri, body) -> Dict:
        json_body = json.dumps(body)
        json_bytes = json_body.encode("utf-8")
        headers = {"Authorization": f"Bearer {self.token}"}

        url = f"{self.base_url}/{uri}"
        req = urllib.request.Request(
            url, data=json_bytes, headers=headers, method="PUT"
        )
        return self._make_request(req)

    def _delete(self, uri) -> Dict:
        headers = {"Authorization": f"Bearer {self.token}"}
        url = f"{self.base_url}/{uri}"
        req = urllib.request.Request(url, headers=headers, method="DELETE")
        return self._make_request(req)

    def _make_request(self, req) -> Dict:
        try:
            logging.info(f"Making request: {req}")
            response = urllib.request.urlopen(req)
            return json.load(response)
        except urllib.error.HTTPError as err:
            error_msg = f"""
                Got error code {err.code} while making request: {str(err)};
                Reason: {err.reason};
                Headers: {err.headers}
            """
            logging.error(error_msg)