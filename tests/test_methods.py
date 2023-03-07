from databricks_srti.endpoint import EndpointApiClient
import pytest
import requests


def test_enabled(responses):
    
    url = "http://fake.url/api/2.0/serving-endpoints"
    client = EndpointApiClient(
        base_url = url,
        token = "FAKETOKEN"
    )

    responses.add(
        responses.POST,
        url = url,
        json = 1
    )

    result = client.create_inference_endpoint(
        endpoint_name = "endpoint",
        served_models = ["mymodel"]
    )

    #assert len(responses.calls) == 1
    assert result is not None