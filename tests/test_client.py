from databricks.model_serving.client import EndpointClient
import pytest
import requests_mock

@pytest.fixture
def response():
    return '{"key": "response"}'


def test_create_inference_endpoint(response):
    
    with requests_mock.Mocker() as m:
        url = "http://fake.url/"
        m.post(requests_mock.ANY, text = response)
        client = EndpointClient(
            base_url = url,
            token = "FAKETOKEN"
        )

        result = client.create_inference_endpoint(
            endpoint_name = "endpoint",
            served_models = ["mymodel"]
        )
    
    assert result is not None

def test_get_inference_endpoint(response):

    with requests_mock.Mocker() as m:
        url = "http://fake.url/"
        m.get(requests_mock.ANY, text = response)
        client = EndpointClient(
            base_url = url,
            token = "FAKETOKEN"
        )
        
        result = client.get_inference_endpoint(endpoint_name = "endpoint")
    
    assert result is not None