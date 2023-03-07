from databricks.model_serving.client import EndpointClient
import pytest
import requests_mock

@pytest.fixture
def response():
    return '{"key": "response"}'

@pytest.fixture
def url():
    return "http://fake.url/"

@pytest.fixture
def client(url):
    client = EndpointClient(
        base_url = url,
        token = "FAKETOKEN"
    )

    return client


def test_create_inference_endpoint(response, client):
    
    with requests_mock.Mocker() as m:
        m.post(requests_mock.ANY, text = response)
        result = client.create_inference_endpoint(
            endpoint_name = "endpoint",
            served_models = ["mymodel"]
        )
    
    assert result is not None

def test_get_inference_endpoint(response, client):

    with requests_mock.Mocker() as m:
        url = "http://fake.url/"
        m.get(requests_mock.ANY, text = response)
        
        result = client.get_inference_endpoint(endpoint_name = "endpoint")
    
    assert result is not None

def test_get_inference_endpoint(response, client):

    with requests_mock.Mocker() as m:
        m.get(requests_mock.ANY, text = response)
        result = client.get_inference_endpoint(endpoint_name = "endpoint")
    
    assert result is not None

def test_list_inference_endpoints(response, client):

    with requests_mock.Mocker() as m:
        m.get(requests_mock.ANY, text = response)
        result = client.list_inference_endpoints()
    
    assert result is not None

def test_update_served_models(response, client):

    with requests_mock.Mocker() as m:
        m.put(requests_mock.ANY, text = response)
        result = client.update_served_models(
            endpoint_name = "endpoint",
            served_models = ["model_a", "model_b"],
            traffic_config = {"model_a": 0.5, "model_b": 0.5}
        )
    
    assert result is not None