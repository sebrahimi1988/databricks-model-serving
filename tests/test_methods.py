from databricks.model_serving.client import EndpointClient
import pytest
import requests_mock

@pytest.fixture
def mock_requests():
    adapter = requests_mock.Adapter()
    adapter.register_uri(
        requests_mock.ANY,
        requests_mock.ANY,
        text='response'
    )

def test_create_inference_endpoint():
    
    with requests_mock.Mocker() as m:
        url = "http://fake.url/"
        m.post(requests_mock.ANY, text = '{"key": "response"}')
        client = EndpointClient(
            base_url = url,
            token = "FAKETOKEN"
        )

        result = client.create_inference_endpoint(
            endpoint_name = "endpoint",
            served_models = ["mymodel"]
        )
    

    assert result is not None