from enum import Enum


class Endpoint(Enum):
    SERVING = "api/2.0/serving-endpoints"
    INVOCATIONS = "serving-endpoints/{}/invocations"
    SERVED_MODELS = "api/2.0/serving-endpoints/{}/served-models"
    EVENTS = "api/2.0/serving-endpoints/{}/events"
    CONFIG = "api/2.0/serving-endpoints/{}/config"
