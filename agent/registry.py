DATA_REGISTRY = {}
MODEL_REGISTRY = {}

def store_data(key, value):
    DATA_REGISTRY[key] = value

def get_data(key):
    return DATA_REGISTRY.get(key)

def store_model(key, model):
    MODEL_REGISTRY[key] = model

def get_model(key):
    return MODEL_REGISTRY.get(key)
