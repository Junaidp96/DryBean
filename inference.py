import json
import joblib
import pandas as pd
from sagemaker_inference import content_types  # Removed 'environments' import

def input_fn(request_body, request_content_type):
    # Parse the incoming request based on content type
    if request_content_type == content_types.JSON:
        data = json.loads(request_body)
        return pd.DataFrame(data)  # Convert to DataFrame if necessary
    raise ValueError(f'Unsupported content type: {request_content_type}')

def predict_fn(input_data, model):
    # Make predictions using the loaded model
    return model.predict(input_data)

def output_fn(prediction, response_content_type):
    # Format the output as JSON
    if response_content_type == content_types.JSON:
        return json.dumps(prediction.tolist()), response_content_type
    raise ValueError(f'Unsupported content type: {response_content_type}')
