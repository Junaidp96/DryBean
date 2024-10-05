import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn import SKLearnModel

# Set up SageMaker session and role
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Define the model with framework and Python version
model = SKLearnModel(
    model_data='s3://drybean-csv/model/model.tar.gz',  # Path to your model in S3
    role=role,
    entry_point='inference.py',  # Your inference script
    framework_version='0.23-1',  # Specify the framework version
    py_version='py3',  # Specify the Python version
    sagemaker_session=sagemaker_session,
)

# Deploy the model to create an endpoint
predictor = model.deploy(
    instance_type='ml.m4.xlarge',  # Specify the instance type
    initial_instance_count=1,  # Specify the number of instances
    endpoint_name='drybean-endpoint01',  # Name for your endpoint
)
