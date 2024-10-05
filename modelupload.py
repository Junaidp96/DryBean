import boto3

# Initialize S3 client
s3 = boto3.client('s3')

# Upload the model to S3
bucket_name = 'drybean-csv'  # Your bucket name
model_path = 'model/model.joblib'  # Local model path
s3_model_path = 'model/model.joblib'  # S3 path where the model will be stored

s3.upload_file(model_path, bucket_name, s3_model_path)
print("Model uploaded to S3.")
