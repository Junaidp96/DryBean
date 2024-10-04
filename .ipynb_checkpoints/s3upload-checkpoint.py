import boto3

# Initialize the S3 client
s3 = boto3.client('s3')

# Specify the bucket name and file paths
bucket_name = 'drybean-csv'  # Replace with your S3 bucket name
file_name = 'drybean.csv'  # Local file in SageMaker Studio
s3_file_path = 'data/drybean.csv'  # S3 folder path with file name

# Upload the file to the 'data' folder in the S3 bucket
s3.upload_file(file_name, bucket_name, s3_file_path)

print(f"{file_name} has been uploaded to {bucket_name}/data/ as {s3_file_path}")

