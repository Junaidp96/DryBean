import argparse
import os
import pandas as pd
import boto3
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the model training function
def model_fn(model_dir):
    """Load model from model_dir"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def train(args):
    # Access bucket and file paths
    bucket = "drybean-csv"  # Your S3 bucket name
    train_data_key = "data/train_data.csv"  # Path to train data
    test_data_key = "data/test_data.csv"  # Path to test data

    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # Download files from S3
    print("Downloading training data from S3...")
    s3.download_file(bucket, train_data_key, 'train.csv')
    print("Downloading test data from S3...")
    s3.download_file(bucket, test_data_key, 'test.csv')

    # Load the datasets into pandas DataFrames
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # Drop rows with missing target values (Class)
    train_df = train_df.dropna(subset=['Class'])
    test_df = test_df.dropna(subset=['Class'])

    # Split data into features and labels
    X_train = train_df.drop(columns=['Class'])  # Assuming 'Class' is the label
    y_train = train_df['Class']

    X_test = test_df.drop(columns=['Class'])
    y_test = test_df['Class']

    # Train model
    print("Training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    print("Evaluating the model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Create model directory if it doesn't exist
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model to the specified model directory
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    print("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments for model directory
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'), help="Directory to save the model")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the training function
    train(args)
