"""import joblib

# Load the pickle file
with open('/workspace/housing-market-analysis/outputs/ml_pipeline/cluster_analysis/v1/cluster_pipeline.pkl', 'rb') as f:
    cluster_pipeline = joblib.load(f)

# Now you can inspect the contents of the cluster_pipeline object
# For example, you can print its attributes or perform any other operations you need
print(cluster_pipeline)
"""


import joblib

file_path = "/workspace/housing-market-analysis/outputs/ml_pipeline/predict_saleprice/v1/clf_pipeline_model.pkl"

try:
    clf_pipeline_model = joblib.load(file_path)
    print("Pipeline object loaded successfully.")
    # Further operations using clf_pipeline_model
except FileNotFoundError:
    print(f"File '{file_path}' not found. Please double-check the file path.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
