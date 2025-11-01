import kagglehub
import os
import json
import shutil
from pathlib import Path

def setup_kaggle_credentials():
    # Get the current directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kaggle_json_path = os.path.join(current_dir, 'kaggle.json')
    
    # Create .kaggle directory in home if it doesn't exist
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Copy kaggle.json to .kaggle directory
    kaggle_dest = os.path.join(kaggle_dir, 'kaggle.json')
    shutil.copy2(kaggle_json_path, kaggle_dest)
    
    # Set proper permissions
    os.chmod(kaggle_dest, 0o600)
    print("Kaggle credentials configured successfully")

def download_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Setup Kaggle credentials first
        setup_kaggle_credentials()
        
        # Download dataset
        print("Downloading dataset...")
        path = kagglehub.dataset_download("thedatasith/sku110k-annotations")
        print("Dataset files downloaded to:", path)
        
        # Move files to current directory if needed
        if path != current_dir:
            for file_path in Path(path).glob('*'):
                dest_path = os.path.join(current_dir, file_path.name)
                if os.path.exists(dest_path):
                    os.remove(dest_path)  # Remove existing file if it exists
                shutil.move(str(file_path), current_dir)
            print("Files moved to dataset directory successfully")

    except Exception as e:
        print("Error during download or setup:", str(e))
        return False
    
    return True

if __name__ == "__main__":
    success = download_dataset()
    if success:
        print("Dataset setup completed successfully")
    else:
        print("Failed to setup dataset")