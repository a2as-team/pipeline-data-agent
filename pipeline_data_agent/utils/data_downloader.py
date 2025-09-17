"""
Dataset downloader for the Pipeline data analysis agent.
Downloads pipeline data from Google Drive if not present locally.
"""

import os
import requests
import pandas as pd
from pathlib import Path


def download_dataset(data_path: str = "data/pipeline_data.parquet") -> str:
    """
    Download dataset from Google Drive if it doesn't exist locally.

    Args:
        data_path: Local path where dataset should be stored

    Returns:
        Path to the dataset file

    Raises:
        Exception: If download fails
    """
    # Google Drive direct download link (updated for better compatibility)
    file_id = "109vhmnSLN3oofjFdyb58l_rUZRa0d6C8"
    DATASET_URL = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=1"

    # Create data directory if it doesn't exist
    data_dir = Path(data_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if dataset already exists
    if os.path.exists(data_path):
        print(f"Dataset already exists at {data_path}")
        return data_path

    print(f"Downloading dataset from Google Drive to {data_path}...")
    print("Note: For large files, Google Drive download may require manual intervention.")
    print("If download fails, please manually download the dataset and place it at:")
    print(f"  {os.path.abspath(data_path)}")

    try:
        # Create session to handle cookies and redirects
        session = requests.Session()

        # Initial request
        response = session.get(DATASET_URL, stream=True)
        response.raise_for_status()

        # Check if response is HTML (indicating download confirmation page)
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type:
            print("Google Drive requires download confirmation for large files.")
            print("Please download manually from:")
            print(f"https://drive.google.com/file/d/{file_id}/view")
            raise Exception("Automatic download not available for large files. Please download manually.")

        # Save to file if we got the actual file
        with open(data_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify the file is valid parquet
        try:
            df = pd.read_parquet(data_path)
            print(f"Dataset downloaded successfully: {len(df)} rows, {len(df.columns)} columns")
            return data_path
        except Exception as e:
            # Remove invalid file
            if os.path.exists(data_path):
                os.remove(data_path)
            raise Exception(f"Downloaded file is not a valid parquet file: {e}")

    except requests.RequestException as e:
        raise Exception(f"Failed to download dataset: {e}")
    except Exception as e:
        raise Exception(f"Failed to save dataset: {e}")


def ensure_dataset_available(data_path: str = None) -> str:
    """
    Ensure dataset is available, downloading if necessary.

    Args:
        data_path: Optional custom path for dataset

    Returns:
        Path to the available dataset
    """
    if data_path is None:
        data_path = os.getenv("DATASET_PATH", "data/pipeline_data.parquet")

    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("Attempting automatic download...")
        try:
            return download_dataset(data_path)
        except Exception as e:
            print(f"Automatic download failed: {str(e)}")
            print("\n" + "="*60)
            print("MANUAL DOWNLOAD REQUIRED")
            print("="*60)
            print("Please manually download the dataset:")
            print("1. Visit: https://drive.google.com/file/d/109vhmnSLN3oofjFdyb58l_rUZRa0d6C8/view")
            print("2. Click 'Download' button")
            print("3. Save as 'pipeline_data.parquet'")
            print(f"4. Place the file at: {os.path.abspath(data_path)}")
            print("5. Restart the agent")
            print("="*60)
            raise Exception(f"Dataset not available. Manual download required at: {os.path.abspath(data_path)}")

    return data_path


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/pipeline_data.parquet"
    try:
        result_path = download_dataset(path)
        print(f"Dataset ready at: {result_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
