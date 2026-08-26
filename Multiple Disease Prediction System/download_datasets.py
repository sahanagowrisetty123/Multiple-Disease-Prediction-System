import os
import urllib.request

def download_datasets():
    """
    Downloads datasets from public GitHub repositories for the Multiple Disease Prediction System.
    """
    # Create the datasets folder if it doesn't exist
    os.makedirs('datasets', exist_ok=True)
    
    # Dictionary mapping filenames to their public GitHub raw URLs
    # These URLs point to standard datasets typically used in ML tutorials
    url_dict = {
        'diabetes.csv': 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv',
        'heart.csv': 'https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv',
        'parkinsons.csv': 'https://raw.githubusercontent.com/chaitanyabaranwal/ParkinsonAnalysis/master/parkinsons.csv'
    }
    
    for filename, url in url_dict.items():
        filepath = os.path.join('datasets', filename)
        print(f"Downloading {filename}...")
        
        try:
            # Download the file
            urllib.request.urlretrieve(url, filepath)
            print(f"Successfully downloaded and saved to {filepath}")
            
        except Exception as e:
            print(f"Failed to download {filename}. Error: {e}")
            
    print("\nAll downloads finished! You can now run 'python train_models.py'")

if __name__ == "__main__":
    print("------------------------------------------")
    print("Starting Dataset Downloader...")
    print("------------------------------------------\n")
    download_datasets()
