from kaggle.api.kaggle_api_extended import KaggleApi 

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()  # Uses ~/.kaggle/kaggle.json by default

# List all files in the dataset
files = api.dataset_list_files('nikhilpandey360/chest-xray-masks-and-labels').files

# Specify the folders you want
folders = ['CXR_png/', 'masks/', 'test/']

# Download files from only those folders
for file in files:
    if any(file.name.startswith(folder) for folder in folders):
        api.dataset_download_file(
            'nikhilpandey360/chest-xray-masks-and-labels',
            file_name=file.name,
            path='desired_download_path',
            force=True
        )
