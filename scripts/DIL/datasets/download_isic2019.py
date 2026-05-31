import argparse
import os
import sys
import json
import pandas as pd
import zipfile

# URLs for dataset components
url_1 = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
url_2 = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv"
url_3 = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-folder",
    type=str,
    help="Where to store raw images, preprocessed images, ground truth, metadata, model",
    required=True,
)
args = parser.parse_args()

# License acceptance (placeholder)
print("Please ensure you have accepted the license at https://challenge.isic-archive.com/data/")

# Create output folder and config
os.makedirs(args.output_folder, exist_ok=True)
data_directory = os.path.join(args.output_folder, "fed_isic2019")
os.makedirs(data_directory, exist_ok=True)
config_file = os.path.join(data_directory, "config.json")

# Check if already downloaded
if os.path.exists(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    if config.get("download_complete"):
        print("You already have downloaded the dataset. Aborting.")
        sys.exit()
else:
    config = {}

# File paths
dest_file_1 = os.path.join(data_directory, "ISIC_2019_Training_Input.zip")
dest_file_2 = os.path.join(data_directory, "ISIC_2019_Training_Metadata.csv")
dest_file_3 = os.path.join(data_directory, "ISIC_2019_Training_GroundTruth.csv")
dest_file_4 = os.path.join(data_directory, "ISIC_2019_Training_Metadata_FL.csv")
# parent_script_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
file1 = os.path.join(data_directory, "HAM10000_metadata.csv")

# Download and unzip data
os.system(f"curl -L -o {dest_file_1} {url_1}")
if zipfile.is_zipfile(dest_file_1):
    print("Zip file downloaded correctly")
    os.system(f"unzip {dest_file_1} -d {data_directory}")
    os.remove(dest_file_1)
else:
    sys.exit("Zip file corrupted")
os.system(f"curl -L -o {dest_file_2} {url_2}")
os.system(f"curl -L -o {dest_file_3} {url_3}")

# Load data
ISIC_2019_Training_Metadata = pd.read_csv(dest_file_2)
ISIC_2019_Training_GroundTruth = pd.read_csv(dest_file_3)
HAM10000_metadata = pd.read_csv(file1)
HAM10000_metadata.rename(columns={"image_id": "image"}, inplace=True)
HAM10000_metadata.drop(
    ["age", "sex", "localization", "lesion_id", "dx", "dx_type"], axis=1, inplace=True
)

# Remove entries with missing lesion_id
to_drop = ISIC_2019_Training_Metadata[ISIC_2019_Training_Metadata["lesion_id"].isnull()]
for image in to_drop["image"]:
    image_path = os.path.join(data_directory, "ISIC_2019_Training_Input", f"{image}.jpg")
    if os.path.exists(image_path):
        os.remove(image_path)
drop_indices = to_drop.index
ISIC_2019_Training_Metadata.drop(index=drop_indices, inplace=True)
ISIC_2019_Training_GroundTruth.drop(index=drop_indices, inplace=True)

# Generate dataset field
ISIC_2019_Training_Metadata["dataset"] = ISIC_2019_Training_Metadata["lesion_id"].str[:4]

# Merge with HAM10000 metadata
result = pd.merge(ISIC_2019_Training_Metadata, HAM10000_metadata, how="left", on="image")
result["dataset"] = result["dataset_x"] + result["dataset_y"].astype(str)
result.drop(["dataset_x", "dataset_y", "lesion_id"], axis=1, inplace=True)

# Print stats
print("Datacenters")
print(result["dataset"].value_counts())
print("Number of lines in Metadata", ISIC_2019_Training_Metadata.shape[0])
print("Number of lines in GroundTruth", ISIC_2019_Training_GroundTruth.shape[0])
print("Number of lines in MetadataFL", result.shape[0])
DIR = os.path.join(data_directory, "ISIC_2019_Training_Input")
N = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) - 2
print("Number of images", N)

# Save processed files
result.to_csv(dest_file_4, index=False)
ISIC_2019_Training_Metadata.to_csv(dest_file_2, index=False)
ISIC_2019_Training_GroundTruth.to_csv(dest_file_3, index=False)

# Update config
config["download_complete"] = N == 23247
with open(config_file, "w") as f:
    json.dump(config, f)

if config["download_complete"]:
    print("Download OK")
else:
    print("Something wrong happened during the download.")
