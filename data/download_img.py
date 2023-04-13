# This will download a dataset of images of about ~ 12 Gb. 
# Make sure you have enough storage space. This will also take a while to download.
# It will create the P1_Data folder at the given path.
# It reads the data frame with the URLs, file names & corresponding folder name from 
# the given path.
# Within the P1_Data folder, for each class it will create a corresponding folder.

# Usage example for Linux:
"""
cd ~/Nextcloud/paper_01/PAI/data # (edit path according to your local setup)

python3 download_img.py \
--df_path "./data_tables/img_url.txt" \
--img_dir "./P1_Data"

# For the Syrphidae test set:
python3 download_img.py \
--df_path "./data_tables/syrphid_img_url.txt" \
--img_dir "./syrphidae"
"""


import argparse
import requests
import time
import os
import pandas as pd
from tqdm import tqdm


def download_and_save_image(url: str, save_path: str):
    """
    Download an image from a given URL and save it to the disk.

    Parameters:
    - `url`: the URL of the image.
    - `save_path`: the path to save the image to, including the file name + extension.
    """

    # Fetch the image from the provided URL (download in chunks)
    response = requests.get(url, stream=True, timeout=10)
    
    # Check if the response is successful. 
    # If yes, download and save the image to the disk.
    # If not, raise an exception.
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            # Iterate over the response data using 1KB chunks
            for chunk in response.iter_content(chunk_size=1024):
                if chunk: # write only alive chunks (if chunk contains any data)
                    f.write(chunk)
    else:
        # If the response was not successful, raise an exception
        response.raise_for_status()


def main(df_path, img_dir):
    # Get the data frame path from the command line, then read it.
    df_data = pd.read_csv(df_path, sep='\t', dtype=str, encoding="UTF-8")

    # Group the data by the 'folder' column
    df_data_grouped = df_data.groupby('folder')

    # Iterate through the grouped data
    for folder, df in df_data_grouped:
        # Create a folder for the current class
        folder_path = os.path.join(img_dir, folder, 'img')
        os.makedirs(folder_path, exist_ok=True)

        # Iterate through the rows in the current class
        for index, row in tqdm(df.iterrows(), 
                               desc=f"Downloading images for {folder}", 
                               total=df.shape[0], 
                               unit="image"):
            # Get the URL and the file name
            url = row['url']
            file_name = row['filename']

            # Create the path to save the image to
            save_path = os.path.join(folder_path, file_name)

            # Check if the file already exists; if it does, skip downloading it.
            # This is useful if you want to resume the download after it was interrupted.
            if os.path.exists(save_path):
                print(f'Skipping {file_name} - already downloaded.')
                continue

            # Download and save the image
            try:
                download_and_save_image(url, save_path)
            except Exception as e:
                print(f'Error downloading file {file_name} from {url}, at row {index}: {e}')


if __name__ == '__main__':
    # Time the execution
    start = time.perf_counter()

    # Parse the command line arguments.
    # Create a parser to get the arguments from the command line.
    parser = argparse.ArgumentParser(
        description="Download images from a data frame with the URLs and file names.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--df_path", type=str, help="./data_tables/img_url.txt")
    parser.add_argument("--img_dir", type=str, help="./P1_Data")
    args = parser.parse_args()

    main(args.df_path, args.img_dir)

    finish = time.perf_counter()
    run_time = round(finish-start, 2)
    print(f'Finished in {run_time} seconds.')