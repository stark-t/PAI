# This script will create YOLO text annotation files from the provided
# annotation DataFrames.

# Usage example for Linux:
"""
cd ~/Nextcloud/paper_01/PAI/data # (edit path according to your local setup)

python3 make_yolo_txt_files.py \
--df_path "./data_tables/img_annotation.txt" \
--output_dir "./P1_Data"

# For the Syrphidae test set:
python3 make_yolo_txt_files.py \
--df_path "./data_tables/syrphid_img_annotation.txt" \
--output_dir "./syrphidae"
"""


import argparse
import time
import os
import pandas as pd


def create_yolo_txt_files(df, output_dir):
    # Group the DataFrame by folder
    grouped_df = df.groupby('folder')

    # Iterate through the groups
    for folder, group in grouped_df:
        # Create the folder for the current group if it doesn't exist
        folder_path = os.path.join(output_dir, folder, 'annotations', 'yolo_txt')
        os.makedirs(folder_path, exist_ok=True)

        # Group the current group by filename
        img_grouped_df = group.groupby('filename')

        # Iterate through the bounding boxes (rows) in the current file group
        for img_name, img_group in img_grouped_df:
            # Create a new text file named after the image (remove the file extension)
            name_without_ext = os.path.splitext(img_name)[0]
            txt_file_path = os.path.join(folder_path, f"{name_without_ext}.txt")

            with open(txt_file_path, 'w') as txt_file:
                # Iterate through the bounding boxes of the current image
                for _, row in img_group.iterrows():
                    # Extract the required information from the row
                    label_id = row['label_id']
                    x_center_rel = row['x_center_rel']
                    y_center_rel = row['y_center_rel']
                    width_rel = row['width_rel']
                    height_rel = row['height_rel']

                    # Write the bounding box information as a line in the text file
                    txt_file.write(f"{label_id} {x_center_rel} {y_center_rel} {width_rel} {height_rel}\n")


if __name__ == '__main__':
    # Time the execution
    start = time.perf_counter()

    # Parse the command line arguments.
    # Create a parser to get the arguments from the command line.
    parser = argparse.ArgumentParser(
        description="Create YOLO text files from a DataFrame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--df_path", type=str, help="./data_tables/img_annotation.txt")
    parser.add_argument("--output_dir", type=str, help="./P1_Data")
    args = parser.parse_args()

    # Read the DataFrame from a tab-delimited text file
    df_path = args.df_path
    df = pd.read_csv(df_path, sep='\t', dtype=str, encoding="UTF-8")
    # Filter out rows with used_for_ai == 0
    df = df[df['used_for_ai'] == "1"]

    # Create the YOLO text files
    create_yolo_txt_files(df, args.output_dir)

    finish = time.perf_counter()
    run_time = round(finish-start, 2)
    print(f'Finished in {run_time} seconds.')