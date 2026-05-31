import os
import json
import shutil 
import pandas as pd 
import random
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool

base_folder = './terra_incognita'

# https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/scripts/download.py#L185
domains = ["38", "46", "100", "43"]


# Manual train/test split
def manual_split(df, train_ratio=0.8):
    indices = list(df.index)
    random.shuffle(indices)
    split_point = int(len(indices) * train_ratio)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    return df.loc[train_indices], df.loc[test_indices]

# Function to copy a single file
def copy_file(row):
    src = row['src_path']
    split = row['split']
    loc_str = row['loc_str']
    cls_name = row['category']

    dst_path = os.path.join(base_folder, split, loc_str, cls_name)
    os.makedirs(dst_path, exist_ok=True)
    shutil.copy2(src, dst_path)

def process_split(annotations_file_list):
    stats = {}
    destination_folder = os.path.join(base_folder)

    images_folder = os.path.join(base_folder, "images", "eccv_18_all_images_sm/")
    
    include_categories = [
        "bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit",
        "raccoon", "squirrel"
    ]

    data = defaultdict(list)

    for annotations_file in annotations_file_list:
        annots = {}
        with open(annotations_file, "r") as f:
            annots = json.load(f)
            for k, v in annots.items():
                data[k].extend(v)

    category_dict = {}
    for item in data['categories']:
        category_dict[item['id']] = item['name']

    
    all_files = []

    for image in data['images']:
        image_location = str(image['location'])

        if image_location not in domains:
            continue
        
        loc_str = 'location_' + str(image_location)
        loc_folder = os.path.join(destination_folder,
                                'location_' + str(image_location) + '/')

        
        # os.makedirs(loc_folder, exist_ok=True)

        image_id = image['id']
        image_fname = image['file_name']


        # Iterate over all annotations 
        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                if image_location not in stats:
                    stats[image_location] = {}

                category = category_dict[annotation['category_id']]

                if category not in include_categories:
                    continue

                if category not in stats[image_location]:
                    stats[image_location][category] = 0
                else:
                    stats[image_location][category] += 1

                loc_cat_folder = os.path.join(loc_folder, category + '/')

                # os.makedirs(loc_cat_folder, exist_ok=True)
                
                dst_path = os.path.join(loc_cat_folder, image_fname)
                src_path = os.path.join(images_folder, image_fname)

                # shutil.copyfile(src_path, dst_path)
                all_files.append([dst_path, src_path, category, loc_str])
    
    df_files = pd.DataFrame(all_files, columns=['dst_path', 'src_path', 'category', 'loc_str'])

    print("Processing data for each localization...")
    for domain_str in df_files['loc_str'].unique():
        subdf = df_files[df_files.loc_str == domain_str].copy()

        # Split into train/test
        train_df, test_df = manual_split(subdf, train_ratio=0.8)
        
        train_df['split'] = 'train'
        test_df['split'] = 'test'
        

        num_cpus_to_use = 32 
        # Convert to list of dicts for multiprocessing
        file_rows = train_df.to_dict(orient='records')
        # Copy in parallel
        with Pool(num_cpus_to_use) as pool:
            pool.map(copy_file, file_rows)

        # Convert to list of dicts for multiprocessing
        file_rows = test_df.to_dict(orient='records')
        # Copy in parallel
        with Pool(num_cpus_to_use) as pool:
            pool.map(copy_file, file_rows)


def main():
    
    # Load annotations 
    cis_test_annotations_file = os.path.join(base_folder, "eccv_18_annotation_files/cis_test_annotations.json")
    cis_val_annotations_file =   os.path.join(base_folder, "eccv_18_annotation_files/cis_val_annotations.json")
    train_annotations_file =   os.path.join(base_folder, "eccv_18_annotation_files/train_annotations.json")
    trans_test_annotations_file =   os.path.join(base_folder, "eccv_18_annotation_files/trans_test_annotations.json")
    trans_val_annotations_file =   os.path.join(base_folder, "eccv_18_annotation_files/trans_val_annotations.json")


    train_val_annotations = [cis_val_annotations_file, 
                             train_annotations_file,  
                             trans_val_annotations_file,
                             cis_test_annotations_file,
                             trans_test_annotations_file
                             ]
    

    process_split(train_val_annotations)



if __name__ == "__main__":
    main()
