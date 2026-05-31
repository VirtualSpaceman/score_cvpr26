import os 
from tqdm import tqdm 
import pandas as pd

# https://github.com/hendrycks/imagenet-r
root = '/experimentos/datasets/imagenet-r'


def process_dataset(root):
    label_mapping_path = '/experimentos/datasets/imagenet-r-labels-mapping.txt'
    label_mapping = pd.read_csv(label_mapping_path, sep=' ', names=['label_id', 'label_name'])
    label_mapping = {label_id: label_name for (label_id, label_name) in 
                     zip(label_mapping['label_id'], label_mapping['label_name'])}
    

    for split_str in ['train', 'test']:
        path_split_load = os.path.join(root, split_str)
        all_label_ids = os.listdir(path_split_load)

        for label_id in tqdm(all_label_ids):
            source_path = os.path.join(path_split_load, label_id)
            dst_path = os.path.join(path_split_load, label_mapping[label_id])
            os.rename(source_path, dst_path)
            # print(source_path, dst_path)
        # print(all_label_ids)



if __name__ == "__main__":
    process_dataset(root)