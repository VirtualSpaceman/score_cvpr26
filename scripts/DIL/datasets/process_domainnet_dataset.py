import os
from tqdm import tqdm
import pandas as pd
from shutil import copy2
from multiprocessing import Pool, cpu_count

uda_folder = './UDA_datasets'
domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

def process_row(args):
    """Process a single row from the dataset."""
    row, domain, split = args
    uda_folder = './UDA_datasets'
    path = row['path']
    class_id = row['label']
    class_name = path.split('/')[1]
    new_folder_path = f'./domainnet/{split}/{domain}/{class_name}' 
    os.makedirs(new_folder_path, exist_ok=True)
    old_path = os.path.join(uda_folder, domain, path) 
    
    filename = path.split('/')[-1]

    # Check if the file exists
    if os.path.exists(os.path.join(new_folder_path, filename)):
        return 

    # otherwise copy to the new path 
    copy2(old_path, new_folder_path)


def process_split(domain, split):
    """Process a specific domain and split."""
    domain_path = os.path.join(uda_folder, domain)
    columns = ['path', 'label']
    
    txt_current = pd.read_csv(os.path.join(domain_path, f"{domain}_{split}.txt"), header=None, sep=' ') 
    txt_current.columns = columns
    
    # Prepare arguments for multiprocessing
    args = [(row, domain, split) for _, row in txt_current.iterrows()]
    
    # Use multiprocessing pool to parallelize the processing
    num_cpus_to_use = 32 
    with Pool(num_cpus_to_use) as pool:
        list(tqdm(pool.imap(process_row, args), total=len(args), desc=f"{domain}-{split}"))

def main():
    for domain in domains:
        for split in ['train', 'test']:
            print(f"Processing domain: {domain}, split: {split}")
            process_split(domain, split)

if __name__ == "__main__":
    main()
