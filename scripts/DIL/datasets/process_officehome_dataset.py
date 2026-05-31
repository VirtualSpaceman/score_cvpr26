import os
from tqdm import tqdm
import torch
from shutil import copy2
from multiprocessing import Pool
from torchvision import datasets 

# https://www.hemanthdv.org/officeHomeDataset.html
officehome_folder = './OfficeHomeDataset'
domains = ['Art', 'Clipart', 'Product', 'Real World']
SPLIT_SEED = 17

def process_row(args):
    """Process a single row from the dataset."""
    image_path, split_name = args
    
    # Get both class name and domain name
    class_name = image_path.split('/')[-2].strip().replace(' ', '_')
    domain_name = image_path.split('/')[-3].strip().replace(' ', '_')
    
    # Set-up new path 
    new_folder_path = os.path.join(officehome_folder, split_name, domain_name, class_name) 
    os.makedirs(new_folder_path, exist_ok=True)
     
    # get image filename  
    filename = image_path.split('/')[-1]

    # Check if the file exists
    if os.path.exists(os.path.join(new_folder_path, filename)):
        return 

    # otherwise copy to the new path 
    copy2(image_path, new_folder_path)


def process_split(domain):
    """Process a specific domain and split."""
    
    fpath = os.path.join(officehome_folder, domain)
    
    dataset = datasets.ImageFolder(fpath, transform=None)
            
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train, val = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    )
    train_idx, val_idx = train.indices, val.indices

    train_file_list = [(dataset.imgs[i][0], 'train') for i in train_idx]
    test_file_list = [(dataset.imgs[i][0], 'test') for i in val_idx]
    
    
    # Use multiprocessing pool to parallelize the processing (processing train)
    num_cpus_to_use = min(os.cpu_count(), 32) 
    with Pool(num_cpus_to_use) as pool:
        list(tqdm(pool.imap(process_row, train_file_list), total=len(train_file_list), desc=f"{domain}-train"))

    # processing test 
    num_cpus_to_use = min(os.cpu_count(), 32) 
    with Pool(num_cpus_to_use) as pool:
        list(tqdm(pool.imap(process_row, test_file_list), total=len(test_file_list), desc=f"{domain}-test"))


def main():
    for domain in domains:
        print(f"Processing domain: {domain}..")
        process_split(domain)

if __name__ == "__main__":
    main()
