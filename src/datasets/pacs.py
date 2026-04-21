import os
import os.path
import json
import hashlib
from torch.utils.data import Dataset
import torch
from PIL import Image


class PACSFromList(Dataset):
    def __init__(self, list_file, root_dir, transform=None, target_transform=None):
        """
        Args:
            list_file (str): Path to the .txt file listing image paths.
            root_dir (str): Root directory containing PACS images.
            transform (callable, optional): Transform to apply to images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths = []
        self.labels = []

        with open(list_file, 'r') as f:
            for line in f:
                img_path, label_idx = line.strip().split(' ')

                
                full_path = os.path.join(root_dir, img_path)
                class_idx = int(label_idx) - 1 
                self.image_paths.append(full_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.transform
        return image, label


class _PACS:
    DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
    CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    
    
    @classmethod
    def get_complementary_domains(cls, domains):
        return [x for x in cls.DOMAINS if x not in domains]
    
    @classmethod
    def get_complementary_classes(cls, classes):
        return [x for x in cls.CLASSES if x not in classes]

    @classmethod    
    def get_md5(cls, subset_config):
        return hashlib.md5(json.dumps(subset_config).encode('utf-8')).hexdigest()
    
    def __init__(self, root, train=True, transform=None, target_transform=None, subset_config=None, download=False):        
        self.root = os.path.abspath(os.path.expanduser(root))
        self.fpath = os.path.join(os.path.abspath(root),
                                  'PACS',
                                  )

        print(f"PACS SUBSET CONFIG: {subset_config=}")
        if not subset_config:
            subset_config = {'domains': self.DOMAINS, 'classes': self.CLASSES}
        assert 'domains' in subset_config and 'classes' in subset_config    
        # print(f"Using subset config: {subset_config}")
        
        if len(subset_config['domains']) > 1:
            print(f"Dataset got no. of domains > 1. Loading the subset only for the first domain from the list: {subset_config['domains']}")
            # raise ValueError("Invalid nubmer of domains")
        
        id = self.get_md5(subset_config)
        
        subset = 'train' if train else 'test'
        filename = f"{subset_config['domains'][0]}_{subset}_kfold.txt"
        self.data = PACSFromList(os.path.join(self.fpath, filename), 
                            os.path.join(self.fpath, 'kfold'),
                            transform,
                            target_transform,
                            )
        
        self.data.class_to_idx = {name: idx for (idx, name) in enumerate(self.CLASSES)}


class PACS:
    BASE_CLASS = _PACS
    default_domain_order = [0, 1, 2, 3]

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 subset_config=None):
        self.train_dataset = _PACS(
            location,
            train=True,
            transform=preprocess,
            target_transform=None,
            download=False,
            subset_config=subset_config,
        ).data

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = _PACS(
            location,
            train=False,
            transform=preprocess,
            target_transform=None,
            download=False,
            subset_config=subset_config,
        ).data
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.classnames = [c.replace('_', ' ') for c in list(self.train_dataset.class_to_idx.keys())]
        