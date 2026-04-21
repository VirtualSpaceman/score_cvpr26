import os
import json
import torch
import hashlib
import pandas as pd 
from PIL import Image
from torch.utils.data import ConcatDataset




class _FedISIC:
    DOMAINS = ['BCN', 'HAM_vidir_molemax', 'HAM_vidir_modern', 'HAM_rosendahl', 'MSK', 'HAM_vienna_dias']
    CLASSES = ['Melanoma', 'Melanocytic nevus',  
                'Basal cell carcinoma', 'Actinic keratosis', 
                'Benign keratosis', 'Dermatofibroma', 
                'Vascular lesion', 'Squamous cell carcinoma']

    @classmethod
    def get_complementary_domains(cls, domains):
        return [x for x in cls.DOMAINS if x not in domains]
    
    @classmethod
    def get_complementary_classes(cls, classes):
        return [x for x in cls.CLASSES if x not in classes]

    @classmethod    
    def get_md5(cls, subset_config):
        return hashlib.md5(json.dumps(subset_config).encode('utf-8')).hexdigest()

    def __init__(self, root, train=True, transform=None, target_transform=None, subset_config=None):
        self.base_folder = os.path.join(os.path.abspath(root), 
                                        'HAM10000', 
                                        'fed_isic2019')
        
        self.image_folder = os.path.join(self.base_folder, 'ISIC_2019_Training_Input')
        self.train = train 
        if not subset_config:
            subset_config = {'domains': self.DOMAINS, 'classes': self.CLASSES}
        assert 'domains' in subset_config and 'classes' in subset_config    
        
        # Load metadata containing train/test information 
        self.metadata = pd.read_csv(os.path.join(self.base_folder, 'train_test_split.csv'))
        self.fold = 'train' if self.train else 'test'
        # Filter train/test split 
        self.metadata = self.metadata[self.metadata.fold == self.fold]
        
        # Training data according center subsets
        if len(subset_config['domains']) == len(self.DOMAINS):
            self.center = None
        else:
            center_idx = self.DOMAINS.index(subset_config['domains'][0])
            if center_idx == -1:
                raise ValueError(f"Unknown Center from config {subset_config}")
            self.center = center_idx
            self.metadata = self.metadata[self.metadata.fold2 == f'{self.fold}_{self.center}']

        images = self.metadata.image.tolist()
        self.image_paths = [
            os.path.join(self.image_folder, image_name + ".jpg")
            for image_name in images
        ]
        self.targets = self.metadata.target.tolist()
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):

        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(self.targets[idx])

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, label 


class FedISIC:
    BASE_CLASS = _FedISIC
    default_class_order = [0, 1, 2, 3, 4, 5, 6, 7]
    default_domain_order = [0, 1, 2, 3, 4, 5]
        
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 subset_config=None):
        self.train_dataset = _FedISIC(
            location,
            train=True,
            transform=preprocess,
            target_transform=None,
            subset_config=subset_config,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = _FedISIC(
            location,
            train=False,
            transform=preprocess,
            target_transform=None,
            subset_config=subset_config,
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            # ConcatDataset([self.train_dataset, self.test_dataset]),
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        self.classnames = [c.replace('_', ' ') for c in self.train_dataset.CLASSES]


if __name__ == '__main__':
    # Test class 

    path = '/hadatasets/levy/datasets/'
    dataclass = FedISIC(preprocess=None,
                        location = path )
    
    
    domains = dataclass.BASE_CLASS.DOMAINS
    print(f"All domains: {domains}")

    for domain_idx, domain_name in enumerate(domains):
        print(f"Domain : {domain_name}")
        subset_config = {'domains': [domains[domain_idx]], 
                            'classes': dataclass.BASE_CLASS.CLASSES, 
                            }

        dataclass_subset = FedISIC(preprocess=None,
                        location = path,
                        subset_config=subset_config)
        
        print(f"Train with len: {len(dataclass_subset.train_dataset)}")
        print(f"Test with len: {len(dataclass_subset.test_dataset)}")
        x_train_sample = dataclass_subset.train_dataset[0]
        x_test_sample = dataclass_subset.test_dataset[0]
