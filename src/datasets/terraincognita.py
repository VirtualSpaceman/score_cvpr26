import os
import os.path
import json
import hashlib

import torch
from torchvision import datasets
# https://lila.science/datasets/caltech-camera-traps


class _TerraIncognita:
    DOMAINS = ["location_38", "location_46", "location_100", "location_43"]
    CLASSES = ["bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit", "raccoon", "squirrel"]

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
        self.fpath = os.path.join(os.path.abspath(root),
                                  'terra_incognita',
                                  'train' if train else 'test')
        
        if not subset_config:
            subset_config = {'domains': self.DOMAINS, 'classes': self.CLASSES}
        assert 'domains' in subset_config and 'classes' in subset_config    
        
        id = self.get_md5(subset_config)
        new_fpath = os.path.join(os.path.abspath(root),
                                    'terra_incognita_subsets',
                                    id,
                                    'train' if train else 'test')
        if not os.path.exists(new_fpath):
            print("Creating a subset from scratch")
            for domain in subset_config['domains']:
                for cls in subset_config['classes']:                    
                    source_cls_path = os.path.join(self.fpath, domain, cls)
                    if not os.path.exists(source_cls_path):
                        print(f"Path {source_cls_path} does not exist, skipping!")
                        continue
                    
                    new_cls_path = os.path.join(new_fpath, cls)
                    os.makedirs(new_cls_path, exist_ok=True)
                    
                    for img_name in os.listdir(source_cls_path):
                        os.symlink(os.path.join(source_cls_path, img_name),
                                   os.path.join(new_cls_path, img_name))

            with open(os.path.join(new_fpath, 'config.json'), 'w') as config_file:
                json.dump(subset_config, config_file)
        else:
            print("Using subset that already exists")
            
        self.fpath = new_fpath
        self.data = datasets.ImageFolder(self.fpath, transform=transform)


class TerraIncognita:
    BASE_CLASS = _TerraIncognita
    
    default_domain_order = [0, 1, 2, 3]
        
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 subset_config=None):
        self.train_dataset = _TerraIncognita(
            location,
            train=True,
            transform=preprocess,
            target_transform=None,
            subset_config=subset_config,
        ).data
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = _TerraIncognita(
            location,
            train=False,
            transform=preprocess,
            target_transform=None,
            subset_config=subset_config,
        ).data
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        self.classnames = [c.replace('_', ' ') for c in list(self.train_dataset.class_to_idx.keys())]
