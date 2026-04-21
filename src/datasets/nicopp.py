import os
import os.path
import json
import hashlib

import torch
from torchvision.datasets import ImageFolder



class _NICOpp:
    DOMAINS = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']
    CLASSES = {"car": 0, "flower": 1, "chair": 2, "truck": 3, "tiger": 4, "wheat": 5, "seal": 6, "wolf": 7, "lion": 8, "dolphin": 9, "lifeboat": 10, "corn": 11, "fishing rod": 12, "owl": 13, "sunflower": 14, "cow": 15, "bird": 16, "clock": 17, "shrimp": 18, "goose": 19, "airplane": 20, "rabbit": 21, "hot air balloon": 22, "lizard": 23, "hat": 24, "spider": 25, "motorcycle": 26, "tortoise": 27, "dog": 28, "crocodile": 29, "elephant": 30, "gun": 31, "fox": 32, "bus": 33, "cat": 34, "sailboat": 35, "giraffe": 36, "cactus": 37, "pumpkin": 38, "train": 39, "ship": 40, "helicopter": 41, "bicycle": 42, "racket": 43, "squirrel": 44, "bear": 45, "scooter": 46, "mailbox": 47, "horse": 48, "pineapple": 49, "frog": 50, "football": 51, "ostrich": 52, "tent": 53, "kangaroo": 54, "monkey": 55, "crab": 56, "sheep": 57, "butterfly": 58, "umbrella": 59}
    CLASSES = [cls_name.strip().replace(' ', '_') for cls_name in CLASSES.keys()]

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
                                  'NICOpp_track1',
                                  'train' if train else 'test')

        if not subset_config:
            subset_config = {'domains': self.DOMAINS, 'classes': self.CLASSES}
        assert 'domains' in subset_config and 'classes' in subset_config    
        # print(f"Using subset config: {subset_config}")
        
        id = self.get_md5(subset_config)
        new_fpath = os.path.join(os.path.abspath(root),
                                    'nicopp_subsets',
                                    id,
                                    'train' if train else 'test')
        # print(new_fpath)
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
        self.data = ImageFolder(self.fpath, transform=transform, allow_empty=True)



class NICOpp:
    BASE_CLASS = _NICOpp
    default_domain_order = [0, 1, 2, 3, 4, 5]

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 subset_config=None):
        self.train_dataset = _NICOpp(
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

        self.test_dataset = _NICOpp(
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
        