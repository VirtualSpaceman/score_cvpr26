

def get_dict_dataset_paths(location: str = None) -> dict:   

    if location:
        PREFFIX = location
    else:
        PREFFIX = '/hadatasets/levy/datasets/'

    paths = {
            'ImageNetR': PREFFIX,
            'DomainNet': PREFFIX,
            'OfficeHome': PREFFIX,
            'FedISIC': PREFFIX,
            'PACS': PREFFIX,
            'RetinaDomains': PREFFIX, 
            'NICOpp': PREFFIX,
            'TerraIncognita': PREFFIX,
        }
    return paths

def get_dict_epochs() -> dict:
    epochs = {
        'DomainNet': 10,
        'OfficeHome': 10,
        'ImageNetR': 10,   
        'PACS': 10,  
        'FedISIC': 20,
        'RetinaDomains': 20,   
        'NICOpp': 15, 
        'TerraIncognita': 15, 
    }
        
    return epochs
