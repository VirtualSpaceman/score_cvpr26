import os
import json
import tqdm
import torch

from src import utils
from src.datasets.common import get_dataloader, maybe_dictionarize, get_concat_dataloader
from src.heads import get_classification_head
from src.modeling import ImageClassifier
from src.datasets.registry import get_dataset
from sklearn.metrics import balanced_accuracy_score


def eval_given_dataset(image_encoder, dataset, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name, classnames=dataset.classnames)
    model = ImageClassifier(image_encoder, classification_head)
    dataloader = dataset.test_loader
    metrics = do_eval(model, dataloader, args.device)
    print(f"Done evaluating on {dataset_name}. Accuracy: {metrics['top1']:.4f}")
    
    return metrics

def eval_given_dataset_concat(image_encoder, dataset, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name, classnames=dataset.classnames)
    model = ImageClassifier(image_encoder, classification_head)
    dataloader = get_concat_dataloader(dataset=dataset, args=args)
    metrics = do_eval(model, dataloader, args.device)
    print(f"Done evaluating on {dataset_name}. Accuracy: {metrics['top1']:.4f}")
    
    return metrics



def eval_single_dataset(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)

    metrics = do_eval(model, dataloader, args.device)
    
    print(f"Done evaluating on {dataset_name}. Accuracy: {metrics['top1']:.4f}")
    
    return metrics



@torch.no_grad()
def do_eval(model, dl, device):    
    correct, n = 0., 0.

    all_y = []
    all_preds = []
    model.eval()
    for data in tqdm.tqdm(dl):
        data = maybe_dictionarize(data)
        x = data['images'].to(device)
        y = data['labels'].to(device)
        all_y.extend(y.cpu())

        logits = utils.get_logits(x, model)
        pred = logits.argmax(dim=1, keepdim=True).to(device)
        all_preds.extend(pred.view(-1).cpu())
        correct += pred.eq(y.view_as(pred)).sum().item()
        n += y.size(0)

    bal_acc = None
    try:
        bal_acc = balanced_accuracy_score(all_y, all_preds).item()
    except: 
        bal_acc = -1
    metrics = {'top1': correct / n,
                'bal_acc': bal_acc}
    
    return metrics


def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for dataset_name in args.eval_datasets:
        print('Evaluating on', dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info

@torch.no_grad()
def eval_get_features_or_logits(image_encoder, dataset, dataset_name, args, logits=False):
    # Get classification head
    classification_head = get_classification_head(args, dataset_name, classnames=dataset.classnames)
    
    # Set model 
    model = ImageClassifier(image_encoder, classification_head)
    model = model.to(args.device)

    # Test dataset dataloader 
    dataloader = dataset.test_loader

    feat_list = []
    y_list = []
    # Set model to evaluation mode
    model.eval()
    for data in tqdm.tqdm(dataloader):
        data = maybe_dictionarize(data)
        x = data['images'].to(args.device)
        y = data['labels'].to(args.device)

        if logits:
            x_feat = utils.get_logits(x, model)
        else: 
            x_feat = model.get_features(x)
        feat_list.append(x_feat)
        y_list.append(y)
    
    feat_list = torch.concat(feat_list)#.to('cpu')
    y_list = torch.concat(y_list)#.to('cpu')
    return feat_list, y_list