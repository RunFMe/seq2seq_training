from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import AutoTokenizer  
from datasets import Dataset
import numpy as np
from datasets import load_metric
import wandb

tokenizer= None

@dataclass
class DatasetConfig:
    path: str
    tokenizer: str
    max_words: int = 15
    shuffle: bool = True

    max_rows: int = 9999999999999999
    seed: int = 42
    val_frac: float = 0.1


def _parse_row(row):
    source, target, *_ = row.split('\t')
    return source, target

def _is_below_max_len(dataset_cfg, pair):
    source, target = pair
    return len(source.split(' ')) <= dataset_cfg.max_words and \
        len(target.split(' ')) <= dataset_cfg.max_words

def get_dataset(dataset_cfg: DatasetConfig):
    # read and filter rows
    pairs = []
    with open(dataset_cfg.path, 'r') as f:
        for i, line in zip(range(dataset_cfg.max_rows), f):
            if i >= dataset_cfg.max_rows:
                break
            try:
                pair = _parse_row(line)
            except Exception as e:
                print(e)
            if _is_below_max_len(dataset_cfg, pair):
                pairs.append(pair)
    # To deduplicate pairs of translations turn them into dict and to pairs again
    pairs = dict(pairs).items()
    
    return Dataset.from_dict({
        'source': [p[0] for p in pairs],
        'target': [p[1] for p in pairs]
    })
    
def get_tokenizer(dataset_cfg: DatasetConfig):
    tokenizer = AutoTokenizer.from_pretrained(dataset_cfg.tokenizer)
    return tokenizer


def preprocess_function(pairs, add_eos=False):
    inputs = pairs['source']
    targets = pairs['target']

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels['input_ids']
    
    if add_eos:
        for row in model_inputs["labels"]:
            row.append(tokenizer.eos_token_id)
    return model_inputs

metric = load_metric("sacrebleu")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
    result = {"bleu": result["score"]}
    return result



def collate_fn(samples):
    for s in samples:
        s.pop('source')
        s.pop('target')
        
    labels = [feature["labels"] for feature in samples] if "labels" in samples[0].keys() else None
    # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
    # same length to return tensors.
    if labels is not None:
        max_label_length = max(len(l) for l in labels)
        padding_side = tokenizer.padding_side
        for feature in samples:
            remainder = [tokenizer.pad_token_id] * (max_label_length - len(feature["labels"]))
            if isinstance(feature["labels"], list):
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
            elif padding_side == "right":
                feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
            else:
                feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

    
    return tokenizer.pad(
            samples,
            padding='longest',
            max_length=128,
            return_tensors='pt')