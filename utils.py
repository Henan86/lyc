from transformers import (BertModel,
                          BertTokenizer,
                          AutoModel,
                          AutoTokenizer,
                          PreTrainedModel)
from datasets import load_dataset, Dataset as hfds
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from model import MODELS, WAPPERS
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
import os
import pickle


def get_model(model_class, model_name, **kwargs):
    """
    该方法载入模型。

    Args:
        model_class: 

    """
    if model_class in WAPPERS:
        model=model_class(model_name, **kwargs)
    elif issubclass(model_class, PreTrainedModel):
        model=model_class.from_pretrained(model_name, **kwargs)
    else:
        raise ValueError()

    if torch.cuda.is_available():
        model.to('cuda')
        if torch.cuda.device_count()>1:
            model=torch.nn.DataParallel(model)
    return model

def get_tokenizer(tokenizer_name, cache_dir=None, is_zh=None, **kwargs):
    if is_zh:
        tokenizer=BertTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, **kwargs)
    else:
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, **kwargs)
    return tokenizer

def to_gpu(inputs):
    if isinstance(inputs, dict):
        return {
            k:v.to('cuda') for k,v in inputs.items()
        }
    else:
        return inputs.to('cuda')

class SentencePairDataset(Dataset):
    def __init__(self, tokenized_a, tokenized_b=None, label=None):
        self.tokenized_a=tokenized_a
        self.tokenized_b=tokenized_b
        self.label=label

    def __len__(self):
        return self.tokenized_a['input_ids'].shape[0]

    def __getitem__(self, index):
        input_a = {
            k:v[index] for k,v in self.tokenized_a.items()
        }
        output=(input_a, )
        if self.tokenized_b is not None:
            input_b={
                k:v[index] for k,v in self.tokenized_b.items()
            }
            output+=(input_b, )
        if self.label is not None:
            output+=(torch.LongTensor([self.label[index]]), )
        return output

def get_dataloader(tokenized_a, tokenized_b=None, batch_size=16, label=None):
    ds=SentencePairDataset(tokenized_a, tokenized_b, label=label)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    return dl
        
def compute_kernel_bias(vecs):
    vecs=np.concatenate(vecs)
    mean=vecs.mean(axis=0, keepdims=True)
    cov=np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mean

def transform_and_normalize(vecs, kernel, bias):
    vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

def get_optimizer_and_schedule(model, num_training_steps=None, num_warmup_steps=3000):
    # params=[{'params': [param for name, param in model.named_parameters() if 'sbert' not in name], 'lr': 5e-5},
    # {'params': [param for name, param in model.named_parameters() if 'sbert' in name], 'lr': 1e-3}]
    
    optimizer=AdamW(model.parameters())

    if num_training_steps is None:
        return optimizer

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    lr_schedule=LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    return optimizer, lr_schedule

def eval(model, tokenizer, ds='atec', n_components=768):
    model.eval()
    input_a, input_b, label = get_tokenized_ds(datasets_paths[ds]['scripts'], datasets_paths[ds]['data_path'], tokenizer, ds)

    with torch.no_grad():
        a_vecs, b_vecs = get_vectors(model, input_a, input_b)
    a_vecs=a_vecs.cpu().numpy()
    b_vecs=b_vecs.cpu().numpy()
    if n_components:
        kernel, bias = compute_kernel_bias([a_vecs, b_vecs])

        kernel=kernel[:, :n_components]
        a_vecs=transform_and_normalize(a_vecs, kernel, bias)
        b_vecs=transform_and_normalize(b_vecs, kernel, bias)
        sims=(a_vecs * b_vecs).sum(axis=1)
    else:
        sims=(a_vecs * b_vecs).sum(axis=1)

    return accuracy_score(sims>0.5, label)

def save_kernel_and_bias(kernel, bias, model_path):
    np.save(os.path.join(model_path, 'kernel.npy'), kernel)
    np.save(os.path.join(model_path, 'bias.npy'), bias)
    
    print(f'Kernal and bias saved in {os.path.join(model_path, "kernel.npy")} and {os.path.join(model_path, "bias.npy")}')