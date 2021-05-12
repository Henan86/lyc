from transformers import (BertModel,
                          BertTokenizer,
                          AutoModel,
                          AutoTokenizer,
                          PreTrainedModel)
from datasets import load_dataset, Dataset as hfds
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .model import MODELS, WAPPERS
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
import os
import pickle
from .data import get_dataloader


def get_model(model_class, model_name, **kwargs):
    """
    该方法载入模型。

    Args:
        model_class: 模型对象，支持lyc.model中定义的模型和huggingface支持的模型。
        model_name: base model 的 name 或 path
        cache_dir: （optional） from_pretrained 方法需要的参数
    
    Return：
        model: model object

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
    """
        Args:
            tokenizer_name: name or path
            cache_dir=None:
            is_zh=None:
    """

    if is_zh:
        tokenizer=BertTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, **kwargs)
    else:
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, **kwargs)
    return tokenizer

def get_vectors(model, tokenized_sents):
    """
    使用model.SentenceEmbeddingModel获取句向量。

    Args:
        model:
        tokenized_sents: BatchedEncoding
    
    Return:
        Tensor([batch_size, embedding_dim])

    """
    ds=hfds.from_dict(tokenized_sents)
    dl=get_dataloader(ds, cols=['input_ids', 'attention_mask', 'token_type_ids'])
    a_results=[]
    for batch in tqdm(dl):
        if torch.cuda.is_available():
            batch=[to_gpu(i) for i in batch]
        output = model(**batch)
        a_embedding = output
        a_results.append(a_embedding)
    output=torch.cat(a_results)
    return output

def to_gpu(inputs):
    """
    to_gpu

    Args:
        inputs: Tensor / dict([Tensor])
    """
    if isinstance(inputs, dict):
        return {
            k:v.to('cuda') for k,v in inputs.items()
        }
    else:
        return inputs.to('cuda')
        
def compute_kernel_bias(vecs):
    """
    BertWhitening 计算SVD需要的kernel和bias
    """

    vecs=np.concatenate(vecs)
    mean=vecs.mean(axis=0, keepdims=True)
    cov=np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mean

def transform_and_normalize(vecs, kernel, bias):
    """
    BertWhitening 白化向量
    """
    vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

def get_optimizer_and_schedule(model, num_training_steps=None, num_warmup_steps=3000):
    """
    获取optimizer和schedule
    """

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
    """
    BertWhitening 保存SVD需要的kernel和bias
    """
    np.save(os.path.join(model_path, 'kernel.npy'), kernel)
    np.save(os.path.join(model_path, 'bias.npy'), bias)
    
    print(f'Kernal and bias saved in {os.path.join(model_path, "kernel.npy")} and {os.path.join(model_path, "bias.npy")}')

def vector_l2_normlize(vecs):
    return vecs/np.sqrt((vecs**2).sum(axis=1, keepdims=True))
