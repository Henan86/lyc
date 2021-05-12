from ..utils import get_model, get_tokenizer, get_vectors, transform_and_normalize
from ..model import SentenceEmbeddingModel
import torch
import os
import numpy as np

class BertWhitening(torch.nn.Module):
    def __init__(self, model_path, max_length=64, n_components=768, kernel_bias_path=None):
        super(BertWhitening, self).__init__()

        self.model = get_model(SentenceEmbeddingModel, model_path, cache_dir='../model_cache', pooling_type='first-last-average')
        self.tokenizer = get_tokenizer('bert-base-chinese', is_zh=True)
        self.max_length=max_length
        self._get_kernel_and_bias(model_path, kernel_bias_path)
        self.n_components=n_components
    
    def forward(self, sents):
        tokenized_sents=self.tokenizer(sents, max_length=self.max_length, padding=True, truncation=True)
        with torch.no_grad():
            vecs=get_vectors(self.model, tokenized_sents)
        vecs=vecs.cpu().numpy()

        if self.n_components:
            kernel, bias = self.kernel, self.bias

            kernel=kernel[:, :self.n_components]
            vecs=transform_and_normalize(vecs, kernel, bias)

        return vecs
    
    def _get_kernel_and_bias(self, model_path, kernel_bias_path):
        if not os.path.exists(model_path):
            assert kernel_bias_path is not None
        else:
            kernel_bias_path=model_path
        
        self.kernel=np.load(os.path.join(kernel_bias_path, 'kernel.npy'))
        self.bias=np.load(os.path.join(kernel_bias_path, 'bias.npy'))


if __name__=='__main__':
    demo=BertWhitening('bert-base-chinese', kernel_bias_path='kernel_path/')
    print(demo.get_embeddings(['我不知道', '我是猪']))