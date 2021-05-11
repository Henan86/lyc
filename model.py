import torch.nn as nn
from transformers import (
    BertModel,
    AutoModel,
)
import torch
import torch.nn.functional as F

"""
Model 对象的开发规范。

Model有两种类型：继承huggingface PreTrainedModel；或者继承nn.Module。
这样方便utils.get_model方法载入模型。

init的参数规范。
huggingface-based：
    # TODO

nn.Module-based: 
    应在init中初始化一个huggingface base model

"""


class SentenceEmbeddingModel(nn.Module):

    def __init__(self, model_name_or_path, pooling_type=None, **kwargs):
        super(SentenceEmbeddingModel, self).__init__()
        self.pooling_type=pooling_type
        self.base=AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True, **kwargs)

        self.pooling_funcs={
            'cls': self._cls_pooling,
            'last-average': self._average_pooling,
            'first-last-average': self._first_last_average_pooling,
            'cls-pooler': self._cls_pooler_pooling
        }
    
    def forward(self, input_ids,
        attention_mask=None,
        token_type_ids=None,
        **kwargs):

        outputs=self.base(input_ids, attention_mask, token_type_ids, **kwargs)
        hidden=outputs.hidden_states
        pooler_outputs=outputs.pooler_output
        
        return self.pooling_funcs[self.pooling_type](hidden, attention_mask, pooler_outputs)
    
    def _cls_pooling(self, hidden, attention_mask, pooled_outputs):
        last_hidden_state=hidden[-1]
        return last_hidden_state[:, 0]

    def _cls_pooler_pooling(self, hidden, attention_mask, pooled_outputs):
        return pooled_outputs
    
    def _average_pooling(self, hidden, attention_mask, pooled_outputs):
        last_hidden_states=hidden[-1]
        last_hidden_states = torch.sum(
            last_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        return last_hidden_states
    
    def _first_last_average_pooling(self, hidden, attention_mask, pooled_outputs):
        first_hidden_states = hidden[1]
        last_hidden_states = hidden[-1]

        first_hidden_states = torch.sum(
            first_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        last_hidden_states = torch.sum(
            last_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        sentence_embedding=torch.mean(
            torch.stack([first_hidden_states, last_hidden_states]), dim=0
        )

        return sentence_embedding


class simcse(SentenceEmbeddingModel):

    def forward(self, *args, **kwargs):
        label = kwargs.pop('label')
        embeddings=super(simcse, self).forward(*args, **kwargs)
        loss = self.cce_losses(label, embeddings)

        return {'loss': loss}
    
    def nll_losses(self, label, embeddings):

        label=F.one_hot(label)
        normalized_embedding = embeddings/torch.sqrt((embeddings**2).sum(-1))[:, None]
        sims=torch.matmul(normalized_embedding, normalized_embedding.T)
        sims = sims - torch.eye(embeddings.shape[0])*100

        sims.clip_(0,1)
        loss = F.binary_cross_entropy(sims.view(-1), label.view(-1).float())
        
        return loss

    def cce_losses(self, label, embeddings):

        normalized_embedding = embeddings/torch.sqrt((embeddings**2).sum(-1))[:, None]
        sims=torch.matmul(normalized_embedding, normalized_embedding.T)
        sims=sims*20 - torch.eye(embeddings.shape[0])*1e12

        loss=F.cross_entropy(label, sims)
        return loss


MODELS={
    'sentence-embedding': SentenceEmbeddingModel,
}
WAPPERS=[
    SentenceEmbeddingModel,
]