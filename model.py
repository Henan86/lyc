import torch.nn as nn
from transformers import (
    BertModel,
    AutoModel,
)

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


MODELS={
    'sentence-embedding': SentenceEmbeddingModel,
}
WAPPERS=[
    SentenceEmbeddingModel,
]