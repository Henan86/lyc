# 框架结构

- hfds_scripts: huggingface datasets 载入scripts，为`.py`文件。
  - 该scripts为处理原数据集的脚本，换句话说，每个数据文件夹/文件对应一个scripts。某些模型的输入数据并不是原封不动的使用原始数据，这时需要在模型的`train.py`中对数据集进行操作：重写`Dataset`对象。
- `data.py`:处理数据方面的方法和对象。
  - `get_tokenized_ds`获取分词后的dataset。
  - `get_dataloader`获取dataloader
  - 常用的`torch Dataset`对象。
- `utils.py`:所有可能有用的工具。
  - `get_tokenizer`, `get_model`, `to_gpu`, `get_optimizer_and_schedule`


## `train_loop.py`
- [x] `train_step`
- [ ] `train_loop` 不要`eval`的单纯训练循环
- [x] `trainer`对象
  - 自定义的`trainer`对象需要实现：
  - [x] 带`eval`的`train_func`
    - 传入一个`eval_func`, which takes `logits/preds` and a `logger` as inputs, computing insides, returns nothing.
  - [x] `logger`
  - [x] `save`

## `eval_loop.py`
已支持的`metrics_computing`方法
- [x] `accuracy`


`eval`可能需要针对不同task解决
- [x] `general_eval` 最简单的直接使用`y_pred`和`y_true`算`metrics`.
- [ ] `classification_eval`
- [ ] `text_generation_eval`
- [ ] `multi-choice_eval`
- [ ] `span-selection_eval`
- [ ] `labeling_eval`
- TODO: `trainer.py`: 需要对`transformers.trainer`进行包装
- TODO: 评估是否需要对`pytorch_lightning`进行高级封装，包括`model` and `trainer`.


## 测试

- [ ] `train_loop.trainer`


## `model.py`

对常用的模型进行包装并统一接口，目前主要就是`huggingface`的模型。还有在此基础上各种衍生模型。

- [ ] BERT
- [ ] SentenceEmbeddingModel
- [ ] AutoModel
- [ ] AutoModelFor ...

## `trainer.py`

包装`pytorch_lightning`, `pytorch` and `huggingface`的`trainer`对象。

- [ ] `pytorch`
- [ ] `huggingface`
- [ ] `pytorch_lightning`
