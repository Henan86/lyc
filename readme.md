# 框架结构

- hfds_scripts: huggingface datasets 载入scripts，为`.py`文件。
  - 该scripts为处理原数据集的脚本，换句话说，每个数据文件夹/文件对应一个scripts。某些模型的输入数据并不是原封不动的使用原始数据，这时需要在模型的`train.py`中对数据集进行操作：重写`Dataset`对象。
- `data.py`:处理数据方面的方法和对象。
  - `get_tokenized_ds`获取分词后的dataset。
  - `get_dataloader`获取dataloader
  - 常用的`torch Dataset`对象。
- `utils.py`:所有可能有用的工具。
  - `get_tokenizer`, `get_model`, `to_gpu`, `get_optimizer_and_schedule`
- TODO: `train_loop.py`
- TODO: `eval_loop.py`
  - `eval`需要针对不同task解决
  - [ ] `classification_eval`
  - [ ] `text_generation_eval`
  - [ ] `multi-choice_eval`
  - [ ] `span-selection_eval`
  - [ ] `labeling_eval`
- TODO: `trainer.py`: 需要对`transformers.trainer`进行包装
- TODO: 评估是否需要对`pytorch_lightning`进行高级封装，包括`model` and `trainer`.