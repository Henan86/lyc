# 框架结构

- hfds_scripts: huggingface datasets 载入scripts，为`.py`文件。
  - 该scripts为处理原数据集的脚本，换句话说，每个数据文件夹/文件对应一个scripts。某些模型的输入数据并不是原封不动的使用原始数据，这时需要在模型的`train.py`中对数据集进行操作：重写`Dataset`对象。
- `data.py`:处理数据方面的方法和对象。
  - `get_tokenized_ds`获取分词后的dataset。
  - `get_dataloader`获取dataloader
  - 常用的`torch Dataset`对象。
- `utils.py`:所有可能有用的工具。
  - `get_tokenizer`, `get_model`, `to_gpu`, `get_optimizer_and_schedule`


## `train.py`
- [x] `train_step`
- [x] `train_loop` 不要`eval`的单纯训练循环
- [x] `trainer`对象
  - 自定义的`trainer`对象需要实现：
  - [x] 带`eval`的`train_func`
    - 传入一个`eval_func`, which takes `logits/preds` and a `logger` as inputs, computing insides, returns nothing.
  - [x] `logger`
  - [x] `save`
- [x] argparser: 处理args的通用parser，包含了大多数常见参数
  - [x] 调用`train.show_all_args`查看所有可用args
  - [x] `get_general_args`直接获取所有常用args

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
- TODO: 评估是否需要对`pytorch_lightning`进行高级封装，包括`model` and `trainer`.


## 测试
- [ ] `train_loop.trainer`


## `model.py`
对常用的模型进行包装并统一接口，目前主要就是`huggingface`的模型。还有在此基础上各种衍生模型。

- [ ] BERT
- [x] SentenceEmbeddingModel
- [ ] AutoModel
- [ ] AutoModelFor ...

## 修改`transformers.Trainer`

### TODO

- [ ] Trainer.add_callback(
- [ ] Trainer.call_model_init(
- [ ] Trainer.compute_loss(
- [ ] Trainer.create_optimizer(
- [ ] Trainer.create_optimizer_and_scheduler(
- [ ] Trainer.create_scheduler(
- [ ] Trainer.evaluate(
- [ ] Trainer.floating_point_ops(
- [ ] Trainer.get_eval_dataloader(
- [ ] Trainer.get_test_dataloader(
- [ ] Trainer.get_train_dataloader(
- [ ] Trainer.hyperparameter_search(
- [ ] Trainer.is_local_process_zero(
- [ ] Trainer.is_world_process_zero(
- [ ] Trainer.log(
- [ ] Trainer.log_metrics(
- [ ] Trainer.metrics_format(
- [ ] Trainer.mro(
- [ ] Trainer.num_examples(
- [ ] Trainer.pop_callback(
- [ ] Trainer.predict(
- [ ] Trainer.prediction_loop(
- [ ] Trainer.prediction_step(
- [ ] Trainer.remove_callback(
- [ ] Trainer.save_metrics(
- [ ] Trainer.save_model(
- [ ] Trainer.save_state(
- [ ] Trainer.store_flos(
- [ ] Trainer.train(
- [ ] Trainer.training_step(

## `pipelines`

- [x] `BertWhitening`

## 支持的`Dataset`

- atec
- zh_wiki
- [x] en_wiki
- [ ] zh_csqa