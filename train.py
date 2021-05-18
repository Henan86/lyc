import torch
from torch.utils.tensorboard import SummaryWriter
from utils import *
from tqdm import tqdm
import os
from dataclasses import dataclass
import argparse


def train_step(
    model, batch, optimizer, lr_schedule=None, clip_grad=None
):
    """
    进行一次forward + 一次backward + optim.step.

    Args:
        model: the outputs of the model are required to include loss.
        batch: Inputs 需要提前配置好device. 输入形式为dict
        optimizer
        lr_schedule: optional
        clip_grad: optional 切割梯度
    """

    outputs=model(**batch)
    if isinstance(model, torch.nn.DataParallel):
        loss=outputs.loss.mean()
    else:
        loss=outputs.loss
    
    optimizer.zero_grad()

    loss.backward()

    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

    optimizer.step()
    lr_schedule.step()

    return loss

def train_loop():
    pass


class LycTrainer:
    def __init__(
        self,
        model,
        train_dl,
        optimizer,
        num_epoch,
        lr_schedule = None,
        require_logger = None,
        log_path = None
   ):
        self.model=model
        self.train_dl=train_dl
        self.optimizer=optimizer
        self.num_epoch=num_epoch
        self.lr_schedule=lr_schedule

        self.num_train_step_per_epoch=len(self.train_dl)

        if require_logger and log_path:
            self.writer=SummaryWriter(log_path)
    
    def save_setting(self, save_type, save_steps = 500, save_path = 'checkpoints/'):
        assert save_type in ['torch', 'hf']
        self.save_method = save_type
        self.save_steps=save_steps
        self.save_path=save_path
        
    def eval_setting(self, eval_func, eval_dl, eval_steps=500):
        self.eval_func=eval_func
        self.eval_dl=eval_dl
        self.eval_steps=eval_steps
    
    def save(self, model, global_step):
        if self.save_method == 'torch':
            file_name = f'Step-{global_step}.bin'
            torch.save(model.state_dict(), os.path.join(self.save_path, file_name))
        elif self.save_method == 'hf':
            dir_name=f'Step-{global_step}/'
            model.save_pretrained(os.path.join(self.save_path, dir_name))

    def train(self):
        
        for epoch in range(self.num_epoch):
            for index, batch in enumerate(tqdm(self.train_dl, desc=f'Traing for epoch: {epoch}')):

                global_current_step = self.num_train_step_per_epoch * epoch + index
                loss=train_step(self.model, batch, self.optimizer, self.lr_schedule)
                if self.writer is not None:
                    self.writer.add_scalar('train_loss', loss, global_current_step)

                # eval
                if self.eval_func is not None and global_current_step % self.eval_steps == 0 and global_current_step !=0:
                    self.eval_func(self.model, self.eval_dl, self.writer, global_current_step)

                # save
                if self.save_method is not None and global_current_step % self.save_steps ==0 and global_current_step !=0:
                    if isinstance(self.model, torch.nn.DataParallel):
                        self.save(model.module, global_current_step)
                    else:
                        self.save(self.model, global_current_step)

@dataclass
class TrainingArgs:
    # must claim
    train_data : str = None
    save_path : str = None
    dataset_scripts : str = None
    # data
    eval_data_path : str = None
    min_sent_len : int = 10
    max_sent_len : int = 256
    data_workers : int = 8
    tokenizer_name_or_path : str = None
    # train
    lr: float = 1e-3
    sm_lr : float = 5e-5
    weight_decay : float = 1e-5
    early_stop : bool = None
    accum_grad : int = 1
    log_path : str = '/logs'
    root_dir : str = '/checkpoints'
    epoches : int = None
    log_steps : int = 500
    version : str = 'v0.1'
    share_emb_prj_weight : bool = True
    batch_size : int = 32
    # model
    model_name_or_path : str = None
    cache_dir : str = None
    # optimizer
    


def get_args(Args = TrainingArgs):
    """
    解析命令行参数，并返回一个所有命中的args。
    Args：
        Args: (optional) dataclass, 默认为TrainingArgs， 可以集成TrainingArgs加入个性化参数

    Return：
        train_args: datacalss

    """

    parser = argparse.ArgumentParser()

    for arg, content in Args.__dataclass_fields__.items():
        if content.type == bool:
            parser.add_argument('--'+arg, action='store_true')
            continue
        parser.add_argument('--'+arg, type=content.type, default=content.default)
    
    args = parser.parse_args()
    args=vars(args)
    train_args = Args(**args)

    return train_args