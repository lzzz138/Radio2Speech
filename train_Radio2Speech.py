import logging              # 日志输出
import time                 # 计时
import os                   # 路径与目录
import numpy as np          # 读取预训练 numpy 权重

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm       # 进度条
import torch.utils.data.distributed
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 写入

from mel_Arguments import ArgParser                 # 参数解析器
from RadioAudio_meldataset import Meldataset        # 数据集类，返回 (audio_amp, radio_amp)
from cnn_transformer.utils import AdamW, get_linear_schedule_with_warmup, LogSTFTMagnitudeLoss, checkpoint
from utils.mel_utils import AverageMeter, dist_average  # 计量与分布式平均

# transformer + unet
from cnn_transformer.transunet import TransUnet as TransUnet

from evaluation import evaluate, evaluate_visual     # 评估函数

def train_one_epoch(net, train_loader, optimizer, scheduler, criterion, epoch, accumulated_iter, tb_log, args):
    batch_time = AverageMeter()      # 记录每 batch 的耗时
    dataload_time = AverageMeter()   # 记录数据加载耗时
    total_loss = 0                   # 累积损失用于进度条显示

    # tqdm 进度条，长度 = len(train_loader)（批次数）
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.epochs}', unit='batch') as pbar:
        torch.cuda.synchronize()
        tic = time.perf_counter()

        for i, batch_data in enumerate(train_loader):
            audio_amp, radio_amp = batch_data
            # 原始维度 (DataLoader 输出):
            # audio_amp: [B, 1, T, F]
            # radio_amp: [B, 1, T, F]
            audio_amp = audio_amp.cuda()
            radio_amp = radio_amp.cuda()

            # 数据加载时间
            torch.cuda.synchronize()
            dataload_time.update(time.perf_counter() - tic)

            # 前向：模型输入 radio_amp => 输出预测 Mel
            # net(radio_amp) 期望输入 [B, 1, T, F]
            # 内部流程：
            # 1. 编码器 in_conv -> [B, 128, T, F]
            # 2. TSB / down_conv 三次下采样 -> [B, 1024, T/8, F/8]
            # 3. patch embedding -> flatten -> [B, N_patch, hidden_size]
            # 4. Transformer 编码 -> 同形状 [B, N_patch, hidden_size]
            # 5. 解码重建 -> [B, 1, T, F]
            audio_pred = net(radio_amp)

            # 损失：预测与目标 Mel L1（LogSTFTMagnitudeLoss 内部当前实现为 F.l1_loss）
            loss = criterion(audio_pred, audio_amp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # 每步调整学习率（线性 warmup 或 StepLR）

            # 计算 batch 时间
            torch.cuda.synchronize()
            batch_time.update(time.perf_counter() - tic)
            tic = time.perf_counter()

            accumulated_iter += 1      # 全局迭代计数（跨 epoch）
            total_loss += loss.item()

            # 间隔打印
            if i % args.disp_iter == 0:
                print('Epoch: [{}][{}/{}], batch time: {:.3f}, Data time: {:.3f}, loss: {:.4f}'
                  .format(epoch, i, len(train_loader),
                          batch_time.average(), dataload_time.average(), loss.item()))

            # 写入 TensorBoard 标量
            tb_log.add_scalar('train/loss', loss.item(), accumulated_iter)
            tb_log.add_scalar('train/epoch', epoch, accumulated_iter)
            tb_log.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], accumulated_iter)

            # 更新进度条（手动）
            pbar.update(1)
            # 分布式显示损失（使用 dist_average 聚合各 GPU）
            if args.distributed:
                pbar.set_postfix(**{'loss (batch)': dist_average([total_loss / (i + 1)], i + 1)[0]})
            else:
                pbar.set_postfix(**{'loss (batch)': total_loss / (i+1)})

    return accumulated_iter   # 返回累计迭代数供下个 epoch 继续记录

def train_net(args):
    # 0. 分布式与 TensorBoard 初始化
    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)
    tb_log = SummaryWriter(log_dir=args.tensorboard_dir)

    # 1. 创建数据集
    train_sampler = None
    val_sampler = None
    train_set = Meldataset(args.list_train)   # __getitem__ -> [1, T, F]
    val_set = Meldataset(args.list_val)

    # 分布式采样器（确保不同进程不重复）
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

    # 2. DataLoader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                              shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, num_workers=4,
                            pin_memory=True, shuffle=False, sampler=val_sampler)
    # 输出 batch 形状：
    # audio_amp / radio_amp: [B, 1, T, F]

    # 3. 构建模型
    net = TransUnet(args.hidden_size, 
                    args.transformer_num_layers, 
                    args.mlp_dim, 
                    args.num_heads, 
                    args.transformer_dropout_rate, 
                    args.transformer_attention_dropout_rate
                    )
    # 加载预训练 Transformer (numpy 权重)，影响内部 Attention / MLP 层初始化
    net.load_from(weights=np.load(args.transformer_pretrained_path))

    # 同步 BN（多 GPU）
    if args.distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net,
                            device_ids=[args.local_rank], output_device=args.local_rank)

    # 4. 优化器与调度器
    if args.opt == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.opt == 'adamw':
        optimizer = AdamW(net.parameters(), args.learning_rate)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)

    if args.lr_scheduler == 'warmup':
        num_update_steps_per_epoch = len(train_loader)
        max_train_steps = args.epochs * num_update_steps_per_epoch
        warmup_steps = int(args.warmup_ratio * max_train_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_train_steps)  # 前 warmup 上升，后线性下降
    else:
        step_size = args.step_size * len(train_loader)  # 把“按 epoch”转换为“按迭代步”
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.4)

    criterion = LogSTFTMagnitudeLoss()  # L1 Mel 频谱损失

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.learning_rate}
    ''')

    # 5. 可选继续训练的 checkpoint
    start_epoch = count_iter = 0
    if args.load_checkpoint is not None:
        dist.barrier()
        logging.info(f'Loading checkpoint net from: {args.load_checkpoint}')
        package = torch.load(args.load_checkpoint, map_location='cpu')
        net_dict = net.state_dict()
        state_dict = {k: v for k, v in package.items() if k in net_dict.keys()}
        net_dict.update(state_dict)
        net.load_state_dict(net_dict)
    history = {'eval_loss': []}  # 记录验证损失序列

    # 6. 训练循环
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)  # 保证每 epoch shuffle 同步
        net.train()
        start = time.time()

        logging.info('-' * 70)
        logging.info('Training...')
        count_iter = train_one_epoch(net, train_loader, optimizer, scheduler, criterion, epoch, count_iter, tb_log, args)
        logging.info(f'Train Summary | End of Epoch {epoch} | Time {time.time() - start:.2f}s')

        logging.info('-' * 70)
        logging.info('Evaluating...')
        # 验证：输入形状与训练相同 [B, 1, T, F]，输出 [B, 1, T, F]
        evaluate(net, val_loader, criterion, epoch, history, tb_log, count_iter)

        # 可视化（加图像到 TensorBoard）
        if (epoch + 1) % args.metrics_every == 0 or epoch == 0 or epoch == args.epochs - 1 :
            logging.info('-' * 70)
            logging.info('Calculating metrics...')
            evaluate_visual(net, val_loader, epoch, tb_log, count_iter, args)

        if args.local_rank == 0:
            checkpoint(net, history, epoch, optimizer, count_iter, args)  # 保存最优/定期模型

def main():
    parser = ArgParser()                       # 解析 YAML + 命令行参数
    args = parser.parse_train_arguments()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    train_net(args)

if __name__ == '__main__':
    main()




