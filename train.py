import random
import os
import shutil
import json
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader

from toolbox.datasets import get_dataset
from toolbox.log import get_logger
from toolbox.models import get_model
from toolbox.loss import get_loss
from toolbox.metrics import runningScore


def run(cfg, logger):
    # 所用数据集名称
    logger.info(f'Conf | use dataset {cfg["dataset"]}')
    # 所用数据增强方法
    logger.info(f'Conf | use augmentation {cfg["augmentation"]}')
    # 图片输入尺寸
    cfg['image_size'] = (cfg['image_h'], cfg['image_w'])
    logger.info(f'Conf | use image size {cfg["image_size"]}')

    # 获取训练集和验证集
    trainset, valset, testset = get_dataset(cfg)
    # batch size大小
    logger.info(f'Conf | use batch_size {cfg["batch_size"]}')

    train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    # 所用模型
    logger.info(f'Conf | use model {cfg["model_name"]}')
    model = get_model(cfg)

    # 是否多gpu训练
    gpu_ids = [int(i) for i in list(cfg['gpu_ids'])]
    logger.info(f'Conf | use GPU {gpu_ids}')
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(cfg['device'])

    # 优化器 & 学习率衰减
    logger.info(f'Conf | use optimizer Adam, lr={cfg["lr"]}, weight_decay={cfg["weight_decay"]}')
    logger.info(f'Conf | use step_lr_scheduler every {cfg["lr_decay_steps"]} steps decay {cfg["lr_decay_gamma"]}')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    # 损失函数 & 类别权重平衡
    logger.info(f'Conf | use loss function {cfg["loss"]}')
    criterion = get_loss(cfg).to(cfg['device'])

    # 训练 & 验证
    logger.info(f'Conf | use epoch {cfg["epoch"]}')

    running_metrics_val = runningScore(cfg['n_classes'])

    for ep in range(cfg['epoch']):

        # training
        model.train()
        best = [0]
        train_loss = 0

        for i, sample in enumerate(train_loader):
            # 载入数据
            img_data = sample['image'].to(cfg['device'])
            img_label = sample['label'].to(cfg['device'])
            # 训练
            out = model(img_data)
            loss = criterion(out, img_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            true_label = img_label.data.cpu().numpy()

            running_metrics_val.update(true_label, pre_label)

        metrics = running_metrics_val.get_scores()
        for k, v in metrics[0].items():
            print(k, v)
        train_miou = metrics[0]['MIoU: ']
        train_acc = metrics[0]['PA: ']
        train_class_acc = metrics[0]['MPA: ']

        if max(best) <= train_miou:
            best.append(train_miou)
            torch.save(model.state_dict(), os.path.join(cfg['logdir'], 'best_train_miou.pth'))

        logger.info(f'Iter | [{ep + 1:3d}/{cfg["epoch"]}] Train loss={train_loss / len(train_loader):.5f}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Train acc={train_acc :.5f}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Train miou={train_miou :.5f}')

        # net = model.eval()
        # eval_loss = 0

        # for j, sample in enumerate(val_loader):
        #     valImg = sample['image'].to(cfg['device'])
        #     valLabel = sample['label'].to(cfg['device'])

        #     out = net(valImg)
        #     loss = criterion(out, valLabel)
        #     eval_loss = loss.item() + eval_loss

        #     pre_label = out.max(dim=1)[1].data.cpu().numpy()
        #     true_label = valLabel.data.cpu().numpy()

        #     running_metrics_val.update(true_label, pre_label)
        # metrics = running_metrics_val.get_scores()
        # print('------------------eval------------------')
        # for k, v in metrics[0].items():
        #     print(k, v)
        # eval_miou = metrics[0]['MIoU: ']
        # eval_acc = metrics[0]['PA: ']

        # logger.info(f'Iter | [{ep + 1:3d}/{cfg["epoch"]}] Valid loss={eval_loss :.5f}')
        # logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Valid acc={eval_acc :.5f}')
        # logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Valid miou={eval_miou :.5f}')


if __name__ == '__main__':

    # 固定的代码
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/cityscapes_bisenet.json",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    # 用args调用config
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    # 训练的各种记录的保存目录
    logdir = f'run/{cfg["dataset"]}/{time.strftime("%Y-%m-%d-%H-%M")}-{random.randint(1000, 10000)}'
    os.makedirs(logdir)
    shutil.copy(args.config, logdir)
    # 初始化日志
    logger = get_logger(logdir)

    logger.info(f'Conf | use logdir {logdir}')

    cfg['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg['logdir'] = logdir

    run(cfg, logger)
