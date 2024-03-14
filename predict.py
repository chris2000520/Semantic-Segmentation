import os
import json
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from toolbox.datasets import get_dataset
from toolbox.models import get_model
from toolbox.log import get_logger
from toolbox.metrics import runningScore


def predict(cfg, runid, use_pth='best_train_miou.pth'):

    dataset = cfg['dataset']
    train_logdir = f'run/{dataset}/{runid}'

    test_logdir = os.path.join('./results', dataset, runid)
    logger = get_logger(test_logdir)

    logger.info(f'Conf | use logdir {train_logdir}')
    logger.info(f'Conf | use dataset {cfg["dataset"]}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 测试集
    trainset, valset, testset = get_dataset(cfg)

    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])

    # model
    model = get_model(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(train_logdir, use_pth)))

    pd_label_color = pd.read_csv(trainset.file_path[2], sep=',')
    name_value = pd_label_color['name'].values
    num_class = len(name_value)
    colormap = []
    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        colormap.append(color)

    cm = np.array(colormap).astype('uint8')

    running_metrics_val = runningScore(12)

    for i, sample in enumerate(test_loader):
        valImg = sample['image'].to(device)
        valLabel = sample['label'].long().to(device)
        out = model(valImg)
        out = F.log_softmax(out, dim=1)

        pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
        pre = cm[pre_label]
        pre1 = Image.fromarray(pre)
        pre1.save(test_logdir + '/' + str(i) + '.png')

        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        true_label = valLabel.data.cpu().numpy()

        running_metrics_val.update(true_label, pre_label)

    metrics = running_metrics_val.get_scores()
    for k, v in metrics[0].items():
        print(k, v)
    test_miou = metrics[0]['mIou: ']
    test_acc = metrics[0]['pixel_acc: ']
    test_class_acc = metrics[0]['class_acc: ']

    logger.info(f'Test | Test Acc={test_acc / (len(test_loader)):.5f}')
    logger.info(f'Test | Test Mean IU={test_miou / (len(test_loader)):.5f}')
    # logger.info(f'Test | Test_class_acc={list(test_class_acc / (len(test_loader)))}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument("-id", type=str, help="predict id")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/camvid_linknet.json",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    args.id = '2024-03-12-14-22-7800'

    predict(cfg, args.id)
