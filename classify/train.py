# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
在分类数据集上训练 YOLOv5 分类器模型

用法 - 单GPU训练:
    $ python classify/train.py --model yolov5s-cls.pt --data imagenette160 --epochs 5 --img 224

用法 - 多GPU DDP训练:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3

数据集:            --data mnist, fashion-mnist, cifar10, cifar100, imagenette, imagewoof, imagenet, 或 'path/to/data'
YOLOv5-cls 模型:  --model yolov5n-cls.pt, yolov5s-cls.pt, yolov5m-cls.pt, yolov5l-cls.pt, yolov5x-cls.pt
Torchvision 模型: --model resnet50, efficientnet_b0, 等等。请参阅 https://pytorch.org/vision/stable/models.html
"""

import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.hub as hub
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将 ROOT 添加到 PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径

from classify import val as validate
from models.experimental import attempt_load
from models.yolo import ClassificationModel, DetectionModel
from utils.dataloaders import create_classification_dataloader
from utils.general import (DATASETS_DIR, LOGGER, TQDM_BAR_FORMAT, WorkingDirectory, check_git_info, check_git_status,
                           check_requirements, colorstr, download, increment_path, init_seeds, print_args, yaml_save)
from utils.loggers import GenericLogger
from utils.plots import imshow_cls
from utils.torch_utils import (ModelEMA, model_info, reshape_classifier_output, select_device, smart_DDP,
                               smart_optimizer, smartCrossEntropyLoss, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()

def train(opt, device):
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, data, bs, epochs, nw, imgsz, pretrained = \
        opt.save_dir, Path(opt.data), opt.batch_size, opt.epochs, min(os.cpu_count() - 1, opt.workers), \
        opt.imgsz, str(opt.pretrained).lower() == 'true'
    cuda = device.type != 'cpu'

    # 目录
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # 创建目录
    last, best = wdir / 'last.pt', wdir / 'best.pt'

    # 保存运行设置
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    # 日志记录器
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    # 下载数据集
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        data_dir = data if data.is_dir() else (DATASETS_DIR / data)
        if not data_dir.is_dir():
            LOGGER.info(f'\n未找到数据集 ⚠️, 缺少路径 {data_dir}, 正在尝试下载...')
            t = time.time()
            if str(data) == 'imagenet':
                subprocess.run(f"bash {ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
            else:
                url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{data}.zip'
                download(url, dir=data_dir.parent)
            s = f"数据集下载成功 ✅ ({time.time() - t:.1f}s), 保存到 {colorstr('bold', data_dir)}\n"
            LOGGER.info(s)

    # 数据加载器
    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])  # 类别数量
    trainloader = create_classification_dataloader(path=data_dir / 'train',
                                                   imgsz=imgsz,
                                                   batch_size=bs // WORLD_SIZE,
                                                   augment=True,
                                                   cache=opt.cache,
                                                   rank=LOCAL_RANK,
                                                   workers=nw)

    test_dir = data_dir / 'test' if (data_dir / 'test').exists() else data_dir / 'val'  # data/test 或 data/val
    if RANK in {-1, 0}:
        testloader = create_classification_dataloader(path=test_dir,
                                                      imgsz=imgsz,
                                                      batch_size=bs // WORLD_SIZE * 2,
                                                      augment=False,
                                                      cache=opt.cache,
                                                      rank=-1,
                                                      workers=nw)

    # 模型
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        if Path(opt.model).is_file() or opt.model.endswith('.pt'):
            model = attempt_load(opt.model, device='cpu', fuse=False)
        elif opt.model in torchvision.models.__dict__:  # TorchVision 模型，如 resnet50, efficientnet_b0
            model = torchvision.models.__dict__[opt.model](weights='IMAGENET1K_V1' if pretrained else None)
        else:
            m = hub.list('ultralytics/yolov5')  # + hub.list('pytorch/vision')  # 模型
            raise ModuleNotFoundError(f'--model {opt.model} 未找到。可用模型为: \n' + '\n'.join(m))
        if isinstance(model, DetectionModel):
            LOGGER.warning("警告 ⚠️ 请传递带有 '-cls' 后缀的 YOLOv5 分类模型，如 '--model yolov5s-cls.pt'")
            model = ClassificationModel(model=model, nc=nc, cutoff=opt.cutoff or 10)  # 转换为分类模型
        reshape_classifier_output(model, nc)  # 更新类别数
    for m in model.modules():
        if not pretrained and hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        if isinstance(m, torch.nn.Dropout) and opt.dropout is not None:
            m.p = opt.dropout  # 设置 dropout
    for p in model.parameters():
        p.requires_grad = True  # 用于训练
    model = model.to(device)

    # 信息
    if RANK in {-1, 0}:
        model.names = trainloader.dataset.classes  # 附加类别名称
        model.transforms = testloader.dataset.torch_transforms  # 附加推理变换
        model_info(model)
        if opt.verbose:
            LOGGER.info(model)
        images, labels = next(iter(trainloader))
        file = imshow_cls(images[:25], labels[:25], names=model.names, f=save_dir / 'train_images.jpg')
        logger.log_images(file, name='Train Examples')
        logger.log_graph(model, imgsz)  # 记录模型

    # 优化器
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=opt.decay)

    # 调度器
    lrf = 0.01  # 最终学习率（lr0 的比例）
    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # 余弦
    lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf  # 线性
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr0, total_steps=epochs, pct_start=0.1,
    #                                    final_div_factor=1 / 25 / lrf)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # DDP 模式
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # 训练
    t0 = time.time()
    criterion = smartCrossEntropyLoss(label_smoothing=opt.label_smoothing)  # 损失函数
    best_fitness = 0.0
    scaler = amp.GradScaler(enabled=cuda)
    val = test_dir.stem  # 'val' 或 'test'
    LOGGER.info(f'图像尺寸 {imgsz} 训练, {imgsz} 测试\n'
                f'使用 {nw * WORLD_SIZE} 个数据加载器工作线程\n'
                f"结果记录到 {colorstr('bold', save_dir)}\n"
                f'开始训练超参数: {colorstr("yaml", opt)}\n')

    for epoch in range(epochs):  # 逐周期
        model.train()

        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=len(trainloader), bar_format=TQDM_BAR_FORMAT)  # 进度条

        optimizer.zero_grad()
        for i, (im, labels) in pbar:
            im, labels = im.to(device, non_blocking=True), labels.to(device)

            # 前向传播
            with amp.autocast(enabled=cuda):
                loss = criterion(model(im), labels)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

            # 记录
            if RANK in {-1, 0}:
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # GPU内存
                pbar.set_description(f"{epoch + 1}/{epochs} "
                                     f"mem: {mem} "
                                     f"loss: {loss.item():.4g} "
                                     f"lr: {optimizer.param_groups[0]['lr']:.3g}")

        # 学习率调度
        scheduler.step()

        # 验证
        if RANK in {-1, 0}:
            final_epoch = (epoch + 1 == epochs)
            results, _, val_file = validate.run(
                data=opt.data,
                batch_size=bs // WORLD_SIZE * 2,
                imgsz=imgsz,
                model=ema.ema,
                dataloader=testloader,
                save_dir=save_dir,
                verbose=opt.verbose,
                plots=final_epoch,
                log_preds=final_epoch,
                compute_loss=None)

            # 保存模型
            fi = results[0]  # fitness
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(results) + [scheduler.optimizer.param_groups[0]['lr']]  # 日志记录值
            callbacks = logger.callback if logger else None
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, final_epoch, opt.save_dir, last, best)

            # 保存最后和最优模型
            if final_epoch or (epoch + 1) % opt.save_period == 0:
                torch.save(model.state_dict(), last)
                if best_fitness == fi:
                    torch.save(model.state_dict(), best)

    # 结束
    if RANK in {-1, 0}:
        n = opt.name or f'{opt.data}-{opt.model.split(".")[0]}'
        LOGGER.info(f"\n训练完成 ({time.time() - t0:.3f}s) 最优结果保存到 {colorstr('bold', best)}, "
                    f"末次结果保存到 {last}\n")
        if opt.verbose and not opt.evolve:
            subprocess.run(f'zip -r {save_dir}.zip {save_dir} -q', shell=True, check=True)  # 保存
            LOGGER.info(f'{colorstr("bold", "结果已保存到")} {save_dir}.zip\n')

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s-cls.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default='imagenette160', help='dataset name')
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='maximum dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train-cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--pretrained', nargs='?', const=True, default=True, help='start from pretrained model')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--decay', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--cutoff', type=int, default=None, help='layer cutoff index (optional)')
    parser.add_argument('--dropout', type=float, help='use dropout regularization (optional)')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
    if RANK in {-1, 0}:
        print_args(vars(opt))
        if not opt.evolve:
            check_git_status()
            check_requirements()

    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')

    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)
    if not opt.evolve:
        (opt.save_dir / 'weights').mkdir(parents=True, exist_ok=True)
    train(opt, device)

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
