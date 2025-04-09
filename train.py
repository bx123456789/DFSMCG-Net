import argparse
import datetime
import os
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from nets.loss import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import  deeplab_dataset_collate,CD_Dataset
from utils.utils import download_weights, seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch
from nets.BIT import BASE_Transformer
import numpy as np




def train_model(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # 初始化模型
    model = BASE_Transformer(input_nc=3, output_nc=args.num_classes, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)
    cls_weights = np.array(args.cls_weights, dtype=np.float32)
   

    if not args.pretrained:
        weights_init(model)
    if args.model_path != '':
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location=device)
        temp_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    model_train = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model_train = model_train.cuda()
    # 记录Loss
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(args.save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=args.input_shape)


    # 数据读取
    with open(os.path.join(args.vocdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(args.vocdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # 显示训练配置
    show_config(
        num_classes=args.num_classes, backbone=args.backbone, model_path=args.model_path, input_shape=args.input_shape,
        Init_Epoch=args.init_epoch, Freeze_Epoch=args.freeze_epoch, UnFreeze_Epoch=args.unfreeze_epoch,
        Freeze_batch_size=args.freeze_batch_size, Unfreeze_batch_size=args.unfreeze_batch_size,
        Freeze_Train=args.freeze_train,
        Init_lr=args.init_lr, Min_lr=args.min_lr, optimizer_type=args.optimizer_type, momentum=args.momentum,
        lr_decay_type=args.lr_decay_type,
        save_period=args.save_period, save_dir=args.save_dir, num_workers=args.num_workers, num_train=num_train,
        num_val=num_val
    )

    UnFreeze_flag = False
    if args.freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    batch_size = args.freeze_batch_size if args.freeze_train else args.unfreeze_batch_size
    Init_lr_fit = min(max(batch_size / 16 * args.init_lr, 3e-4), 5e-4)
    Min_lr_fit = min(max(batch_size / 16 * args.min_lr, 3e-6), 5e-5)

    # 优化器设置
    optimizer = {
        'adam': optim.Adam(model.parameters(), lr=Init_lr_fit, betas=(args.momentum, 0.999),
                           weight_decay=args.weight_decay),
        'sgd': optim.SGD(model.parameters(), lr=Init_lr_fit, momentum=args.momentum, nesterov=True,
                         weight_decay=args.weight_decay),
        'adamw': optim.AdamW(model.parameters(), lr=Init_lr_fit, betas=(args.momentum, 0.999),
                             weight_decay=args.weight_decay),
        'rmsprop': optim.RMSprop(model.parameters(), lr=Init_lr_fit, momentum=args.momentum, alpha=0.99,
                                 weight_decay=args.weight_decay),
        'adagrad': optim.Adagrad(model.parameters(), lr=Init_lr_fit, lr_decay=0, weight_decay=args.weight_decay,
                                 initial_accumulator_value=0.1),
        'adamax': optim.Adamax(model.parameters(), lr=Init_lr_fit, betas=(args.momentum, 0.999),
                               weight_decay=args.weight_decay)
    }.get(args.optimizer_type)

    lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.unfreeze_epoch)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    # 训练数据加载器
    train_dataset = CD_Dataset(train_lines, args.input_shape, args.num_classes, True,
                               args.vocdevkit_path, args.image_format, args.label_format)
    # 验证数据加载器
    # 创建数据集
    val_dataset = CD_Dataset(val_lines, args.input_shape, args.num_classes, False,
                             args.vocdevkit_path, args.image_format, args.label_format)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True,
                     drop_last=True, collate_fn=deeplab_dataset_collate,
                     worker_init_fn=partial(worker_init_fn, rank=0, seed=args.seed))
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate,
                         worker_init_fn=partial(worker_init_fn, rank=0, seed=args.seed))
    eval_callback = EvalCallback(model, args.input_shape, args.num_classes, val_lines, args.vocdevkit_path, log_dir,
                                 args.cuda,
                                 eval_flag=args.eval_flag, period=args.eval_period)

    for epoch in range(args.init_epoch, args.unfreeze_epoch):
        if epoch >= args.freeze_epoch and not UnFreeze_flag and args.freeze_train:
            for param in model.backbone.parameters():
                param.requires_grad = True
            batch_size = args.unfreeze_batch_size
            UnFreeze_flag = True
            gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers,
                             pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                             worker_init_fn=partial(worker_init_fn, rank=0, seed=args.seed))
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                                 worker_init_fn=partial(worker_init_fn, rank=0, seed=args.seed))

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(
            model_train,model , loss_history, eval_callback, optimizer, epoch,
            epoch_step, epoch_step_val, gen, gen_val, args.unfreeze_epoch, args.cuda,
            cls_weights, args.num_classes, args.save_period,
            args.save_dir,
        )

    loss_history.writer.close()
def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Model Training")

    # 基本设置
    parser.add_argument("--cuda", type=bool, default=True, help="是否使用GPU加速")
    parser.add_argument("--seed", type=int, default=11, help="随机种子，用于保证实验的可复现性")
    # 数据集相关参数
    parser.add_argument("--image_format", type=str, choices=["tif", "jpg", "png"], default="png",
                        help="训练图像的文件格式，支持 tif, jpg, png")
    parser.add_argument("--label_format", type=str, choices=["tif", "jpg", "png"], default="png",
                        help="标签图像的文件格式，支持 tif, jpg, png")
    # 模型相关参数
    parser.add_argument("--num_classes", type=int, default=2, help="分类总数+1（包括背景）")
    parser.add_argument("--backbone", type=str, default="mobilenet", help="主干网络类型，影响特征提取部分")
    parser.add_argument("--pretrained", type=bool, default=False, help="是否加载主干网络的预训练权重")
    parser.add_argument("--model_path", type=str, default="", help="模型权重的路径，若为空则不加载任何预训练权重")
    parser.add_argument("--downsample_factor", type=int, default=16, help="下采样倍数，8或16")
    parser.add_argument("--input_shape", type=int, nargs=2, default=[512, 512], help="输入图片尺寸")

    # 训练过程设置
    parser.add_argument("--init_epoch", type=int, default=0, help="初始Epoch，断点续训可从特定Epoch开始")
    parser.add_argument("--freeze_epoch", type=int, default=50, help="冻结阶段训练的Epoch数")
    parser.add_argument("--freeze_batch_size", type=int, default=2, help="冻结阶段的batch size")
    parser.add_argument("--unfreeze_epoch", type=int, default=1000, help="总训练Epoch数")
    parser.add_argument("--unfreeze_batch_size", type=int, default=2, help="解冻阶段的batch size")
    parser.add_argument("--freeze_train", type=bool, default=False, help="是否进行冻结训练")

    # 优化器及学习率设置
    parser.add_argument("--init_lr", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--min_lr", type=float, default=5e-4 * 0.01, help="最小学习率")
    parser.add_argument("--optimizer_type", type=str, choices=["adam", "sgd", "adamw", "rmsprop", "adagrad", "adamax"],
                        default="adam", help="优化器类型，可选 'adam', 'sgd', 'adamw', 'rmsprop', 'adagrad', 'adamax'")
    parser.add_argument("--momentum", type=float, default=0.9, help="优化器的momentum参数")
    parser.add_argument("--weight_decay", type=float, default=0, help="权重衰减系数")

    # 学习率衰减设置
    parser.add_argument("--lr_decay_type", type=str, choices=["step", "cos"], default="step", help="学习率衰减类型")

    # 模型保存及日志设置
    parser.add_argument("--save_period", type=int, default=1, help="每隔多少个Epoch保存一次权重")
    parser.add_argument("--save_dir", type=str, default="logs", help="权重和日志文件保存路径")

    # 评估设置
    parser.add_argument("--eval_flag", type=bool, default=True, help="是否在训练时进行评估")
    parser.add_argument("--eval_period", type=int, default=1, help="每隔多少个Epoch评估一次")

    # 数据集路径
    parser.add_argument("--vocdevkit_path", type=str, default="VOCdevkit1/VOCdevkit1", help="数据集路径")

    # 损失函数设置默认使用CE损失函数,若需修改损失函数在utils.utils_fit.py文件中
    parser.add_argument("--cls_weights", type=float, nargs='+', default=[1.0, 1.0],
                        help="分类权重，平衡各类别损失权重")

    # 数据加载设置
    parser.add_argument("--num_workers", type=int, default=1, help="多线程加载数据，设置为1即关闭多线程")

    return parser.parse_args()

if __name__ == "__main__":
    # 获取命令行参数
    args = parse_args()
    # 调用训练函数
    train_model(args)
