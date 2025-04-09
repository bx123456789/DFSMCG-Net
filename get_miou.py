import os
import argparse
from PIL import Image
from tqdm import tqdm
from cd_mode import CD_Model
from utils.utils_metrics import compute_mIoU, show_results




# 主要逻辑
def evaluate_miou(args):
    # 设置路径和参数
    VOCdevkit_path = args.VOCdevkit_path
    miou_out_path = args.miou_out_path
    pred_dir = os.path.join(miou_out_path, args.pred_dir_name)
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")

    # 创建输出目录
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    # 加载模型
    print("Load model.")
    model = CD_Model(
        model_path=args.model_path,
        num_classes=args.num_classes,
        input_shape=args.input_shape,
        cuda=args.cuda
    )
    print("Load model done.")

    # 预测并保存结果
    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path_A = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/A/" + image_id + ".png")
        image_path_B = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/B/" + image_id + ".png")

        image_A = Image.open(image_path_A)
        image_B = Image.open(image_path_B)

        # 对图像进行预测
        predicted_image = model.get_miou_png(image_A, image_B)
        predicted_image.save(os.path.join(pred_dir, image_id + ".png"))
    print("Get predict result done.")

    # 计算 mIoU
    print("Get miou.")
    hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, args.num_classes, args.name_classes)
    print("Get miou done.")
    show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, args.name_classes)

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate mIoU for segmentation results")
    # cuda设置
    parser.add_argument("--cuda", type=bool, default=True, help="是否使用GPU加速")
    # 模型权重文件
    parser.add_argument("--model_path", type=str, default="logs/best_epoch_weights.pth", help="模型权重文件路径")
    # 分类个数
    parser.add_argument("--num_classes", type=int, default=2, help="分类个数（包括背景类）")
    # 类别名称
    parser.add_argument("--name_classes", type=str, nargs='+', default=["0", "1"],
                        help="类别名称列表，使用空格分隔。示例：--name_classes 0 1")
    # 图片大小
    parser.add_argument("--input_shape", type=int, nargs=2, default=[512, 512], help="输入图片尺寸")
    # 数据集路径
    parser.add_argument("--VOCdevkit_path", type=str, default="VOCdevkit1/VOCdevkit1",
                        help="VOC 数据集的根目录路径")
    # 计算结果输出地址
    parser.add_argument("--miou_out_path", type=str, default="miou_out",
                        help="用于存储mIoU计算结果的输出目录")
    # 预测结果存储地址
    parser.add_argument("--pred_dir_name", type=str, default="miou_out_images",
                        help="存储预测结果的文件夹名称")

    return parser.parse_args()

# 主入口
if __name__ == "__main__":
    # 获取命令行参数
    args = parse_args()
    # 调用评估函数
    evaluate_miou(args)
