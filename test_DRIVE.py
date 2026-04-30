import argparse
import logging
import os
import random
import sys
import numpy as np
import glob
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# 导入所有数据集类
from datasets.dataset_DRIVE import Drive_dataset
from datasets.dataset_OCTA_SS import Octa_SS_dataset
from datasets.dataset_OCTA_3M import OCTA_3M_dataset

from utils import test_single_volume
from networks.vit_seg_modeling0 import VisionTransformer as ViT_seg
from networks.vit_seg_modeling0 import CONFIGS as CONFIGS_ViT_seg
import segmentation_models_pytorch as smp

# ==============================================================================
# 功能函数：交互式选择数据集
# ==============================================================================
def select_dataset_interactively():
    datasets = ['DRIVE', 'OCTA_SS', 'OCTA_3M']
    print("\n" + "="*40)
    print("        医学图像分割数据集选择器")
    print("="*40)
    for i, d in enumerate(datasets):
        print(f" [{i}] : {d}")
    print("="*40)
    while True:
        try:
            choice = input(f"请选择数据集编号 (0-{len(datasets)-1}) [默认 0]: ").strip()
            if choice == "": return datasets[0]
            idx = int(choice)
            if 0 <= idx < len(datasets): return datasets[idx]
            else: print("❌ 无效编号！")
        except ValueError: print("❌ 请输入数字！")

# ==============================================================================
# 功能函数：交互式选择模型
# ==============================================================================
def select_model_interactively(dataset_name):
    models = [
        'transunet', 'unet', 'att_unet', 'resnet', 'linknet', 'fpn', 'pspnet',
        'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32'
    ]
    print("\n" + "="*40)
    print(f"    {dataset_name} 数据集模型【测试】选择器")
    print("="*40)
    for i, m in enumerate(models):
        print(f" [{i}] : {m}")
    print("="*40)
    while True:
        try:
            choice = input(f"请输入要测试的模型编号 (0-{len(models)-1}) [默认 0]: ").strip()
            if choice == "": return models[0]
            idx = int(choice)
            if 0 <= idx < len(models): return models[idx]
            else: print("❌ 无效编号！")
        except ValueError: print("❌ 请输入数字！")

# ---------------------------------------------------------
# 参数配置
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='DRIVE', help='experiment_name')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--img_size', type=int, default=512, help='input patch size')
parser.add_argument('--is_savenii', action="store_true", default=True)
parser.add_argument('--n_skip', type=int, default=3)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
parser.add_argument('--vit_patches_size', type=int, default=16)

args = parser.parse_args()

def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        case_name = sampled_batch['case_name'][0]

        if len(image.shape) == 3:
            image = image.unsqueeze(1)
        if len(label.shape) == 3:
            label = label.unsqueeze(1)
        
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        
        metric_list += np.array(metric_i)
    
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"

if __name__ == "__main__":
    # 1. 交互选择数据集和模型
    args.dataset = select_dataset_interactively()
    args.model_name = select_model_interactively(args.dataset)

    # 2. 自动修正具体 ViT 参数
    if args.model_name in ['ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32']:
        args.vit_name = args.model_name
        args.vit_patches_size = int(args.model_name.split('_')[-1])
    elif args.model_name == 'transunet':
        args.vit_name = 'R50-ViT-B_16'
        args.vit_patches_size = 16

    if not args.deterministic:
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        cudnn.benchmark, cudnn.deterministic = False, True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    INPUT_CHANNELS = 1

    # 数据集配置表 (在此添加新数据集路径)
    dataset_config = {
        'DRIVE': {
            'Dataset': Drive_dataset,
            'volume_path': '/work/ly_s/WORK/Segment/TransUNet-main/data/DRIVE',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'OCTA_SS': {
            'Dataset': Octa_SS_dataset,
            'volume_path': '/work/ly_s/WORK/Segment/TransUNet-main/data/OCTA-SS',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'OCTA_3M': {
            'Dataset': OCTA_3M_dataset,
            'volume_path': '/work/ly_s/WORK/Segment/TransUNet-main/data/OCTA_3M',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    
    config = dataset_config[args.dataset]
    args.Dataset = config['Dataset']
    args.volume_path = config['volume_path']
    args.num_classes = config['num_classes']
    args.z_spacing = config['z_spacing']

    # ---------------------------------------------------------
    # 【精准匹配权重路径】
    # ---------------------------------------------------------
    base_model_dir = "/work/ly_s/WORK/Segment/TransUNet-main/model"
    
    # 根据截图中的命名规则：TU_{数据集}_{尺寸}_{模型名}
    # 注意：如果 args.dataset 是 'OCTA_SS'，而文件夹名是 'OCTA-SS'，需要做个简单转换
    folder_dataset_name = args.dataset.replace('_', '-') if 'OCTA' in args.dataset else args.dataset
    
    # 构建目标文件夹名称
    # 示例结果: TU_DRIVE_512_att_unet 或 TU_OCTA-SS_512_transunet
    folder_name = f"TU_{folder_dataset_name}_{args.img_size}_{args.model_name}"
    target_folder = os.path.join(base_model_dir, folder_name, 'TU')

    print(f"\n📂 正在检索目录: {target_folder}")

    if not os.path.exists(target_folder):
        # 备选方案：尝试直接使用 args.dataset 不做横杠转换
        folder_name_alt = f"TU_{args.dataset}_{args.img_size}_{args.model_name}"
        target_folder_alt = os.path.join(base_model_dir, folder_name_alt, 'TU')
        if os.path.exists(target_folder_alt):
            target_folder = target_folder_alt
        else:
            print(f"❌ 未找到对应的权重目录！")
            print(f"   尝试过: {folder_name}")
            print(f"   也尝试过: {folder_name_alt}")
            sys.exit(1)

    # 获取文件夹下所有的 .pth 权重文件
    pth_files = glob.glob(os.path.join(target_folder, '*.pth'))
    if not pth_files:
        print(f"❌ 目录存在但其中没有 .pth 文件: {target_folder}")
        sys.exit(1)

    # 自动选择权重：优先 best_model.pth，其次 latest_model.pth，最后选最新的 epoch
    snapshot_path = None
    for priority in ['best_model.pth', 'latest_model.pth']:
        p = os.path.join(target_folder, priority)
        if os.path.exists(p):
            snapshot_path = p
            break
    
    if snapshot_path is None:
        pth_files.sort(key=os.path.getmtime, reverse=True)
        snapshot_path = pth_files[0]

    print(f"🎯 成功匹配权重: {os.path.basename(snapshot_path)}")

    # 动态测试结果与日志保存路径
    args.test_save_dir = os.path.join(args.volume_path, 'test_results', args.model_name)
    log_folder = f'./test_log/test_log_{args.dataset}_{args.model_name}'
    
    os.makedirs(args.test_save_dir, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    logging.basicConfig(filename=log_folder + '/test_results.txt', level=logging.INFO, 
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # ---------------------------------------------------------
    # 配置与实例化模型 (保持不变)
    # ---------------------------------------------------------
    net = None
    is_vit_family = args.model_name in ['transunet', 'kan_transunet', 'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32']

    if is_vit_family:
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if not hasattr(config_vit, 'patches'):
            from ml_collections import ConfigDict
            config_vit.patches = ConfigDict({'size': (args.vit_patches_size, args.vit_patches_size)})
        else:
            config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)

        if 'R50' in args.vit_name:
            config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
            config_vit.skip_channels = [512, 256, 64, 16]
        else:
            config_vit.skip_channels = [0, 0, 0, 0]
        
        img_size_tuple = (args.img_size, args.img_size)
        net = ViT_seg(config_vit, img_size=img_size_tuple, num_classes=config_vit.n_classes).cuda()

    elif args.model_name == 'resnet':
        net = smp.FPN(encoder_name="resnet34", encoder_weights="imagenet", in_channels=INPUT_CHANNELS, classes=args.num_classes).cuda()
    elif args.model_name == 'att_unet':
        net = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=INPUT_CHANNELS, classes=args.num_classes, decoder_attention_type="scse").cuda()
    elif args.model_name == 'linknet':
        net = smp.Linknet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=INPUT_CHANNELS, classes=args.num_classes).cuda()
    elif args.model_name == 'fpn':
        net = smp.FPN(encoder_name="resnet34", encoder_weights="imagenet", in_channels=INPUT_CHANNELS, classes=args.num_classes).cuda()
    elif args.model_name == 'unet':
        net = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=INPUT_CHANNELS, classes=args.num_classes).cuda()
    elif args.model_name == 'pspnet':
        net = smp.PSPNet(encoder_name="resnet34", encoder_weights=None, in_channels=INPUT_CHANNELS, classes=args.num_classes).cuda()

    if net is None:
        print(f"\n❌ 模型 {args.model_name} 的测试实例化尚未支持！")
        sys.exit(1)

    net.load_state_dict(torch.load(snapshot_path))
    print(f"✅ 成功加载 {args.model_name} 权重！准备开始测试...")

    class ResizeWrapper(nn.Module):
        def __init__(self, model): super().__init__(); self.model = model
        def forward(self, x): 
            return F.interpolate(self.model(x), size=x.shape[-2:], mode='bilinear', align_corners=False)

    net = ResizeWrapper(net) 
    inference(args, net, args.test_save_dir)