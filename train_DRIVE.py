import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import subprocess
import sys
import ssl  # 处理下载权限
import urllib.request
import segmentation_models_pytorch as smp
from ml_collections import ConfigDict  # 用于修复配置对象

from networks.vit_seg_modeling0 import VisionTransformer as ViT_seg
from networks.vit_seg_modeling0 import CONFIGS as CONFIGS_ViT_seg

# 【新增】确保导入所有相关数据集的 trainer
from trainer import trainer_synapse, trainer_OCTA_SS, trainer_DRIVE, trainer_OCTA_3M

# ==============================================================================
# 功能函数：1. 自动下载预训练权重
# ==============================================================================
def download_pretrained_weights(save_path, url):
    ssl_context = ssl._create_unverified_context()
    
    if not os.path.exists(save_path) or os.path.getsize(save_path) < 1024:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"\n🌐 正在从远程下载预训练权重... \n下载链接: {url}\n保存位置: {os.path.abspath(save_path)}")
        try:
            def progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = block_num * block_size * 100 / total_size
                    sys.stdout.write(f"\r进度: {percent:5.1f}%")
                    sys.stdout.flush()
            
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, save_path, reporthook=progress)
            print("\n✅ 下载完成！")
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
    else:
        print(f"📦 检测到本地已存在预训练权重: {os.path.abspath(save_path)}")

# ==============================================================================
# 功能函数：2. 交互式选择数据集 (新增)
# ==============================================================================
def select_dataset_interactively():
    datasets = ['DRIVE', 'OCTA-SS', 'OCTA_3M', 'Synapse']
    print("\n" + "="*40)
    print("      🏥 医学图像分割 - 数据集选择器")
    print("="*40)
    for i, d in enumerate(datasets):
        print(f" [{i}] : {d}")
    print("="*40)
    
    while True:
        try:
            choice = input(f"请输入数据集对应的编号 (0-{len(datasets)-1}) [默认 0]: ").strip()
            if choice == "": return datasets[0]
            idx = int(choice)
            if 0 <= idx < len(datasets): return datasets[idx]
            else: print("❌ 无效编号！")
        except ValueError: print("❌ 请输入数字！")

# ==============================================================================
# 功能函数：3. 交互式选择模型 
# ==============================================================================
def select_model_interactively(dataset_name):
    models = [
        'transunet', 'unet', 'att_unet', 'resnet', 'linknet', 'fpn', 'pspnet',
        'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32'
    ]
    print("\n" + "="*40)
    print(f"      🤖 {dataset_name} - 模型训练选择器")
    print("="*40)
    for i, m in enumerate(models):
        print(f" [{i}] : {m}")
    print("="*40)
    
    while True:
        try:
            choice = input(f"请输入模型对应的编号 (0-{len(models)-1}) [默认 0]: ").strip()
            if choice == "": return models[0]
            idx = int(choice)
            if 0 <= idx < len(models): return models[idx]
            else: print("❌ 无效编号！")
        except ValueError: print("❌ 请输入数字！")

# ==============================================================================
# 自动选择显存最大的 GPU 功能函数
# ==============================================================================
def get_device_id_with_max_memory():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE, encoding='utf-8')
        free_memory = [int(x) for x in result.stdout.strip().split('\n')]
        return free_memory.index(max(free_memory))
    except: return 0

# 参数配置 (这里的默认值会被交互式选择覆盖或被 config 更新)
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='/work/ly_s/WORK/Segment/TransUNet-main/model')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse')
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--n_skip', type=int, default=3)
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16') 
parser.add_argument('--vit_patches_size', type=int, default=16)
args = parser.parse_args()


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 步骤 1 & 2：交互式启动配置
    # ------------------------------------------------------------------
    args.dataset = select_dataset_interactively()
    args.model_name = select_model_interactively(args.dataset)

    # ------------------------------------------------------------------
    # 步骤 3：模型参数修正
    # ------------------------------------------------------------------
    if args.model_name in ['ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32']:
        args.vit_name = args.model_name
        args.vit_patches_size = int(args.model_name.split('_')[-1])
        args.is_pretrain = True
    else:
        args.is_pretrain = True if args.model_name == 'transunet' else False

    # GPU 与 种子设定
    best_gpu_id = get_device_id_with_max_memory()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # 开启防碎片化
    
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    
    # ------------------------------------------------------------------
    # 步骤 4：数据集与路径配置引擎 (动态挂载参数)
    # ------------------------------------------------------------------
    dataset_config = {
        'Synapse': {'root_path': '../data/Synapse/train_npz', 'list_dir': './lists/lists_Synapse', 'num_classes': 9, 'in_channels': 1},
        'OCTA-SS': {'root_path': '/work/ly_s/WORK/Segment/TransUNet-main/data/OCTA-SS', 'list_dir': '', 'num_classes': 2, 'in_channels': 1},
        'OCTA_3M': {'root_path': '/work/ly_s/WORK/Segment/TransUNet-main/data/OCTA_3M', 'list_dir': '', 'num_classes': 2, 'in_channels': 1},
        'DRIVE': {'root_path': '/work/ly_s/WORK/Segment/TransUNet-main/data/DRIVE', 'list_dir': '', 'num_classes': 2, 'in_channels': 1},
    }
    
    # 将选择的配置注入全局 args
    args.num_classes = dataset_config[args.dataset]['num_classes']
    args.root_path = dataset_config[args.dataset]['root_path']
    INPUT_CHANNELS = dataset_config[args.dataset]['in_channels']
    
    # 动态保存路径：严格根据【数据集_尺寸_模型名】隔离实验记录
    snapshot_path = os.path.join(args.output_dir, f"TU_{args.dataset}_{args.img_size}_{args.model_name}", 'TU')
    if not os.path.exists(snapshot_path): 
        os.makedirs(snapshot_path)
    
    print(f"\n🚀 启动训练配置核对:")
    print(f"   ► 数据集: {args.dataset}")
    print(f"   ► 模型名: {args.model_name}")
    print(f"   ► 显卡 ID: {best_gpu_id}")
    print(f"   ► Patch Size: {args.vit_patches_size}")
    print(f"   ► 存放路径: {snapshot_path}\n")

    net = None
    is_vit_family = args.model_name in ['transunet', 'kan_transunet', 'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32']

    # ------------------------------------------------------------------
    # 步骤 5：模型构建
    # ------------------------------------------------------------------
    if is_vit_family:
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        
        # 强制初始化 patches 配置字典
        patch_size = (args.vit_patches_size, args.vit_patches_size)
        if not hasattr(config_vit, 'patches'):
            config_vit.patches = ConfigDict({'size': patch_size})
        else:
            config_vit.patches.size = patch_size

        # 混合模型 (R50) 的 Skip Channels 处理
        if 'R50' in args.vit_name:
            grid_size = args.img_size // args.vit_patches_size
            config_vit.patches.grid = (grid_size, grid_size)
            config_vit.skip_channels = [512, 256, 64, 16] 
        else:
            config_vit.skip_channels = [0, 0, 0, 0]

        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        
        if args.is_pretrain:
            weight_filename = f"{args.vit_name}.npz"
            target_weight_path = f"/work/ly_s/WORK/Segment/model/vit_checkpoint/imagenet21k/{weight_filename}"
            target_url = f"https://storage.googleapis.com/vit_models/imagenet21k/{weight_filename}"
            
            download_pretrained_weights(target_weight_path, target_url)
            
            if os.path.exists(target_weight_path):
                print(f"🧬 正在将权重注入模型...")
                net.load_from(weights=np.load(target_weight_path))
                print(f"✅ 成功加载预训练权重！")

    elif args.model_name == 'resnet':
        net = smp.FPN(encoder_name="resnet34", encoder_weights="imagenet", in_channels=INPUT_CHANNELS, classes=args.num_classes).cuda()
    elif args.model_name == 'att_unet':
        net = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=INPUT_CHANNELS, classes=args.num_classes, decoder_attention_type="scse").cuda()
    elif args.model_name == 'linknet':
        net = smp.Linknet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=INPUT_CHANNELS, classes=args.num_classes).cuda()
    elif args.model_name == 'fpn':
        net = smp.FPN(encoder_name="resnet34", encoder_weights="imagenet", in_channels=INPUT_CHANNELS, classes=args.num_classes).cuda()
    elif args.model_name == 'unet':
        net = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=INPUT_CHANNELS, classes=args.num_classes).cuda()
    elif args.model_name == 'pspnet':
        net = smp.PSPNet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=INPUT_CHANNELS, classes=args.num_classes).cuda()

    # ------------------------------------------------------------------
    # 步骤 6：触发对应的 Trainer
    # ------------------------------------------------------------------
    trainer = {
        'Synapse': trainer_synapse, 
        'OCTA-SS': trainer_OCTA_SS, 
        'DRIVE': trainer_DRIVE,
        'OCTA_3M': trainer_OCTA_3M  # 【新增绑定】
    }
    
    if net is not None:
        trainer[args.dataset](args, net, snapshot_path)
    else:
        print("❌ 模型初始化失败，无法启动训练！")