import argparse
import logging
import os
# 告诉 PyTorch 优先使用可扩展内存块，减少碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
# 在训练开始前清空一次缓存
torch.cuda.empty_cache()
import random
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import subprocess
import sys
import ssl  # 处理下载权限
import urllib.request
import segmentation_models_pytorch as smp
from ml_collections import ConfigDict  # 新增：用于修复配置对象

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse, trainer_OCTA_SS, trainer_DRIVE

# ==============================================================================
# 功能函数：1. 自动下载预训练权重
# ==============================================================================
def download_pretrained_weights(save_path):
    ssl_context = ssl._create_unverified_context()
    url = "https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz"
    
    if not os.path.exists(save_path) or os.path.getsize(save_path) < 1024:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"\n🌐 正在从远程下载预训练权重... \n保存位置: {os.path.abspath(save_path)}")
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
# 功能函数：2. 交互式选择模型
# ==============================================================================
def select_model_interactively():
    models = ['transunet', 'unet', 'att_unet', 'resnet', 'linknet', 'fpn']
    print("\n" + "="*40)
    print("      DRIVE 数据集模型训练选择器")
    print("="*40)
    for i, m in enumerate(models):
        print(f" [{i}] : {m}")
    print("="*40)
    
    while True:
        try:
            choice = input("请输入模型对应的编号 (0-5) [默认 0]: ").strip()
            if choice == "": return models[0]
            idx = int(choice)
            if 0 <= idx < len(models): return models[idx]
            else: print("❌ 无效编号！")
        except ValueError: print("❌ 请输入数字！")

# ==============================================================================
# 功能函数：3. 自动寻找显存剩余最多的 GPU
# ==============================================================================
def get_device_id_with_max_memory():
    """
    通过 nvidia-smi 自动获取当前剩余显存最大的 GPU ID
    """
    try:
        # 获取各显卡剩余显存 (MiB)
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE,
            encoding='utf-8'
        )
        free_memory = [int(x) for x in result.stdout.strip().split('\n')]
        best_id = free_memory.index(max(free_memory))
        print(f"🚀 自动选择最优 GPU: [ID {best_id}] (当前剩余显存: {free_memory[best_id]} MiB)")
        return best_id
    except Exception as e:
        print(f"⚠️ 无法自动获取 GPU 状态 ({e})，默认使用 ID 0")
        return 0

# 参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='/work/ly_s/WORK/Segment/TransUNet-main/model/DRIVE')
parser.add_argument('--root_path', type=str, default='/work/ly_s/WORK/Segment/TransUNet-main/data/DRIVE')
parser.add_argument('--dataset', type=str, default='DRIVE')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--max_epochs', type=int, default=150)

parser.add_argument('--batch_size', type=int, default=1)

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
    # 1. 选择模型
    args.model_name = select_model_interactively()

    # 2. 自动选择显存最大的 GPU
    best_gpu_id = get_device_id_with_max_memory()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
    
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {'root_path': '../data/Synapse/train_npz', 'list_dir': './lists/lists_Synapse', 'num_classes': 9, 'in_channels': 1},
        'OCTA-SS': {'root_path': '/work/ly_s/WORK/Segment/TransUNet-main/data/OCTA-SS', 'list_dir': '', 'num_classes': 2, 'in_channels': 1},
        'DRIVE': {'root_path': args.root_path, 'list_dir': '', 'num_classes': 2, 'in_channels': 3},
    }
    
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    INPUT_CHANNELS = dataset_config[dataset_name]['in_channels']
    
    args.is_pretrain = True if args.model_name == 'transunet' else False
    
    snapshot_path = os.path.join("/work/ly_s/WORK/Segment/TransUNet-main/model", 'TU_' + dataset_name + str(args.img_size), 'TU1')
    if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)
        
    net = None
    if args.model_name in ['transunet', 'kan_transunet']:
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        
        # ==================== 【保持不变：修复 ZeroDivisionError 部分】 ====================
        patch_size = (args.vit_patches_size, args.vit_patches_size)
        if not hasattr(config_vit, 'patches'):
            config_vit.patches = ConfigDict({'size': patch_size})
        else:
            config_vit.patches.size = patch_size

        if 'R50' in args.vit_name:
            grid_size = args.img_size // args.vit_patches_size
            config_vit.patches.grid = (grid_size, grid_size)
            config_vit.skip_channels = [512, 256, 64, 16] 
        else:
            config_vit.skip_channels = [0, 0, 0, 0]
        # ==============================================================================

        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        
        if args.is_pretrain:
            target_weight_path = "./model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
            download_pretrained_weights(target_weight_path)
            
            if os.path.exists(target_weight_path):
                print(f"🧬 正在将权重注入模型...")
                net.load_from(weights=np.load(target_weight_path))
                print(f"✅ 成功加载预训练权重！")

    elif args.model_name == 'unet':
        net = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=INPUT_CHANNELS, classes=args.num_classes).cuda()
    
    # 启动训练
    trainer = {'Synapse': trainer_synapse, 'OCTA-SS': trainer_OCTA_SS, 'DRIVE': trainer_DRIVE}
    if net is not None:
        trainer[dataset_name](args, net, snapshot_path)