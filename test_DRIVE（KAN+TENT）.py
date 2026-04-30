import argparse
import logging
import os
import random
import sys
import copy  # [新增] 用于备份权重
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim # [新增] 优化器
import torch.nn.functional as F # [新增] 用于 resize
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入 dataset_DRIVE
from datasets.dataset_DRIVE import Drive_dataset

from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# -------------------- TENT 超参数 --------------------
TENT_STEPS = 5        # 迭代步数
TENT_LR = 1e-4        # 学习率
TENT_OPTIM = "Adam"   
REG_LAMBDA = 1e-3     # L2 回拉强度
# ----------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--volume_path', type=str,
                    default='/home/ly_s/WORK/Segment/TransUNet-main/data/DRIVE', 
                    help='root dir for validation volume data')

parser.add_argument('--dataset', type=str,
                    default='DRIVE', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

# [修改] TENT 结果保存路径
parser.add_argument('--test_save_dir', type=str, 
                    default='/home/ly_s/WORK/Segment/TransUNet-main/data/DRIVE/test/result_tent', 
                    help='saving prediction as nii!')

parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')

# [新增] TENT 参数
parser.add_argument('--tent_steps', type=int, default=TENT_STEPS, help='TENT adaptation steps')
parser.add_argument('--tent_lr', type=float, default=TENT_LR, help='TENT learning rate')

args = parser.parse_args()


# ==========================================================================================
#  TENT 模块实现
# ==========================================================================================

def get_tent_params(model):
    """收集 BN 参数，若无则收集最后层参数"""
    for p in model.parameters():
        p.requires_grad = False

    tent_params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.weight is not None:
                m.weight.requires_grad = True
                tent_params.append(m.weight)
            if m.bias is not None:
                m.bias.requires_grad = True
                tent_params.append(m.bias)

    if len(tent_params) == 0:
        named = list(model.named_parameters())
        count = 0
        for name, p in reversed(named):
            if not p.requires_grad:
                p.requires_grad = True
                tent_params.append(p)
                count += 1
            if count >= 4:
                break
    return tent_params

def backup_bn_stats(model):
    backups = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if hasattr(m, 'running_mean') and hasattr(m, 'running_var'):
                backups.append((m, m.running_mean.clone(), m.running_var.clone()))
    return backups

def restore_bn_stats(backups):
    for m, rm, rv in backups:
        m.running_mean.data.copy_(rm)
        m.running_var.data.copy_(rv)

def tent_adapt_segmentation(model, image, steps, lr, optimizer_name=TENT_OPTIM, reg_lambda=REG_LAMBDA):
    tent_params = get_tent_params(model)
    if len(tent_params) == 0: return

    bn_backups = backup_bn_stats(model)
    orig_vals = {p: p.detach().clone() for p in tent_params} if reg_lambda > 0 else {}

    model.train() 

    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(tent_params, lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(tent_params, lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        outputs = model(image) # (B, Num_Classes, H, W)
        
        probs = torch.softmax(outputs, dim=1) 
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
        loss = entropy.mean()

        if reg_lambda > 0:
            reg = 0.0
            for p in tent_params:
                reg = reg + (p - orig_vals[p]).pow(2).sum()
            loss = loss + reg_lambda * reg

        loss.backward()
        optimizer.step()

    restore_bn_stats(bn_backups)
    model.eval()

# ==========================================================================================


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    
    # 备份初始模型权重 (Episodic Adaptation)
    logging.info("Backing up initial model state for episodic adaptation...")
    initial_model_state = copy.deepcopy(model.state_dict())
    
    metric_list = 0.0
    
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        case_name = sampled_batch['case_name'][0]

        # [维度处理] (B, H, W) -> (B, 1, H, W)
        if len(image.shape) == 3:
            image = image.unsqueeze(1)
        if len(label.shape) == 3:
            label = label.unsqueeze(1)
        
        # ------------------- TENT Adaptation -------------------
        # 1. 重置模型状态
        model.load_state_dict(initial_model_state)
        image_cuda = image.cuda()
        
        # 2. [关键修复] 强制 Resize 到 224x224
        # ViT 的位置编码是固定的 (196个 patch)，如果 DRIVE 图像尺寸不是 224，
        # 直接输入 TENT 会报 "tensor a (36) must match tensor b (196)" 错误。
        if image_cuda.size(2) != args.img_size or image_cuda.size(3) != args.img_size:
            # 使用双线性插值缩放到 224x224
            image_tent = F.interpolate(image_cuda, size=(args.img_size, args.img_size), 
                                       mode='bilinear', align_corners=False)
        else:
            image_tent = image_cuda
            
        # 3. 运行 TENT (使用 Resize 后的图像更新参数)
        tent_adapt_segmentation(model, image_tent, steps=args.tent_steps, lr=args.tent_lr)
        # -------------------------------------------------------

        # 4. 推理 (使用原始尺寸图像，test_single_volume 会处理切片)
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 注册 DRIVE Dataset
    dataset_config = {
        'DRIVE': {
            'Dataset': Drive_dataset,
            'volume_path': args.volume_path, # 对应 /data/DRIVE
            'list_dir': args.list_dir,
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    
    dataset_name = 'DRIVE' 
    args.dataset = dataset_name
    
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    args.exp = 'TU_' + dataset_name + str(args.img_size)
    
    # [修改] 指定权重文件的绝对路径
    snapshot_path = "/home/ly_s/WORK/Segment/TransUNet-main/model/TU_DRIVE512/TU1/best_model.pth"

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = snapshot_path
    if not os.path.exists(snapshot):
        print(f"Error: Weight file not found at {snapshot}")
        sys.exit(1)
        
    net.load_state_dict(torch.load(snapshot))
    print(f"Loaded model from {snapshot}")
    
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+"_TENT.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(f"TENT enabled: steps={args.tent_steps}, lr={args.tent_lr}")

    args.is_savenii = True 
    # [修改] 指定保存路径
    args.test_save_dir = '/home/ly_s/WORK/Segment/TransUNet-main/data/DRIVE/test/KAN-TU-TENT'
    
    test_save_path = args.test_save_dir
    os.makedirs(test_save_path, exist_ok=True)
    
    inference(args, net, test_save_path)