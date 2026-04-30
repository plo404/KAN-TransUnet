import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# [修改点1] 导入 dataset_DRIVE.py
# 请确保 datasets/dataset_DRIVE.py 文件存在且类名为 Drive_dataset
from datasets.dataset_DRIVE import Drive_dataset

from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()

# [修改点] 设置数据集根目录
# Drive_dataset 会自动根据 split="test" 拼接出 /test/image 和 /test/mask
# 所以这里指向 .../data/DRIVE 即可
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

# [修改点] 指定结果保存路径
parser.add_argument('--test_save_dir', type=str, 
                    default='/home/ly_s/WORK/Segment/TransUNet-main/data/DRIVE/test/KAN-TransUnet', 
                    help='saving prediction as nii!')

parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    # [修改点2] 使用新的 Drive_dataset 类
    db_test = args.Dataset(base_dir=args.volume_path, split="test")
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # 获取数据
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        case_name = sampled_batch['case_name'][0]

        # [维度处理]
        # TransUNet 的 utils.test_single_volume 期望输入形状为 (1, C, H, W) 或者是 3D 形式
        # 这里我们将 (B, H, W) -> (B, 1, H, W) 以适配接口
        if len(image.shape) == 3:
            image = image.unsqueeze(1)
        if len(label.shape) == 3:
            label = label.unsqueeze(1)
        
        # 执行推理
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

    # [修改点3] 注册 DRIVE Dataset
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
    args.is_pretrain = True # 测试时需要设为 True 吗？其实这个参数主要用于 train 时加载 imagenet 权重。这里无所谓，因为下面是手动 load_state_dict

    args.exp = 'TU_' + dataset_name + str(args.img_size)
    
    # [修改点] 指定权重文件的绝对路径
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
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    args.is_savenii = True 
    # [修改点] 指定保存路径
    args.test_save_dir = '/home/ly_s/WORK/Segment/TransUNet-main/data/DRIVE/test/KAN-TransUnet'
    
    test_save_path = args.test_save_dir
    os.makedirs(test_save_path, exist_ok=True)
    
    inference(args, net, test_save_path)