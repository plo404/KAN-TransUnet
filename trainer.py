import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_OCTA_SS(args, model, snapshot_path):
    from datasets.dataset_OCTA_SS import Octa_SS_dataset, RandomGenerator
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = Octa_SS_dataset(base_dir=args.root_path, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    
    def get_parameter_groups(model, weight_decay=1e-4):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "spline_scaler" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]

    param_groups = get_parameter_groups(model, weight_decay=0.0001)
    optimizer = optim.SGD(param_groups, lr=base_lr, momentum=0.9)
    
    # 显存优化初始化 (复刻 DRIVE)
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler() 
    accumulation_steps = 4 

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    best_performance = 0.0
    
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        model.train()
        epoch_dice_scores = [] 
        optimizer.zero_grad()
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            # 混合精度前向传播 (复刻 DRIVE)
            with autocast():
                outputs = model(image_batch)
                
                if outputs.shape[-2:] != label_batch.shape[-2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=label_batch.shape[-2:], mode='bilinear', align_corners=False
                    )

                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = (0.5 * loss_ce + 0.5 * loss_dice) / accumulation_steps
            
            # 缩放反向传播 (复刻 DRIVE)
            scaler.scale(loss).backward()
            
            batch_dice = 1.0 - loss_dice.item()
            epoch_dice_scores.append(batch_dice)

            # 梯度累加更新 (复刻 DRIVE)
            if (i_batch + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/total_loss', loss * accumulation_steps, iter_num)
            writer.add_scalar('info/batch_dice', batch_dice, iter_num)

            if iter_num % 20 == 0:
                outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[0, ...] * 255, iter_num)
                writer.add_image('train/GroundTruth', label_batch[0, ...].unsqueeze(0) * 255, iter_num)

        average_epoch_dice = np.mean(epoch_dice_scores)
        logging.info('Epoch %d : Average Dice: %f' % (epoch_num, average_epoch_dice))
        writer.add_scalar('info/epoch_avg_dice', average_epoch_dice, epoch_num)

        if average_epoch_dice > best_performance:
            best_performance = average_epoch_dice
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("🏆 New Best Model! Saved to %s" % save_mode_path)
        
        if epoch_num >= max_epoch - 1:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_DRIVE(args, model, snapshot_path):
    # 1. 引入 DRIVE 数据集类
    from datasets.dataset_DRIVE import Drive_dataset, RandomGenerator
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    # 2. 初始化 DRIVE 数据集
    db_train = Drive_dataset(base_dir=args.root_path, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    
    # 3. 优化器参数分组
    def get_parameter_groups(model, weight_decay=1e-4):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "spline_scaler" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]

    param_groups = get_parameter_groups(model, weight_decay=0.0001)
    optimizer = optim.SGD(param_groups, lr=base_lr, momentum=0.9)
    
    # ==============================================================================
    # 【核心新增：显存优化初始化】
    # ==============================================================================
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler() # 混合精度缩放器
    accumulation_steps = 4 # 梯度累加步数（如果 batch_size=1，等效于 batch_size=4）
    # ==============================================================================

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    best_performance = 0.0
    
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        model.train()
        epoch_dice_scores = [] 
        
        # 确保梯度清零
        optimizer.zero_grad()
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            # 【优化：混合精度前向传播】
            with autocast():
                outputs = model(image_batch)
                
                if outputs.shape[-2:] != label_batch.shape[-2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=label_batch.shape[-2:], mode='bilinear', align_corners=False
                    )

                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                # 损失除以累加步数
                loss = (0.5 * loss_ce + 0.5 * loss_dice) / accumulation_steps
            
            # 【优化：缩放反向传播】
            scaler.scale(loss).backward()
            
            # 计算指标用于 log
            batch_dice = 1.0 - loss_dice.item()
            epoch_dice_scores.append(batch_dice)

            # 【优化：梯度累加更新】
            if (i_batch + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/total_loss', loss * accumulation_steps, iter_num)
            writer.add_scalar('info/batch_dice', batch_dice, iter_num)

            if iter_num % 20 == 0:
                outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[0, ...] * 255, iter_num)
                writer.add_image('train/GroundTruth', label_batch[0, ...].unsqueeze(0) * 255, iter_num)

        # 验证与保存逻辑保持不变
        average_epoch_dice = np.mean(epoch_dice_scores)
        logging.info('Epoch %d : Average Dice: %f' % (epoch_num, average_epoch_dice))
        writer.add_scalar('info/epoch_avg_dice', average_epoch_dice, epoch_num)

        if average_epoch_dice > best_performance:
            best_performance = average_epoch_dice
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("🏆 New Best Model! Saved to %s" % save_mode_path)
        
        if epoch_num >= max_epoch - 1:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_OCTA_3M(args, model, snapshot_path):
    from datasets.dataset_OCTA_3M import OCTA_3M_dataset, RandomGenerator
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = OCTA_3M_dataset(base_dir=args.root_path, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    
    def get_parameter_groups(model, weight_decay=1e-4):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "spline_scaler" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]

    param_groups = get_parameter_groups(model, weight_decay=0.0001)
    optimizer = optim.SGD(param_groups, lr=base_lr, momentum=0.9)
    
    # 显存优化初始化 (复刻 DRIVE)
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler() 
    accumulation_steps = 4 

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    best_performance = 0.0
    
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        model.train()
        epoch_dice_scores = [] 
        optimizer.zero_grad()
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            # 混合精度前向传播 (复刻 DRIVE)
            with autocast():
                outputs = model(image_batch)
                
                if outputs.shape[-2:] != label_batch.shape[-2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=label_batch.shape[-2:], mode='bilinear', align_corners=False
                    )

                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = (0.5 * loss_ce + 0.5 * loss_dice) / accumulation_steps
            
            # 缩放反向传播 (复刻 DRIVE)
            scaler.scale(loss).backward()
            
            batch_dice = 1.0 - loss_dice.item()
            epoch_dice_scores.append(batch_dice)

            # 梯度累加更新 (复刻 DRIVE)
            if (i_batch + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/total_loss', loss * accumulation_steps, iter_num)
            writer.add_scalar('info/batch_dice', batch_dice, iter_num)

            if iter_num % 20 == 0:
                outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[0, ...] * 255, iter_num)
                writer.add_image('train/GroundTruth', label_batch[0, ...].unsqueeze(0) * 255, iter_num)

        average_epoch_dice = np.mean(epoch_dice_scores)
        logging.info('Epoch %d : Average Dice: %f' % (epoch_num, average_epoch_dice))
        writer.add_scalar('info/epoch_avg_dice', average_epoch_dice, epoch_num)

        if average_epoch_dice > best_performance:
            best_performance = average_epoch_dice
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("🏆 New Best Model! Saved to %s" % save_mode_path)
        
        if epoch_num >= max_epoch - 1:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"