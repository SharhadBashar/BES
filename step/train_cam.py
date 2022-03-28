import torch
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

from net.resnet50_cam import CAM

import voc12.dataloader
from misc import pyutils, torchutils



def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            # here?
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss1.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss')))

    return


def run(args):
    mean = torch.Tensor([0.485, 0.456, 0.406])[None, ..., None, None].cuda()
    std = torch.Tensor([0.229, 0.224, 0.225])[None, ..., None, None].cuda()


    model = CAM()
    '''
    original resize_long is set as '(320, 640)', parts of image would be cropped if 'resize_long' is bigger than 
    'crop_size'. To prevent that, 'resize_long' is set as '(320, 512)'.
    '''
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_aug_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 512), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)
            size = pack['size'].cuda()

            x = model(img)

            img_size = (img.cuda() - mean) / std

            x_size = model(img_size)
            
            weights = [1e-3, 1e-2, 1e-1, 1, 10, 20, 50, 100, 200, 500]
            
            N_c, C = x_size.shape
            sum_N_c = 0

            for n in range(N_c):
                sum_C = torch.sum((x_size[n] - size[n, 1:].cuda()) ** 2)
                sum_N_c += sum_C / C

            # if (ep <= 1 and step <= 200): weight = weights[0]
            # else: weight = weights[2]
            weight = weights[0]

            loss_size = weight * sum_N_c / N_c
            

            loss_cam = F.multilabel_soft_margin_loss(x, label) 

            loss = loss_cam + loss_size

            avg_meter.add({'loss': loss.item(), 'loss_cam': loss_cam.item(), 'loss_size_cam': loss_size})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print(step, weight,
                      'step:%5d /%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss')),
                      'loss_cam:%.4f' % (avg_meter.pop('loss_cam')),
                      'loss_size:%.4f' % (avg_meter.pop('loss_size_cam')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            validate(model, val_data_loader)
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.cam_weights_name)
    torch.cuda.empty_cache()
