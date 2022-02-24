import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
from torch.utils import data
from tqdm import tqdm
from models import get_model
from loss import get_loss_function
from loss import create_losses
from loader import get_loader
from utils import get_logger
from metrics import runningScore, averageMeter
from schedulers import get_scheduler
from optimizers import get_optimizer
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import datetime
from scipy import ndimage
import math
import cv2
np.set_printoptions(threshold=np.inf)

def direct_field(a, norm=True):
    """ a: np.ndarray, (h, w)
    """
    if a.ndim == 3:
        a = np.squeeze(a)

    h, w = a.shape
    
    a_Image = Image.fromarray(np.uint8(a))
    a = a_Image.resize((w, h), Image.NEAREST)
    a = np.array(a)
    
    accumulation = np.zeros((2, h, w), dtype=np.float32)
    for i in np.unique(a)[1:]:
        img = (a == i).astype(np.uint8)
        dst, labels = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
        index = np.copy(labels)
        index[img > 0] = 0
        place = np.argwhere(index > 0)
        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, h, w))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(img.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel
        if norm:
            dr = np.sqrt(np.sum(diff**2, axis = 0))
        else:
            dr = np.ones_like(img)

        # direction = np.zeros((2, h, w), dtype=np.float32)
        # direction[0, b>0] = np.divide(diff[0, b>0], dr[b>0])
        # direction[1, b>0] = np.divide(diff[1, b>0], dr[b>0])

        direction = np.zeros((2, h, w), dtype=np.float32)
        direction[0, img>0] = np.divide(diff[0, img>0], dr[img>0])
        direction[1, img>0] = np.divide(diff[1, img>0], dr[img>0])

        accumulation[:, img>0] = 0
        accumulation = accumulation + direction
    return accumulation

def train(cfg, logger):

    # Setup Seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))
    
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    print(ts)
    logger.info("Start time {}".format(ts))
    # Setup Device
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:{}".format(cfg["training"]["gpu_idx"]) if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)

    # Setup Dataloader
    '''if(cfg["data"]["dataset"] == 'Brats'):
        cfg["training"]["batch_size"] = 8'''
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        split=cfg["data"]["train_split"],
    )

    v_loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
    )
    
    e_loader = data_loader(
        data_path,
        split=cfg["data"]["test_split"],
    )

    n_classes = t_loader.n_classes
    n_val = len(v_loader.files['val'])
    n_test = len(e_loader.files['test'])

    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"]
    )
    
    testloader = data.DataLoader(
        e_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"]
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes, n_val)
    running_metrics_test = runningScore(n_classes, n_test)

    # Setup Model
    model = get_model(cfg["model"], n_classes).to(device)
    model = torch.nn.DataParallel(model, device_ids=[cfg["training"]["gpu_idx"]])

    # Setup Optimizer, lr_scheduler and Loss Function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    # Resume Trained Model
    
    '''cfg["training"]["resume"] = 'UNetDRNN_BrainWeb_best_2021_04_28_19_51_49.pkl'
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"],map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))
    '''
    
    # Start Training
    val_loss_meter = averageMeter()
    time_meter = averageMeter()
    name = cfg["model"]["arch"]
    start_iter = 0
    best_dice = -100.0
    best_loss = 100
    i = start_iter
    flag = True
    criterion = create_losses.Total_loss(boundary=False)
    while i <= cfg["training"]["train_iters"] and flag:
        for (images, labels, img_name) in trainloader:
            i += 1
            start_ts = time.time()
            #print(labels)
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            net_out = model(images)
            
            if (name == "UNet3"):
                loss = loss_fn(input=net_out, target=labels)
                pred = net_out.data.max(1)[1].cpu().numpy()
            else:
                seg_out, df_out = net_out[:2]
                # add Auxiliary Segmentation
                if len(net_out) >= 3 and net_out[2] is not None:
                    auxseg_out = net_out[2]
                    auxseg_loss = F.cross_entropy(auxseg_out, labels)
                else:
                    auxseg_loss = torch.tensor([0.], dtype=torch.float32, device=device)

                alpha = 1.0 
                if net_out[0] is not None and net_out[1] is not None:
                    seg_loss = loss_fn(input=seg_out, target=labels)
                    # direction field Loss
                    gts_df = direct_field(labels.numpy()[0], norm=True)
                    gts_df = torch.from_numpy(gts_df)
            
                    if(len(list(gts_df.size()))<4):
                        gts_df = gts_df.unsqueeze(0)
                    if(len(list(labels.size()))<4):
                        labels = labels.unsqueeze(0)
                    
                    df_loss = criterion(df_out, gts_df, labels)
                    
                    loss = alpha*(seg_loss + 1. * df_loss + 1.0*auxseg_loss)
                else:
                    seg_loss = loss_fn(input=net_out[2], target=labels)
                    loss = seg_loss
            #print(seg_loss, df_loss, auxseg_loss)

            if(str(loss.item())=='nan'):
                i -= 1;continue
            if(math.isnan(loss.item())):
                i -= 1;continue
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            time_meter.update(time.time() - start_ts)
            
            # print train loss
            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    time_meter.avg / cfg["training"]["batch_size"],
                )
                print(print_str)
                logger.info(print_str)
                time_meter.reset()
                
                #sys.exit(0)
            # validation
            if ((i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"]["train_iters"]):
            #if i==1:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val, img_name_val) in tqdm(enumerate(valloader)):
                        
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        net_out = model(images_val)
                        
                        if (name == "UNet3"):
                            val_loss = loss_fn(input=net_out, target=labels_val)
                            pred = net_out.data.max(1)[1].cpu().numpy()
                        else:
                            seg_out, df_out = net_out[:2]
                            # add Auxiliary Segmentation
                            if len(net_out) >= 3 and net_out[2] is not None:
                                auxseg_out = net_out[2]
                                auxseg_loss = F.cross_entropy(auxseg_out, labels_val)
                            else:
                                auxseg_loss = torch.tensor([0.], dtype=torch.float32, device=device)

                            alpha = 1.0 
                            
                            if net_out[0] is not None and net_out[1] is not None:
                                seg_loss = loss_fn(input=seg_out, target=labels_val)
                                # direction field Loss
                                gts_df = direct_field(labels_val.numpy()[0], norm=True)
                                gts_df = torch.from_numpy(gts_df)
                        
                                if(len(list(gts_df.size()))<4):
                                    gts_df = gts_df.unsqueeze(0)
                                if(len(list(labels_val.size()))<4):
                                    labels_val = labels_val.unsqueeze(0)
                                df_loss = criterion(df_out, gts_df, labels_val)
                                
                                val_loss = alpha*(seg_loss + 1. * df_loss + 1.0*auxseg_loss)
                            else:
                                seg_out = net_out[2]
                                seg_loss = loss_fn(input=seg_out, target=labels_val)
                                val_loss = seg_loss
                            
                            if(str(val_loss.item())=='nan'):
                                val_loss = alpha*(seg_loss + 1.0*auxseg_loss)
                            elif(math.isnan(val_loss.item())):
                                val_loss = alpha*(seg_loss + 1.0*auxseg_loss)

                            seg_out = F.softmax(seg_out, dim=1)
                            _, pred = torch.max(seg_out, 1)
                            pred = pred.unsqueeze(1).numpy()

                        gt = labels_val.data.cpu().numpy()
                        running_metrics_val.update(gt, pred, i_val)
                        val_loss_meter.update(val_loss.item())
                #sys.exit(0)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))
                if(val_loss_meter.avg < best_loss):
                    best_loss = val_loss_meter.avg
                # print val metrics
                score, class_dice = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))

                for k, v in class_dice.items():
                    logger.info("{}: {}".format(k, v))

                val_loss_meter.reset()
                running_metrics_val.reset()
                #sys.exit(0)
                # save model
                if score["Dice : \t"] >= best_dice:
                    best_dice = score["Dice : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_dice": best_dice,
                    }
                    save_path = os.path.join(
                        cfg["training"]["model_dir"], "{}_{}_{}.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"], ts),
                    )
                    #save_path = os.path.join(
                    #    cfg["training"]["model_dir"], "{}_{}.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    #)
                    print('Best val acc = ', score["Dice : \t"])
                    torch.save(state, save_path)
                #sys.exit(0)
            # for testing
            if (i + 1) == cfg["training"]["train_iters"]:
                cfg["training"]["resume"] = './trained/' + cfg["model"]["arch"] + '_' + cfg["data"]["dataset"] +'_'+ ts + '.pkl'
                if cfg["training"]["resume"] is not None:
                    if os.path.isfile(cfg["training"]["resume"]):
                        logger.info("Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"]))
                    checkpoint = torch.load(cfg["training"]["resume"],map_location=device)
                    model.load_state_dict(checkpoint["model_state"])
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    scheduler.load_state_dict(checkpoint["scheduler_state"])
                    start_iter = checkpoint["epoch"]
                    logger.info("Loaded checkpoint '{}' (iter {})".format(cfg["training"]["resume"], checkpoint["epoch"]))
                else:
                    logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))
                model.eval()
                with torch.no_grad():
                    for i_test, (images_test, labels_test, img_name_test) in tqdm(enumerate(testloader)):
                        
                        images_test = images_test.to(device)
                        labels_test = labels_test.to(device)
                        net_out = model(images_test)
                        
                        if (name == "UNet3"):
                            pred = net_out.data.max(1)[1].cpu().numpy()
                        else:
                            seg_out, df_out = net_out[:2]
                            
                            seg_out = F.softmax(seg_out, dim=1)
                            _, pred = torch.max(seg_out, 1)
                            pred = pred.unsqueeze(1).numpy()

                        gt = labels_test.data.cpu().numpy()

                        running_metrics_test.update(gt, pred, i_test)

                logger.info("Testing Result:")
                print("Testing Result:")
                # print test metrics
                score, class_dice = running_metrics_test.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))

                for k, v in class_dice.items():
                    logger.info("{}: {}".format(k, v))

                running_metrics_test.reset()
                
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./configs/unetrnn_brainweb.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    if not os.path.exists(logdir): os.makedirs(logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let's go!")

    train(cfg, logger)



