import argparse
import time
import torch
import os
import numpy as np
from torch.utils import data
from tqdm import tqdm
import random
import shutil
from torch.autograd import Variable
from loss import get_loss_function
from loss.cross_entropy2d import cross_entropy2d
from models import get_model
from optimizers import get_optimizer
from schedulers import get_scheduler
from utils import get_logger
from loader import get_loader
from metrics import runningScore, averageMeter
from utils_data import JointTransform2D, ImageToImage2D, Image2D
from torch.utils.data import DataLoader
import yaml
import math
import cv2
from PIL import Image
import datetime
from scipy import ndimage
import torch.nn.functional as F


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
        dst, labels = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
                                                      labelType=cv2.DIST_LABEL_PIXEL)
        index = np.copy(labels)
        index[img > 0] = 0
        place = np.argwhere(index > 0)
        nearCord = place[labels - 1, :]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, h, w))
        nearPixel[0, :, :] = x
        nearPixel[1, :, :] = y
        grid = np.indices(img.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel
        if norm:
            dr = np.sqrt(np.sum(diff ** 2, axis=0))
        else:
            dr = np.ones_like(img)

        # direction = np.zeros((2, h, w), dtype=np.float32)
        # direction[0, b>0] = np.divide(diff[0, b>0], dr[b>0])
        # direction[1, b>0] = np.divide(diff[1, b>0], dr[b>0])

        direction = np.zeros((2, h, w), dtype=np.float32)
        direction[0, img > 0] = np.divide(diff[0, img > 0], dr[img > 0])
        direction[1, img > 0] = np.divide(diff[1, img > 0], dr[img > 0])

        accumulation[:, img > 0] = 0
        accumulation = accumulation + direction
    return accumulation
def train(cfg, logger):

    #Setup Seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    print(ts)
    logger.info("Start time {}".format(ts))
    #setup Device
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:{}".format(cfg["training"]["gpu_idx"]) if torch.cuda.is_available() else "cpu")

    #Setup Augmentations
    augmentation = cfg["training"].get("augmentations",None)

    #setup Dataloader
    '''if(cfg["data"]["dataset"] == 'Brats'):
            cfg["training"]["batch_size"] = 8'''
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        split=cfg["data"]["train_split"],
    )
    # train data

    v_loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
    )
    # val data

    # e_loader = data_loader(
    #     data_path,
    #     split=cfg["data"]["test_split"],
    # )
    # trainortext_code data

    n_classes = t_loader.n_classes
    n_val = len(v_loader.files['val'])
    # n_test = len(e_loader.files['trainortext_code'])
    # tf_train = JointTransform2D(crop=None, p_flip=0.5, color_jitter_params=None, long_mask=True)
    # tf_val = JointTransform2D(crop=None, p_flip=0, color_jitter_params=None, long_mask=True)
    # train_dataset = ImageToImage2D(t_loader, tf_train)
    # val_dataset = ImageToImage2D(v_loader, tf_val)
    dataloader = DataLoader(t_loader, batch_size=4, shuffle=True)
    valloader = DataLoader(v_loader, 1, shuffle=True)


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

    # testloader = data.DataLoader(
    #     e_loader,
    #     batch_size=cfg["training"]["batch_size"],
    #     num_workers=cfg["training"]["n_workers"]
    # )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes, n_val)
    # running_metrics_test = runningScore(n_classes, n_test)

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
    epoch = start_iter
    flag = True
    direc = cfg["training"]["model_dir"]
    while epoch <= cfg["training"]["train_iters"] and flag:
        '''
        for (images, labels, img_name) in trainloader:
            epoch += 1
            start_ts = time.time()
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)

            # print train loss
            if (epoch +1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    epoch + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    time_meter.avg / cfg["training"]["batch_size"],
                )

                print(print_str)
                logger.info(print_str)
                time_meter.reset()

            # validation
            if (epoch + 1) % cfg["training"]["val_interval"] == 0 or (epoch + 1) == cfg["training"]["train_iters"]:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val, img_name_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred, i_val)
                        val_loss_meter.update(val_loss.item())

                logger.info("Iter %d Loss: %.4f" % (epoch + 1, val_loss_meter.avg))

                # print val metrics
                score, class_dice = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))

                for k, v in class_dice.items():
                    logger.info("{}: {}".format(k, v))

                val_loss_meter.reset()
                running_metrics_val.reset()

                # save model
                if score["Dice : \t"] >= best_dice:
                    best_dice = score["Dice : \t"]
                    state = {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_dice": best_dice,
                    }
                    save_path = os.path.join(
                        cfg["training"]["model_dir"], "{}_{}.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)

            if (epoch + 1) == cfg["training"]["train_iters"]:
                flag = False
                break
        '''
        # MedT training
        epoch_running_loss = 0
        epoch += 1
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):
            X_batch = Variable(X_batch.to(device))
            y_batch = Variable(y_batch.to(device))

            # ===================forward=====================

            output = model(X_batch)

            tmp2 = y_batch.detach().cpu().numpy()
            tmp = output.detach().cpu().numpy()
            tmp[tmp >= 0.5] = 1
            tmp[tmp < 0.5] = 0
            tmp2[tmp2 > 0] = 1
            tmp2[tmp2 <= 0] = 0
            tmp2 = tmp2.astype(int)
            tmp = tmp.astype(int)

            yHaT = tmp
            yval = tmp2
            print(yHaT[0, 1, :, :])
            loss = cross_entropy2d(output, y_batch)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_running_loss += loss.item()

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch, cfg["training"]["train_iters"], epoch_running_loss / (batch_idx + 1)))

        if epoch == 10:
            for param in model.parameters():
                param.requires_grad = True
        if (epoch % cfg["training"]["print_interval"]) == 0:
            for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
                # print(batch_idx)
                if isinstance(rest[0][0], str):
                    image_filename = rest[0][0]
                else:
                    image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

                X_batch = Variable(X_batch.to(device))
                y_batch = Variable(y_batch.to(device))
                # start = timeit.default_timer()
                y_out = model(X_batch)
                # stop = timeit.default_timer()
                # print('Time: ', stop - start)
                tmp2 = y_batch.detach().cpu().numpy()
                tmp = y_out.detach().cpu().numpy()
                tmp[tmp >= 0.5] = 1
                tmp[tmp < 0.5] = 0
                tmp2[tmp2 > 0] = 1
                tmp2[tmp2 <= 0] = 0
                tmp2 = tmp2.astype(int)
                tmp = tmp.astype(int)

                # print(np.unique(tmp2))
                yHaT = tmp
                yval = tmp2

                epsilon = 1e-20

                del X_batch, y_batch, tmp, tmp2, y_out

                yHaT[yHaT == 1] = 255
                yval[yval == 1] = 255

                fulldir = direc + "/{}/".format(epoch)
                # print(fulldir+image_filename)
                if not os.path.isdir(fulldir):
                    os.makedirs(fulldir)

                cv2.imwrite(fulldir + image_filename + ".png", yHaT[0, 1, :, :])
                # cv2.imwrite(fulldir+'/gt_{}.png'.format(count), yval[0,:,:])
            fulldir = direc + "/{}/".format(epoch)
            torch.save(model.state_dict(), fulldir + cfg["model"]["arch"] + ".pth")
            torch.save(model.state_dict(), direc + "final_model.pth")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="condig")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./configs/train_dataset.yml",
        help="Configuration file to use"
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1,100000)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    if not os.path.exists(logdir): os.makedirs(logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)
    logger = get_logger(logdir)
    logger.info("Let's go!")

    train(cfg, logger)