import os
import collections
from os.path import join as pjoin
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import scipy.misc as m
from PIL import Image
import torchvision.transforms
import mmcv


class mrbrainsLoader_new(data.Dataset):

    def __init__(self, root, split="train"):
        self.root = root
        self.split = split
        self.resize = (224, 224)
        self.pad_val = 0
        self.seg_pad_val = 255
        self.n_classes = 4
        self.files = collections.defaultdict(list)
        # self.tf = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.196, 0.179, 0.323], [0.257, 0.257, 0.401])
        #     ]
        # )

        for split in ["train", "val", "trainval"]:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])


    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        # img_path = self.root + 'imgs/rgb/' + img_name + '.bmp'
        t1_path = self.root + 'imgs/t1/' + img_name + '.bmp'
        t2_path = self.root + 'imgs/t1ir/' + img_name + '.bmp'
        pd_path = self.root + 'imgs/t2flair/' + img_name + '.bmp'
        lbl_path = self.root + 'gt/' + img_name + '.bmp'

        # np:(h,w,c)
        # img = m.imread(img_path)
        t1 = m.imread(t1_path)
        t2 = m.imread(t2_path)
        pd = m.imread(pd_path)
        H, W = t1.shape[0:2]
        rsize = 224
        # bbox = np.array([(H-rsize)/2, (W-rsize)/2, (H+rsize)/2, (W+rsize)/2])
        bbox = np.array([8, 8, 231, 231])
        t1 = mmcv.imcrop(t1, bbox)
        t2 = mmcv.imcrop(t2, bbox)
        pd = mmcv.imcrop(pd, bbox)
        # t1 = mmcv.impad(t1, shape=self.resize, pad_val=self.pad_val)
        # t2 = mmcv.impad(t2, shape=self.resize, pad_val=self.pad_val)
        # pd = mmcv.impad(pd, shape=self.resize, pad_val=self.pad_val)
        lbl = m.imread(lbl_path)
        # lbl = mmcv.impad(lbl, shape=self.resize, pad_val=self.seg_pad_val)
        lbl = mmcv.imcrop(lbl, bbox)
        img = np.stack((t1, t2, pd), axis=0)
        # img = np.stack((img, img, img), axis=0)
        img, lbl = self.transform(img, lbl)


        return img, lbl, img_name


    def transform(self, img, lbl):
        # img = self.tf(img)
        img = img.astype(float) / 255.0
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def get_brainweb_colormap(self):
        return np.asarray([[0, 0, 0], [255, 255, 255], [92, 179, 179], [221, 218, 93]])

    def encode_segmap(self, mask):
        mask = mask.astype(np.uint8)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for ii, label in enumerate(self.get_brainweb_colormap()):
            label_mask[np.where((mask == (np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8) + label)))] = ii
        label_mask = label_mask.astype(np.uint8)
        return label_mask


    def decode_segmap(self, label_mask, plot=False):

        label_colors = self.get_brainweb_colormap()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colors[ll, 0]
            g[label_mask == ll] = label_colors[ll, 1]
            b[label_mask == ll] = label_colors[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb.astype(np.uint8)

    def setup_annotations(self):
        target_path = pjoin(self.root, 'class4')
        if not os.path.exists(target_path): os.makedirs(target_path)

        print("Pre-encoding segmentaion masks...")
        for ii in tqdm(self.files['trainval']):
            fname = ii + '.bmp'
            lbl_path = pjoin(self.root, 'class10', 'crisp_' + fname)
            lbl = self.encode_segmap(m.imread(lbl_path))
            m.imsave(pjoin(target_path, 'pre_encoded', 'c4_' + fname), lbl)
            rgb = self.decode_segmap(lbl)
            m.imsave(pjoin(target_path, 'rgb_decoded', 'c4_rgb_' + fname), rgb)


def debug_load():
    # root = 'C:/Users/86130/Desktop/kkl_seg2021/kkl_segmentatiom/datasets/mrbrains/'
    root = 'C:/Users/parade/Desktop/kkl_segmentatiom/datasets/mrbrains/'

    t_loader = mrbrainsLoader_new(
		root,
		split='trainval')

    n_classes = t_loader.n_classes

    trainLoader = data.DataLoader(t_loader,
								  batch_size=1,
								  num_workers=4,
								  shuffle=True)

    for (images, labels, img_name) in trainLoader:
        # m.imsave(pjoin('/home/jwliu/disk/kxie/CNN_LSTM/dataset/brainweb/imgs/rgb', '{}.bmp'.format(img_name)),images)

        labels = np.squeeze(labels.data.numpy())
        decoded = t_loader.decode_segmap(labels, plot=False)
        m.imsave(pjoin('/home/jwliu/disk/kxie/CNN_LSTM/result_image_when_training/mrbrains', '{}.bmp'.format(img_name[0])), decoded)
        print('.')

        # tensor2numpy
        # print(img_name[0])
        # out = images.numpy() * 255
        # out = out.astype('uint8')
        # out = np.squeeze(out)
        #
        # lbl = labels.numpy()
        # lbl = lbl.astype('uint8')
        # lbl = np.squeeze(lbl)

        # chw->hwc
        # out = np.transpose(out, (1,2,0))

        # io.imshow(out)
        # plt.show()
        #
        # print(img_name)
        # print(images)
        # print(labels)


if __name__ == '__main__':
    debug_load()

