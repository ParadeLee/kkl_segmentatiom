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
from torchvision import transforms


class mrbrainsLoader_new_ann(data.Dataset):

    def __init__(self, root, split="train"):
        self.root = root
        self.split = split
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
        # t1 = m.imread(t1_path)
        # t2 = m.imread(t2_path)
        # pd = m.imread(pd_path)
        img = m.imread(t1_path)
        lbl = m.imread(lbl_path)
        # t1_g = m.imread(t1_path, flatten=True)
        # t2_g = m.imread(t2_path, flatten=True)
        # pd_g = m.imread(pd_path, flatten=True)
        gray = m.imread(t1_path, flatten=True)

        # newsize = 224
        # t1 = m.imresize(t1, [newsize, newsize], interp='bilinear', mode=None)
        # t2 = m.imresize(t2, [newsize, newsize], interp='bilinear', mode=None)
        # pd = m.imresize(pd, [newsize, newsize], interp='bilinear', mode=None)
        # lbl = m.imresize(lbl, [newsize, newsize], interp='bilinear', mode=None)

        # img = np.stack((t1, t2, pd), axis=0)
        img = np.stack((img, img, img), axis=0)
        gray = np.array([gray])
        # gray = np.stack((t1_g, t2_g, pd_g), axis=0)
        img, gray, lbl = self.transform(img, gray, lbl)

        # gray = img.convert('L')

        return img, gray, lbl, img_name


    def transform(self, img, gray, lbl):
        # img = self.tf(img)
        img = img.astype(float) / 255.0
        img = torch.from_numpy(img).float()
        gray = gray.astype(float) / 255.0
        gray = torch.from_numpy(gray).float()
        lbl = torch.from_numpy(lbl).long()

        return img, gray, lbl

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
    root = 'C:/Users/86130/Desktop/kkl_seg2021/kkl_segmentatiom/datasets/mrbrains/'

    t_loader = mrbrainsLoader_new_ann(
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

