import os
import sys
import collections
from os.path import join as pjoin
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import scipy.misc as m
from torchvision import transforms
import skimage
from skimage import io,transform

class hyperLoader(data.Dataset):
    """docstring for hyperLoader"""

    def __init__(self, root, split="train"):
        root = '/data/home/ywen/sxx/hyper/'
        self.root = root
        self.split = split
        self.n_classes = 2
        self.files = collections.defaultdict(list)
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.Normalize([0.196, 0.179, 0.323], [0.257, 0.257, 0.401])
            ]
        )

        for split in ["train", "val", "test"]:
            path = os.path.join(self.root, split+'.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])


    def __getitem__(self, index):
        img_name = self.files[self.split][index]

        # multi-modality
        img_path = self.root + 'img/' + img_name
        lbl_path = self.root + 'gt/' + img_name
        
        img = m.imread(img_path)
        lbl = m.imread(lbl_path)
        #print(np.array(img).shape)
        newsize = 128
        img = m.imresize(img,[newsize,newsize], interp='bilinear', mode=None)
        lbl = m.imresize(lbl,[newsize,newsize], interp='bilinear', mode=None)
        
        img, lbl = self.transform(img, lbl)
        
        return img, lbl, img_name


    def transform(self, img, lbl):
        img = self.tf(img)
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def get_brainweb_colormap(self):
        return np.asarray([[0, 0, 0], [255, 255, 255]])

    def encode_segmap(self, mask):
        mask = mask.astype(np.uint8)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for ii, label in enumerate(self.get_brainweb_colormap()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
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


