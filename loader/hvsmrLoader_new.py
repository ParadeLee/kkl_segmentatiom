import collections
from os.path import join as pjoin
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import scipy.misc as m
import imageio
import mmcv



class hvsmrLoader_new(data.Dataset):
    """
    docstring for hvsmrLoader
    segmentation of the blood pool and ventricular myocaedium
    """

    def __init__(self, root, split="train"):
        # root = '/data/home/ywen/lk/datasets/hvsmr2016/'
        self.root = root
        self.split = split

        self.resize = (256, 256)
        self.pad_val = 0
        self.seg_pad_val = 255

        self.n_classes = 3
        # self.n_classes = 4  # padding
        self.files = collections.defaultdict(list)

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
        img_path = self.root + 'imgs/' + img_name + '.bmp'
        lbl_path = self.root + 'gt/' + img_name + '.bmp'

        # np:(h,w,c)
        img = imageio.imread(img_path, as_gray=False, pilmode="RGB")
        lbl = imageio.imread(lbl_path)

        padding_img = mmcv.impad(img, shape=self.resize, pad_val=self.pad_val)
        padding_lbl = mmcv.impad(lbl, shape=self.resize, pad_val=self.seg_pad_val)

        # bbox = np.array([0, 0, 223, 223])
        # crop_img = mmcv.imcrop(padding_img, bbox)
        # crop_lbl = mmcv.imcrop(padding_lbl, bbox)


        img, lbl = self.transform(padding_img, padding_lbl)

        return img, lbl, img_name


    def transform(self, img, lbl):
        img = img.transpose((2, 0, 1))
        img = img.astype(float) / 255.0
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def get_brainweb_colormap(self):
        """
        0:bg
        1:blood pool
        2:myocardium

        """

        return np.asarray([[0, 0, 0], [127, 127, 127], [255, 255, 255]])


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


def debug_load():
    root = '/data/home/ywen/lk/datasets/hvsmr2016/'
    # root = './datasets/hvsmr/'

    t_loader = hvsmrLoader_new(
        root,
        split='trainval')

    trainLoader = data.DataLoader(t_loader,
                                  batch_size=1,
                                  num_workers=4,
                                  shuffle=True)

    for (images, labels, img_name) in trainLoader:
        labels = np.squeeze(labels.data.numpy())
        decoded = t_loader.decode_segmap(labels, plot=False)
        m.imsave(pjoin('/data/home/ywen/lk/kkl_seg2021/kkl_segmentatiom/trained_models/hvsmr/', '{}.bmp'.format(img_name[0])), decoded)


if __name__ == '__main__':
    debug_load()

