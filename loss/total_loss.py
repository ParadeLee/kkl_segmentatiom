import torch
import torch.nn.functional as F
import numpy as np


class aan_gradient_loss:
    """
    Sigma-weighted mean squared error for image reconstruction.
    """

    def __init__(self, penalty='l1'):
        super().__init__()
        self.penalty = penalty

    def _diffs_with_boundary(self, y):
        y = y.permute(0, 2, 3, 1)

        b = y[..., 1][..., np.newaxis]  # source
        y = y[..., 0][..., np.newaxis]  # mask
        # (1, 2, 240, 1)
        # vol_shape = y.get_shape().as_list()[2:]  # (240, 240)
        # ndims = len(vol_shape)  # 2
        ndims = 2
        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = y.permute(r)  # 置换
            b = b.permute(r)
            dfi = torch.mul((y[1:, ...] - y[:-1, ...]),
                            1 / ((10 * (b[1:, ...] - b[:-1, ...])) * (10 * (b[1:, ...] - b[:-1, ...])) + 1))
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)



        return df

    def loss_with_boundary(self, _, y_pred):
        if self.penalty == 'l1':
            df = [torch.mean(torch.abs(f)) for f in self._diffs_with_boundary(y_pred)]

        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            df = [torch.mean(f * f) for f in self._diffs_with_boundary(y_pred)]

        return torch.add(df[0], df[1]) / len(df)


def total_loss(input, grad, target, weight=None, reduction='mean'):
    a = input

    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    a = aan_gradient_loss(penalty="l2")

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss1 = F.cross_entropy(
        input, target, weight=weight, reduction=reduction, ignore_index=255
    )
    loss2 = a.loss_with_boundary(a, grad)

    return loss1 + loss2