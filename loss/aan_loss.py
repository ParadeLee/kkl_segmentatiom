import numpy as np
import tensorflow as tf
import keras.backend as K


class aan_gradient_loss:
    """
    Sigma-weighted mean squared error for image reconstruction.
    """

    def __init__(self, penalty='l1'):
        self.penalty = penalty

    def _diffs_with_boundary(self, y):
        b = y[..., 1][..., np.newaxis]  #source
        y = y[..., 0][..., np.newaxis]  #mask
        vol_shape = y.get_shape().as_list()[1:-1]  #(128,128,128)
        ndims = len(vol_shape)  #3

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]   #
            y = K.permute_dimensions(y, r)   #置换
            b = K.permute_dimensions(b, r)
            dfi = tf.multiply((y[1:, ...] - y[:-1, ...]),1 / ((10 * (b[1:, ...] - b[:-1, ...])) * (10 * (b[1:, ...] - b[:-1, ...])) + 1))

            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss_with_boundary(self, _, y_pred):
        if self.penalty == 'l1':
            df = [tf.reduce_mean(tf.abs(f)) for f in self._diffs_with_boundary(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            df = [tf.reduce_mean(f * f) for f in self._diffs_with_boundary(y_pred)]
        return tf.add_n(df) / len(df)