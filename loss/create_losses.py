import torch
import torch.nn as nn

from . import df_loss
from . import mag_angle_loss
#from . import surface_loss

class Total_loss():
    def __init__(self, boundary=False):
        self.df_loss = mag_angle_loss.EuclideanAngleLossWithOHEM()
        '''self.boundary = boundary
        if boundary:
            self.boundary_loss = SurfaceLoss(idc=[1,2,3])'''
        
    def __call__(self, df_out, gts_df, gts):
        
        df_loss = self.df_loss(df_out, gts_df, gts[:, None, ...])
        return df_loss

