import copy
# from .axialnet import *
from .CRDN_old import UNetRNN
from .SwinUNet import SwinTransformerSys
from .MISSFormer import MISSFormer


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    if name == "UNetRNN":
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3, feature_scale=4, decoder="LSTM", bias=True,
                      is_deconv=True, is_batchnorm=True, selfeat=True, shift_n=5, auxseg=True)
    elif name == "SwinUnet":
        model = model()
    elif name == "MISS":
        model = model()
    else:
        pass
    return model



def _get_model_instance(name):
    try:
        return {
            "UNetDRNN": UNetRNN,
            "SwinUnet": SwinTransformerSys,
            "MISS": MISSFormer
        }[name]
    except:
        raise ("Model {} not available".format(name))
