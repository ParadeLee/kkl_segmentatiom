import copy
# from .axialnet import *
from .CRDN_att import UNetRNN
from .CRDN_old import UNetRNN as UR
from .MISSFormer import MISSFormer
from .MTUNet import MTUNet
from .MTUNet_Ctrans import MTC
from .MTU_CRDN import MTRNN
from .sxx import UNetDRNN
from .new import ARNN
from .new2 import ARNN2


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    if name == "UNetRNN":
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3, feature_scale=4, decoder="LSTM", bias=True)
    elif name == "UR":
        model = model()
    elif name == "sxx":
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3,
                      feature_scale=4, decoder="vanilla", bias=True, is_deconv=True, is_batchnorm=True, selfeat=True, shift_n=5, auxseg=True)
    elif name == "new":
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3,
                      feature_scale=4, decoder="vanilla", bias=True, is_deconv=True, is_batchnorm=True, selfeat=True, shift_n=5, auxseg=True)
    elif name == "new2":
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3,
                      feature_scale=4, decoder="vanilla", bias=True, is_deconv=True, is_batchnorm=True, selfeat=True, shift_n=5, auxseg=True)
    elif name == "new":
        model = model()
    elif name == "MTC":
        model = model()
    elif name == "MTRNN":
        model = model()
    else:
        pass
    return model



def _get_model_instance(name):
    try:
        return {
            "UNetRNN": UNetRNN,
            "UR": UR,
            "MTUnet": MTUNet,
            "new2": ARNN2,
            "MTC": MTC,
            "MTRNN": MTRNN,
            "sxx": UNetDRNN,
            "new": ARNN
        }[name]
    except:
        raise ("Model {} not available".format(name))
