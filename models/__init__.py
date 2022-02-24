import copy
# from .axialnet import *
from .CRDN_att import UNetRNN
from .CRDN_old import UNetRNN as UR
from .MISSFormer import MISSFormer
from .MTUNet import MTUNet
from .newtrans import newTrans
from .MTUNet_Ctrans import MTC
from .MTU_CRDN import MTRNN


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    if name == "UNetRNN":
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3, feature_scale=4, decoder="LSTM", bias=True)
    elif name == "UR":
        model = model()
    elif name == "MISS":
        model = model()
    elif name == "MTUnet":
        model = model()
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
            "MISS": MISSFormer,
            "MTUnet": MTUNet,
            "new": newTrans,
            "MTC": MTC,
            "MTRNN": MTRNN,
        }[name]
    except:
        raise ("Model {} not available".format(name))
