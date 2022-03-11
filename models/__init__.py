import copy
# from .CRDN_att import UNetRNN
from .demo2 import ARNN
from .mlpRNN import MlpRnn
from .kkl_trans import kkTrans


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    # if name == "UNetRNN":
    #     model = model(input_channel=3, n_classes=n_classes, kernel_size=3, feature_scale=4, decoder="LSTM", bias=True)
    if name == "mlpRnn":
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3,
                      feature_scale=4, decoder="vanilla", bias=True, is_deconv=True, is_batchnorm=True, selfeat=True, shift_n=5, auxseg=True)
    elif name == "kkl":
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3,
                      feature_scale=1, decoder="vanilla", bias=True, is_deconv=True, is_batchnorm=True, selfeat=True, shift_n=5, auxseg=True)
    elif name == "kktrans":
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3,
                      feature_scale=4, decoder="vanilla", bias=True, is_deconv=True, is_batchnorm=True, selfeat=True, shift_n=5, auxseg=True)
    else:
        pass
    return model



def _get_model_instance(name):
    try:
        return {
            # "UNetRNN": UNetRNN,
            "kkl": ARNN,
            "mlpRnn": MlpRnn,
            "kktrans": kkTrans,
        }[name]
    except:
        raise ("Model {} not available".format(name))
