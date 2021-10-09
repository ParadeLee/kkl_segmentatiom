import copy
from .axialnet import *
from .CRDN_old import UNetRNN


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    if name == "UNetRNN":
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3, feature_scale=4, decoder="vanilla", bias=True,
                      is_deconv=True, is_batchnorm=True, selfeat=True, shift_n=5, auxseg=True)
    elif name == "MedT":
        model = model(AxialBlock_dynamic, AxialBlock_wopos, [1, 2, 4, 1], s=0.125, num_classes=4, zero_init_residual=True,
                         groups=8, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, img_size=256, imgchan=3)
    else:
        pass
    return model



def _get_model_instance(name):
    try:
        return {
            "UNetDRNN":UNetRNN,
            "MedT": medt_net,
        }[name]
    except:
        raise ("Model {} not available".format(name))
