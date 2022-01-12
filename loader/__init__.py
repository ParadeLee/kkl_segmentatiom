from loader.brainwebLoader_new import brainwebLoader_new
from loader.brainwebLoader import brainwebLoader
from loader.mrbrainsLoader import mrbrainsLoader
from loader.hvsmrLoader import hvsmrLoader
from loader.mrbrainsLoader_new import mrbrainsLoader_new_ann
from loader.brainwebLoader_new_ann import brainwebLoader_ann
from loader.hvsmrLoader_new import hvsmrLoader_ann

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "BrainWeb": brainwebLoader,
        "BrainWeb_new": brainwebLoader_new,
        "BrainWeb_ann": brainwebLoader_ann,
        "MRBrainS": mrbrainsLoader,
        "MRBrainS_new": mrbrainsLoader_new_ann,
        # "Brats": BratsLoader,
		# 'Hyper': hyperLoader,
        "HVSMR": hvsmrLoader,
        "HVSMR_ann": hvsmrLoader_ann,
    }[name]