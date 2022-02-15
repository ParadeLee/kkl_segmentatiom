from loader.brainwebLoader_new import brainwebLoader_new
from loader.brainwebLoader import brainwebLoader
from loader.mrbrainsLoader_new import mrbrainsLoader_new
from loader.hvsmrLoader import hvsmrLoader

from loader.hvsmrLoader_new import hvsmrLoader_new

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "BrainWeb": brainwebLoader,
        "BrainWeb_new": brainwebLoader_new,
        "MRBrainS": mrbrainsLoader_new,
        # "Brats": BratsLoader,
		# 'Hyper': hyperLoader,
        "HVSMR": hvsmrLoader,
        "HVSMR_new": hvsmrLoader_new,
    }[name]