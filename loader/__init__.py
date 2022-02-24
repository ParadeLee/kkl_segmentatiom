from loader.brainwebLoader_new import brainwebLoader_new
from loader.brainwebLoader import brainwebLoader
from loader.hvsmrLoader import hvsmrLoader
from loader.mrbrainsLoader import mrbrainsLoader


def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "BrainWeb": brainwebLoader,
        "BrainWeb_new": brainwebLoader_new,
        "MRBrainS": mrbrainsLoader,
        # "Brats": BratsLoader,
		# 'Hyper': hyperLoader,
        "HVSMR": hvsmrLoader,
    }[name]