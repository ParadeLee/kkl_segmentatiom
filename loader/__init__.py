from loader.brainwebLoader_new import brainwebLoader_new
from loader.mrbrainsLoader import mrbrainsLoader


def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "BrainWeb": brainwebLoader_new,
        "MRBrainS": mrbrainsLoader,
        # "Brats": BratsLoader,
		# 'Hyper': hyperLoader,
        # "HVSMR": hvsmrLoader
    }[name]