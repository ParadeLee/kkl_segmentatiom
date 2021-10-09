from loader.brainwebLoader import brainwebLoader
from loader.mrbrainsLoader import mrbrainsLoader


def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "BrainWeb": brainwebLoader,
        "MRBrainS": mrbrainsLoader,
        # "Brats": BratsLoader,
		# 'Hyper': hyperLoader,
        # "HVSMR": hvsmrLoader
    }[name]