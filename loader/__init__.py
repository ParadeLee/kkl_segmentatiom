from loader.brainwebLoader import brainwebLoader
from loader.hvsmrLoader import hvsmrLoader
from loader.mrbrainsLoader import mrbrainsLoader
from loader.hyperLoader import hyperLoader

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "BrainWeb": brainwebLoader,
        "MRBrainS": mrbrainsLoader,
		"Hyper": hyperLoader,
        "HVSMR": hvsmrLoader,
    }[name]
