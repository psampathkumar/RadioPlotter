"""Example Script."""
import numpy as np

from RadioPlotter.viewevent import view_footprint


def get_max_value0(pulses, pos_array, outp_meta):
    """
    Max value of first polarity.

    Visual proxy for Fluence
    """
    return np.max(np.abs(pulses), axis=1)[:, 0]


def get_max_value1(pulses, pos_array, outp_meta):
    """
    Max value of second polarity.

    Visual proxy for Fluence
    """
    return np.max(np.abs(pulses), axis=1)[:, 1]


def get_max_value2(pulses, pos_array, outp_meta):
    """
    Max value of third polarity.

    Visual proxy for Fluence
    """
    return np.max(np.abs(pulses), axis=1)[:, 2]


pos1 = np.load("./examples/data/positions.npy")
real1 = np.load("./examples/data/real.npy")
meta1 = np.load("./examples/data/meta.npy")
view_footprint(
    {
        "vbvvb": {
            "pos": pos1,
            "real": real1,
            "meta": meta1,
            "hack": False,
        },
    },
    {
        "Max Value 0": get_max_value0,
        "Max Value 1": get_max_value1,
        "Max Value 2": get_max_value2,
    },
)
