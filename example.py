"""Example Script."""

import argparse

import numpy as np

from RadioPlotter.viewevent import view_footprint
from RadioPlotter.viewplots import view_plots


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


def get_lateral_distance(pulses, pos_array, outp_meta):
    return np.sqrt(pos_array[:, 0] ** 2 + pos_array[:, 1] ** 2)


def get_max_value2(pulses, pos_array, outp_meta):
    """
    Max value of third polarity.

    Visual proxy for Fluence
    """
    return np.max(np.abs(pulses), axis=1)[:, 2]


pos1 = np.load("./examples/data/positions.npy")
real1 = np.load("./examples/data/real.npy")
meta1 = np.load("./examples/data/meta.npy")
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p" "--plotter",
    action="store_true",
    help="Use interactive Plotter instead of event viewer",
)
opt = parser.parse_args()
if opt.p__plotter:
    view_plots(
        {
            "vbvvb": {
                "pos": pos1,
                "real": real1,
                "meta": meta1,
            },
        },
        {
            "vB max fluence vs Lateral Distance": {
                "xfunc": get_lateral_distance,
                "yfunc": get_max_value0,
            },
            "vvB max Fluence vs Lateral Distance": {
                "xfunc": get_lateral_distance,
                "yfunc": get_max_value1,
            },
        },
        # pulse_process=spectrum,
    )
else:
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
