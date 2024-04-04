#! /usr/bin/env python
"""
Take antenna number as input and plot the pulses.
"""
import pandas as pd
import matplotlib.pyplot as plt


def pulse_plotter(antenna_number, file):
    """
    Assume file is setup as 
    x y z time antenna_number
    and each row corresponds to one antenna number
    """
    names = ["pulse_x", "pulse_y", "pulse_z", "time", "antennaID"]
    pulses = pd.read_csv(file, names=names)
    pulses = pulses[pulses.antennaID == antenna_number]

    plt.title(f"pulses for antenna {antenna_number}")
    plt.plot(pulses["pulse_x"], label = "x polarization")
    plt.plot(pulses["pulse_y"], label = "y polarization")
    plt.plot(pulses["pulse_z"], label = "z polarization")
    plt.savefig(f"pulses_antenna_{antenna_number}.png", dpi = 300)
    plt.close()

    return None