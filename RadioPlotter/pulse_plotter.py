#! /usr/bin/env python
"""
Take antenna number as input and plot the pulses.
"""
import pandas as pd
import matplotlib.pyplot as plt


def pulse_plotter(antenna_number, pulses):
    """
    Assume the pulse array is setup as 
    3x256x240, for 3 polaritations, 256 time bins and 240 antennas
    """
    antenna_data = pulses[:, :, antenna_number]
    timec7 = antenna_data[:, -4]
    exc7 = antenna_data[:, -3]
    eyc7 = antenna_data[:, -2]
    ezc7 = antenna_data[:, -1]

    plt.title(f"pulses for antenna {antenna_number}")
    plt.plot(timec7, exc7, label = "x polarization")
    plt.plot(timec7, eyc7, label = "y polarization")
    plt.plot(timec7, ezc7, label = "z polarization")
    plt.xlabel("time in bins")
    plt.ylabel("efield")
    plt.legend()
    plt.savefig(f"pulses_antenna_{antenna_number}.png", dpi = 300)
    plt.close()

    return None