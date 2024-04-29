"""
Plot Viewer which sees two scalar functions against each other.
"""
import numpy as np
import matplotlib.pyplot as plt
from RadioPlotter.viewevent import get_attributes, get_default_key


def view_plots(data, scalar_fns):
    pulses, pos, meta = get_attributes(data, get_default_key(data))
    xfunc = scalar_fns["xfunc"]
    yfunc = scalar_fns["yfunc"]

    def onpick(event):
        if len(event.ind) == 0:
            return
        dataind = event.ind[0]
        ax["B"].clear()
        ax["C"].clear()
        ax["B"].plot(pulses[dataind, :, 0])
        ax["B"].set_xticks([])
        ax["B"].set_title(dataind)
        ax["C"].plot(pulses[dataind, :, 1])
        ax["C"].set_xticks([])
        fig.canvas.draw_idle()

    fig, ax = plt.subplot_mosaic("AB\nAC")
    mask = pos[:, 1] < 50
    print(xfunc(pulses, pos, meta)[mask].shape)
    ax["A"].scatter(
        xfunc(pulses, pos, meta)[mask],
        yfunc(pulses, pos, meta)[mask],
        marker=".",
        picker=5,
    )
    ax["A"].scatter(
        xfunc(pulses, pos, meta)[mask],
        yfunc(pulses, pos, meta)[mask],
        marker=".",
        picker=5,
    )
    ax["A"].set_ylabel(scalar_fns["yfunc"].__name__)
    ax["A"].set_xlabel(scalar_fns["xfunc"].__name__)
    fig.canvas.callbacks.connect("pick_event", onpick)
    plt.show()
