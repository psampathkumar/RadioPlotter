#! /usr/bin/env python
"""
Plot radio pulses and fluence maps.
"""
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy.interpolate as intp
from radiotools.analyses import energy_fluence

fnt_size = 20
plt.rc("font", size=fnt_size)  # controls default text size
plt.rc("axes", titlesize=fnt_size)  # fontsize of the title
plt.rc("axes", labelsize=fnt_size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=fnt_size)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=fnt_size)  # fontsize of the y tick labels
plt.rc("legend", fontsize=fnt_size)


def plot_scatter_interactive(real, sim):
    plots = []
    for i in range(2):
        fig = go.Figure(
            data=go.Scatter(
                x=np.arange(real.shape[0]),
                y=real[:, 0, i],
                mode="lines",
                name=f"real: - {i}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(sim.shape[0]),
                y=sim[:, i],
                mode="lines",
                name=f"sim: - {i}",
            )
        )
        plots.append(fig)
    return plots


def plot_hist(data):
    import plotly.express as px

    fig = px.histogram(data)
    return fig


def plot_pulses_interactive(real, sim, antenna=7):
    plots = []
    for i in range(2):
        fig = go.Figure(
            data=go.Scatter(
                x=np.arange(256),
                y=real[antenna, :, i],
                mode="lines",
                name=f"real: {antenna} - {i}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(256),
                y=sim[antenna, :, i],
                mode="lines",
                name=f"sim: {antenna} - {i}",
            )
        )
        plots.append(fig)
    return plots


def plot_pulses(pulses):
    """
    Plot pulses from efield.
    """

    timec7 = pulses[:, -4]
    exc7 = pulses[:, -3]
    eyc7 = pulses[:, -2]
    ezc7 = pulses[:, -1]

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(40, 30))
    ax[0].plot(timec7, exc7)
    ax[0].set_title("x - C7 CoREAS")

    ax[1].plot(timec7, eyc7)
    ax[1].set_title("y - C7 CoREAS")

    ax[2].plot(timec7, ezc7)
    ax[2].set_title("z - C7 CoREAS")

    fig.suptitle("Pulses")
    fig.supylabel("Electric Field [$\\mu$V/m]")
    fig.supxlabel("Time [s]")
    [i.grid(which="both", linestyle="dashed") for i in ax]
    plt.tight_layout()
    plt.savefig("pulse.pdf", format="pdf")
    plt.show()


def plot_interpolated_footprint(
    positions,
    energy_fluences,
    interpolator=intp.RBFInterpolator,
    mark_antennas=None,
    text=None,
    radius=np.inf,
):
    """

    Plot the interpolated footprint.

    Parameters
    ----------
    positions: np.array
        Antenna positions
    energy_fluences: np.array
        Energy fluence for each antenna
    interpolator: scipy interpolator class
        interpolator for the intermediate points
    mark_antenna: list
        index of antenna to highlight

    Returns
    -------
    None
    """
    x_pos = positions[:, 0]
    y_pos = positions[:, 1]
    if energy_fluences.ndim == 1:
        energy_fluences = energy_fluences.reshape((*energy_fluences.shape, 1))
    print(energy_fluences.shape)
    nplots = energy_fluences.shape[-1]
    fig, ax = plt.subplots(
        1,
        nplots * 2,
        gridspec_kw={"width_ratios": [40, 1] * nplots},
        figsize=[int(2.66 * fnt_size), fnt_size],
    )
    for ii in range(energy_fluences.shape[-1]):
        energy_flu = energy_fluences[:, ii]
        print(energy_flu.shape)
        if np.min(energy_flu) == np.max(energy_flu):
            print(np.min(energy_flu))
            continue

        # define positions where to interpolate
        xs = np.linspace(np.nanmin(x_pos), np.nanmax(x_pos), 100)
        ys = np.linspace(np.nanmin(y_pos), np.nanmax(y_pos), 100)
        xx, yy = np.meshgrid(xs, ys)
        # points within a circle
        in_star = xx**2 + yy**2 <= np.nanmax(x_pos**2 + y_pos**2)
        # interpolated values! but only in the star. outsite set to nan
        if interpolator is not None:
            interp_func = interpolator(
                list(zip(x_pos, y_pos)), energy_flu, kernel="linear"
            )
            fp_interp = np.where(
                in_star.flatten(),
                interp_func(np.array([xx, yy]).reshape(2, -1).T),
                np.nan,
            ).reshape(100, 100)
        else:
            fp_interp = energy_flu.reshape((100, 100))
            fp_interp = np.where(in_star, fp_interp, np.nan)

        cmap = "inferno"  # set the colormap
        # with vmin/vmax control that both
        # pcolormesh and scatter use the same colorscale
        pcm = ax[2 * ii + 0].pcolormesh(
            xx,
            yy,
            fp_interp,
            vmin=np.percentile(energy_flu, 1),
            vmax=np.percentile(energy_flu, 99),
            cmap=cmap,
            shading="gouraud",
        )  # use shading="gouraud" to make it smoother
        _ = ax[2 * ii + 0].scatter(
            x_pos,
            y_pos,
            edgecolor="w",
            facecolor="none",
            s=fnt_size / 2,
            lw=fnt_size / 10,
        )
        cbi = fig.colorbar(pcm, pad=0.02, cax=ax[2 * ii + 1])
        cbi.set_label(r"$Time (ns)$", fontsize=2 * fnt_size)

        ax[2 * ii + 0].set_ylabel("vvxxB (m)", fontsize=2 * fnt_size)
        ax[2 * ii + 0].set_xlabel("vxB (m)", fontsize=2 * fnt_size)
        ax[2 * ii + 0].set_facecolor("white")
        ax[2 * ii + 0].set_aspect(1)
        ax[2 * ii + 0].set_xlim(max(-radius, np.min(xs)), min(radius, np.max(xs)))
        ax[2 * ii + 0].set_ylim(max(-radius, np.min(ys)), min(radius, np.max(ys)))
        if text is not None:
            ax[2 * ii + 0].set_title(f"{text[ii]}", fontsize=2 * fnt_size)
        else:
            ax[2 * ii + 0].set_title(f"{ii}", fontsize=2 * fnt_size)
        # print("vmin = ", np.amin(energy_flu))
        # print("vmax = ", np.amax(energy_flu))
        ax[2 * ii + 0].tick_params(labelsize=2 * fnt_size)
        ax[2 * ii + 1].tick_params(labelsize=2 * fnt_size)
    if mark_antennas is not None:
        _ = ax[2 * ii + 0].scatter(
            x_pos[mark_antennas],
            y_pos[mark_antennas],
            edgecolor="r",
            facecolor="none",
            s=500.0,
            lw=5.0,
        )
        for i in mark_antennas:
            _ = ax[2 * ii + 0].annotate(
                str(i),
                (x_pos[i] + 10, y_pos[i] + 10),
                color="green",
                size="large",
            )
    plt.tight_layout()
    plt.show()


def plot_fluence_maps(
    pulses,
    pos_array,
    interp=True,
):
    """
    Plot fluence maps from given hdf5 file.

    Parameters
    ----------
    pulses:
        pulses seen in all the antennas
    pos_array:
        positions of all the antennas
    interp:
        interpolate the intermediate points when plotting fluences

    Returns
    -------
    None
    """
    energy_fluences = []
    positions = []
    for index in range(len(pulses)):
        pos = pos_array[index]
        trace_vB = pulses[index]  # 0,1,2,3: t, vxB, vxvxB, v
        positions.append(pos)
        ef = energy_fluence.calculate_energy_fluence_vector(
            trace_vB[:, 1:], trace_vB[:, 0], remove_noise=True
        )
        energy_fluences.append(ef)

    positions = np.array(positions)
    energy_fluences = np.array(energy_fluences)

    print(len(positions), len(energy_fluences))
    assert len(positions) == len(energy_fluences)
    print(len(positions))
    plot_interpolated_footprint(positions, energy_fluences, interp)
