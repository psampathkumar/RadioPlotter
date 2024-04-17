import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from scipy.interpolate import RBFInterpolator

from RadioPlotter.utilities.customelements import MyRadioButtons


def get_attributes(data, key):
    """Separate pulses, position and meta data from attribute dictionary."""
    pos = data[key]["pos"]
    pulses = data[key]["real"]
    meta = data[key]["meta"]
    return pulses, pos, meta


def get_default_key(dictt):
    """Get the first key of dict."""
    return list(dictt.keys())[0]


class UpdatePlots:
    def __init__(
        self,
        fig,
        ax,
        scalar_fns,
        data,
        scalar_key=None,
        data_key=None,
    ):
        if scalar_key is None:
            self._key = get_default_key(scalar_fns)
        else:
            self._key = scalar_key
        self.antenna_plt = None
        self.data = data
        self.scalar_fns = scalar_fns
        self.fig = fig
        self.ax = ax

        if data_key is None:
            self._dkey = get_default_key(data)
        else:
            self._dkey = data_key
        self.pulses, self.pos, self.meta = get_attributes(self.data, self._dkey)
        self.plotted_antenna = []

    @property
    def dkey(self):
        """key to the data dictionary."""
        return self._dkey

    @dkey.setter
    def dkey(self, key):
        assert key in self.data
        self._dkey = key
        self.pulses, self.pos, self.meta = get_attributes(self.data, self._dkey)

    @property
    def key(self):
        """key to the scalar function dictionary."""
        return self._key

    @key.setter
    def key(self, key):
        assert key in self.scalar_fns
        self._key = key

    def update_plots(self, event=None):
        self.ax["A"].clear()
        x_pos = self.pos[:, 0]
        y_pos = self.pos[:, 1]
        print(self._key)
        scalar = self.scalar_fns[self._key](self.pulses, self.pos, self.meta)
        if self.data[self.dkey]["hack"]:
            for index in range(len(scalar)):
                if index % 8 == 0:
                    scalar[index] = (scalar[index + 4] + scalar[index + 5]) / 2
                if index % 8 == 2:
                    scalar[index] = (scalar[index - 1] + scalar[index + 1]) / 2
        xs = np.linspace(np.nanmin(x_pos), np.nanmax(x_pos), 100)
        ys = np.linspace(np.nanmin(y_pos), np.nanmax(y_pos), 100)
        xx, yy = np.meshgrid(xs, ys)
        # points within a circle
        in_star = xx**2 + yy**2 <= np.nanmax(x_pos**2 + y_pos**2)
        print(x_pos.shape, y_pos.shape, scalar.shape)
        interp_func = RBFInterpolator(list(zip(x_pos, y_pos)), scalar, kernel="quintic")
        fp_interp = np.where(
            in_star.flatten(),
            interp_func(np.array([xx, yy]).reshape(2, -1).T),
            np.nan,
        ).reshape(100, 100)
        pcm = self.ax["A"].pcolormesh(
            xx,
            yy,
            fp_interp,
            vmin=np.percentile(scalar, 0),
            vmax=np.percentile(scalar, 100),
            cmap="inferno",
            shading="gouraud",
        )  # use shading="gouraud" to make it smoother
        cbi = self.fig.colorbar(pcm, pad=0.2, cax=self.ax["B"], aspect=10)
        cbi.set_label(self._key, fontsize=20)
        self.ax["A"].set_ylabel("y / m")
        self.ax["A"].set_xlabel("x / m")
        self.ax["A"].set_facecolor("black")
        self.ax["A"].set_aspect(1)
        self.ax["A"].set_xlim(np.min(x_pos), np.max(x_pos))
        self.ax["A"].set_ylim(np.min(y_pos), np.max(y_pos))
        print("vmin = ", np.amin(scalar))
        print("vmax = ", np.amax(scalar))

        self.antenna_plt = self.ax["A"].scatter(
            x_pos,
            y_pos,
            edgecolor="w",
            facecolor="none",
            s=5.0,
            lw=1.0,
            picker=True,
        )
        self.fig.canvas.draw_idle()

    def onpick(self, event):
        if event.artist != self.antenna_plt:
            return
        n = len(event.ind)
        if not n:
            return
        else:
            dataind = event.ind[0]
        self.plotted_antenna.append(dataind)
        self.plot_antennas([dataind])

    def plot_antennas(self, datainds):
        for dataind in datainds:
            x_pos = self.pos[:, 0]
            y_pos = self.pos[:, 1]
            self.ax["A"].scatter(
                x_pos[dataind],
                y_pos[dataind],
                edgecolor="r",
                facecolor="none",
                s=50.0,
                lw=5.0,
            )
            self.ax["A"].annotate(
                str(dataind),
                (x_pos[dataind] + 10, y_pos[dataind] + 10),
                color="green",
                size="large",
            )
            self.fig.canvas.flush_events()
            self.ax["C"].plot(
                self.pulses[dataind, :, 0],
                label=f"{self.dkey}: Index:{dataind}, xpos:{x_pos[dataind]:.2f} "
                f"ypos:"
                f" {y_pos[dataind]:.2f}",
            )
            self.ax["C"].legend()
            self.fig.canvas.flush_events()
            self.ax["D"].plot(self.pulses[dataind, :, 1])
            self.fig.canvas.flush_events()
            try:
                self.ax["E"].plot(self.pulses[dataind, :, 2])
            except IndexError:
                pass
            except KeyError:
                pass
            self.fig.canvas.draw_idle()
        return True

    def update_skeys(self, skey=None):
        if skey is not None:
            self.key = skey
            print(self.key)
            self.ax["A"].clear()
            self.update_plots()

    def update_dkeys(self, dkey=None):
        if dkey is not None:
            self.dkey = dkey
            self.ax["A"].clear()
            self.update_plots()
            self.plot_antennas(self.plotted_antenna)

    def box_plots(self, event):
        if self.data[self.dkey]["hack"]:
            indices = np.logical_not(np.arange(len(self.pulses)) % 8 == 0)
        else:
            indices = np.arange(len(self.pulses))
        self.ax["C"].boxplot(
            self.pulses[indices, :, 0], showfliers=False, meanline=True
        )
        self.ax["C"].set_xticks([])
        self.ax["D"].boxplot(
            self.pulses[indices, :, 1], showfliers=False, meanline=True
        )
        self.ax["D"].set_xticks([])
        try:
            self.ax["E"].boxplot(
                self.pulses[indices, :, 2], showfliers=False, meanline=True
            )
            self.ax["E"].set_xticks([])
        except KeyError:
            pass
        except IndexError:
            pass
        self.fig.canvas.draw_idle()

    def clear_plots(self, event):
        self.ax["A"].clear()
        self.ax["B"].clear()
        self.ax["C"].clear()
        self.ax["D"].clear()
        try:
            self.ax["E"].clear()
        except KeyError:
            pass
        self.plotted_antenna = []
        self.fig.canvas.draw_idle()


def view(
    data,
    scalar_fns,
):
    # get attributes from first key
    pulses, pos, meta = get_attributes(data, list(data.keys())[0])

    if pulses.shape[-1] == 1:
        mosaic_string = """
        ABC
        """
        width_ratio = [48, 1, 51]
    elif pulses.shape[-1] == 2:
        mosaic_string = """
        AABC
        AABD
        """
        width_ratio = [24, 24, 1, 51]
    elif pulses.shape[-1] == 3:
        mosaic_string = """
        AAABC
        AAABD
        AAABE
        """
        width_ratio = [16, 16, 16, 1, 51]
    else:
        raise RuntimeError("Only 3 polarizations are supported")
    fig, ax = plt.subplot_mosaic(
        mosaic_string,
        figsize=(20, 10),
        gridspec_kw={"width_ratios": width_ratio},
    )
    for aa in ax.keys():
        print(ax[aa].get_subplotspec().get_geometry())
    print(pos.shape)

    # Plot the first key by default and setup pulse picker
    upplt = UpdatePlots(fig, ax, scalar_fns, data)
    upplt.update_plots()
    fig.canvas.mpl_connect("pick_event", upplt.onpick)

    # Add a button to clear the pulse plots
    axclear = fig.add_axes([0.5, 0.01, 0.05, 0.025])
    bnext = Button(axclear, "Clear")
    bnext.on_clicked(upplt.clear_plots)

    # Buttons for the scalar plots
    print(meta.shape)
    print(scalar_fns.keys())
    axradio = fig.add_axes(
        [
            0.03,
            0.95,
            0.07 * np.where(len(scalar_fns.keys()) > 6, 6, len(scalar_fns.keys())),
            0.025 * (len(scalar_fns.keys()) // 6 + 1),
        ]
    )
    radio = MyRadioButtons(
        axradio,
        scalar_fns.keys(),
        orientation="horizontal",
        active=0,
    )
    radio.on_clicked(upplt.update_skeys)
    axradio2 = fig.add_axes([0.06, 0.07, 0.05 * len(data.keys()), 0.025])
    radio2 = MyRadioButtons(
        axradio2,
        data.keys(),
        orientation="horizontal",
        active=0,
    )
    radio2.on_clicked(upplt.update_dkeys)
    fig.canvas.mpl_connect("pick_event", upplt.onpick)

    axclear = fig.add_axes([0.45, 0.01, 0.05, 0.025])
    bbox = Button(axclear, "Box Plots")
    bbox.on_clicked(upplt.box_plots)

    plt.tight_layout()
    plt.show()
    return
