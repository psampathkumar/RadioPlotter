import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as intp
from matplotlib.widgets import Button


def view(
    pos,
    real,
    meta,
    scalar_fns,
    hack=True,
    interp=True,
):
    if real.shape[-1] == 1:
        mosaic_string = """
        ABC
        """
        width_ratio = [48, 1, 51]
    elif real.shape[-1] == 2:
        mosaic_string = """
        AABC
        AABD
        """
        width_ratio = [24, 24, 1, 51]
    elif real.shape[-1] == 3:
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

    x_pos = pos[:, 0]
    y_pos = pos[:, 1]

    class UpdatePlots:
        def __init__(self, key=list(scalar_fns.keys())[0]):
            self.key = key
            self.antenna_plt = None

        def update_plots(self, event=None):
            ax["A"].clear()
            scalar = scalar_fns[self.key](real, pos, meta)
            print(scalar.shape)
            if hack:
                for index in range(len(scalar)):
                    if index % 8 == 0:
                        scalar[index] = (
                            scalar[index + 4] + scalar[index + 5]
                        ) / 2
                    if index % 8 == 2:
                        scalar[index] = (
                            scalar[index - 1] + scalar[index + 1]
                        ) / 2
            xs = np.linspace(np.nanmin(x_pos), np.nanmax(x_pos), 100)
            ys = np.linspace(np.nanmin(y_pos), np.nanmax(y_pos), 100)
            xx, yy = np.meshgrid(xs, ys)
            # points within a circle
            in_star = xx**2 + yy**2 <= np.nanmax(x_pos**2 + y_pos**2)
            print(x_pos.shape, y_pos.shape, scalar.shape)
            interp_func = intp.Rbf(
                x_pos,
                y_pos,
                scalar,
                smooth=0,
                function="quintic",
            )
            fp_interp = np.where(in_star, interp_func(xx, yy), np.nan)
            pcm = ax["A"].pcolormesh(
                xx,
                yy,
                fp_interp,
                vmin=np.percentile(scalar, 0),
                vmax=np.percentile(scalar, 100),
                cmap="inferno",
                shading="gouraud",
            )  # use shading="gouraud" to make it smoother
            cbi = fig.colorbar(pcm, pad=0.2, cax=ax["B"], aspect=10)
            cbi.set_label(self.key, fontsize=20)
            ax["A"].set_ylabel("y / m")
            ax["A"].set_xlabel("x / m")
            ax["A"].set_facecolor("black")
            ax["A"].set_aspect(1)
            ax["A"].set_xlim(np.min(x_pos), np.max(x_pos))
            ax["A"].set_ylim(np.min(y_pos), np.max(y_pos))
            print("vmin = ", np.amin(scalar))
            print("vmax = ", np.amax(scalar))

            self.antenna_plt = ax["A"].scatter(
                x_pos,
                y_pos,
                edgecolor="w",
                facecolor="none",
                s=5.0,
                lw=1.0,
                picker=True,
            )

        def onpick(self, event):
            if event.artist != self.antenna_plt:
                return
            n = len(event.ind)
            if not n:
                return
            else:
                dataind = event.ind[0]

            ax["A"].scatter(
                x_pos[dataind],
                y_pos[dataind],
                edgecolor="r",
                facecolor="none",
                s=50.0,
                lw=5.0,
            )
            ax["A"].annotate(
                str(dataind),
                (x_pos[dataind] + 10, y_pos[dataind] + 10),
                color="green",
                size="large",
            )
            fig.canvas.flush_events()
            ax["C"].plot(
                real[dataind, :, 0],
                label=f"Index:{dataind}, xpos:{pos[dataind, 0]:.2f} ypos: {pos[dataind, 1]:.2f}",
            )
            ax["C"].legend()
            fig.canvas.flush_events()
            ax["D"].plot(real[dataind, :, 1])
            fig.canvas.flush_events()
            try:
                ax["E"].plot(real[dataind, :, 2])
            except IndexError:
                pass
            except KeyError:
                pass
            fig.canvas.draw_idle()
            return True

    def box_plots(event):
        ax["C"].clear()
        ax["D"].clear()
        ax["C"].boxplot(real[:, :, 0], showfliers=False, meanline=True)
        ax["C"].set_xticks([])
        ax["D"].boxplot(real[:, :, 1], showfliers=False, meanline=True)
        ax["D"].set_xticks([])
        try:
            ax["E"].clear()
            ax["E"].boxplot(real[:, :, 2], showfliers=False, meanline=True)
            ax["E"].set_xticks([])
        except KeyError:
            pass
        except IndexError:
            pass
        fig.canvas.draw_idle()

    def clear_plots(event):
        ax["A"].clear()
        ax["B"].clear()
        ax["C"].clear()
        ax["D"].clear()
        try:
            ax["E"].clear()
        except KeyError:
            pass
        fig.canvas.draw_idle()

    # Plot the first key by default and setup pulse picker
    defaultevent = UpdatePlots()
    defaultevent.update_plots()
    fig.canvas.mpl_connect("pick_event", defaultevent.onpick)

    # Add a button to clear the pulse plots
    axclear = fig.add_axes([0.45, 0.01, 0.05, 0.025])
    bbox = Button(axclear, "Box Plots")
    bbox.on_clicked(box_plots)
    # Add a button to clear the pulse plots

    axclear = fig.add_axes([0.5, 0.01, 0.05, 0.025])
    bnext = Button(axclear, "Clear")
    bnext.on_clicked(clear_plots)

    # Buttons for the scalar plots
    axscalar = []
    buttonscalar = []
    for i, key in enumerate(scalar_fns):
        axscalar.append(fig.add_axes([0.06 * (i + 1), 0.97, 0.05, 0.025]))
        buttonscalar.append(Button(axscalar[i], key))
        upplt = UpdatePlots(key)
        buttonscalar[i].on_clicked(upplt.update_plots)
        fig.canvas.mpl_connect("pick_event", upplt.onpick)

    plt.tight_layout()
    plt.show()
    return
