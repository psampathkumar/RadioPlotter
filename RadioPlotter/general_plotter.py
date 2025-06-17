import matplotlib.pyplot as plt
import numpy as np
import scipy
def box_hist(X, Y, bins=8, xlabel="X", ylabel="Y"):
    import matplotlib.ticker as ticker

    # Use numpy.histogram for binning
    counts, bin_edges = np.histogram(X, bins=bins)

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    width = (np.max(bin_edges) - np.min(bin_edges)) / (bins * 3)
    # Prepare a dictionary to store data for violin plots
    box_data = []
    for i, center in enumerate(bin_centers):
        # Filter data points within the current bin range
        bin_data = [y for x, y in zip(X, Y) if bin_edges[i] <= x < bin_edges[i + 1]]
        box_data.append(bin_data)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.boxplot(
        box_data, showfliers=False, positions=bin_centers, widths=width
    )  # Format bin center
    h = ax.hist2d(
        X,
        Y,
        bins=50,
        range=[
            [np.min(x_meta), np.max(x_meta)],
            [np.percentile(raderr, 0.1), np.percentile(raderr, 99.1)],
        ],
        density=True,
        cmap="Reds",
    )
    plt.colorbar(h[3], ax=ax, cmap="Reds")
    ax.xaxis.set_major_formatter("{x:0.1f}")
    # Overall plot customizations
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def bias_resolution(X, Y,bins = 8, xlabel='X', ylabel='Y'):
    # Use numpy.histogram for binning
    counts, bin_edges = np.histogram(X, bins=bins)

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Prepare a dictionary to store data for violin plots
    box_data = []
    for i, center in enumerate(bin_centers):
        # Filter data points within the current bin range
        bin_data = [y for x, y in zip(X, Y) if bin_edges[i] <= x < bin_edges[i+1]]
        box_data.append(bin_data)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10,7))
    mean_array = np.zeros((*bin_centers.shape,4))
    std_array = np.zeros((*bin_centers.shape,4))
    for i, dd in enumerate(box_data):
        dd = np.array(dd)
        a = (dd,)#[dd < np.percentile(dd,99)],)
        if len(a[0]) < 2:
            continue

        mean_boot = scipy.stats.bootstrap(a, np.mean, method='percentile')
        std_boot = scipy.stats.bootstrap(a, np.std, method='percentile')
        mean_array[i,0] = np.mean(a)
        mean_array[i,1] = mean_boot.confidence_interval.low
        mean_array[i,2] = mean_boot.confidence_interval.high
        mean_array[i,3] = mean_boot.standard_error
        std_array[i, 0] = np.std(a)
        std_array[i, 1] = std_boot.confidence_interval.low
        std_array[i, 2] = std_boot.confidence_interval.high
        std_array[i, 3] = std_boot.standard_error
    print(mean_array[:,0])
    print(mean_array[:,1:].T.shape)
    print(bin_centers.shape)
    ax.errorbar(bin_centers, mean_array[:,0], yerr=mean_array[:,3].T, label='bias', fmt='o', barsabove=True)
    ax.errorbar(bin_centers, std_array[:,0], yerr=std_array[:,3].T,label='resolution', fmt='o', barsabove=True)
    ax.xaxis.set_major_formatter('{x:2.0f}')
    # Overall plot customizations
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.legend()