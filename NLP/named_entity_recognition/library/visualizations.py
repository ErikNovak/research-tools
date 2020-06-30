"""Visualization package
This module contains methods for visualizing different
types of data.
"""
import matplotlib.pyplot as plt
import numpy as np
import math

# used for documentation
from typing import List, Tuple

def heatmap(data: List[List[float]], figsize: Tuple[int, int] = (10, 10),
            x_tick_labels: List[str] = None, y_tick_labels: List[str] = None,
            data_labels: bool = False, save_to_file: str = None,
            vmin: float = None, vmax: float = None) -> None:
    """Generates a heatmap from the given data

    It visualizes the data with the heatmap visualization,
    providing visual insight in the provided data.

    Args:
        data: A 2-dimensional numpy array containing values to be visualized.
        figsize: A tuple of integers representing the size of the
            generated figure (Default: (10, 10)).
        x_tick_labels: The tick labels that should be added to
            the x axis (Default: None).
        y_tick_labels: The tick labels that should be added to
            the y axis (Default: None).
        data_labels: The boolean value telling to show the actual data value
            in the heatmap or not (Default: false).
        save_to_file: the string of the file path to which we wish save the
            image (Default: None).
    """
    # initializes the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')


    ax.set_ylim(len(data)-0.5, -0.5)

    vmin = vmin if vmin is not None else np.amin(data)
    vmax = vmax if vmax is not None else np.amax(data)
    # add the heatmap visualization
    im = ax.imshow(data, vmin=vmin, vmax=vmax)
    # add a colorbar showing the scale (legend)
    cbar = ax.figure.colorbar(im, ax=ax)

    if x_tick_labels:
        # setting up the x tick labels
        ax.set_xticks(np.arange(data.shape[0]))
        ax.set_xticklabels(x_tick_labels)
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                rotation_mode="anchor")

    if y_tick_labels:
        # setting up the y tick labels
        ax.set_yticks(np.arange(data.shape[1]))
        ax.set_yticklabels(y_tick_labels)

    # To each position add the actual data value
    if data_labels:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(i, j, "%.2f" % data[i][j],
                    ha="center", va="center", color="w")

    fig.tight_layout()

    if save_to_file:
        plt.savefig(save_to_file, bbox_inches='tight')

    plt.show()


def loss_plots(data: List[List[float]], figsize: Tuple[int, int] = (10, 10),
            titles: List[str] = None, save_to_file: str = None) -> None:
    """Generates a set of loss plots from the data

    Visualizes a series of plots from the given data. The main
    intention of the function is to compare different loss
    plots side-by-side.

    Args:
        data: The list of plot data to be visualized.
        figsize: A tuple of integers representing the size of the
            generated figure (Default: (10, 10)).
        titles: The list of titles for each loss plot (Default: None).
        save_to_file: the string of the file path to which we wish save the
            image (Default: None).
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    # for each plot make a separate visualization
    for idx, plot in enumerate(data):
        color = ['mediumslateblue', 'darkorange', 'olivedrab'][idx]
        ax.plot(plot, color=color)

    fig.tight_layout()

    if save_to_file:
        plt.savefig(save_to_file, bbox_inches='tight')

    plt.show()