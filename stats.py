#from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy import stats
import numpy as np
from arch.bootstrap import IIDBootstrap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray


def _decorate_axis(ax, wrect=10, hrect=10, ticklabelsize="large"):
    """Helper function for decorating plots."""
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))
    return ax


def _annotate_and_decorate_axis(
    ax,
    labelsize="x-large",
    ticklabelsize="x-large",
    xticks=None,
    xticklabels=None,
    yticks=None,
    legend=False,
    grid_alpha=0.2,
    legendsize="x-large",
    xlabel="",
    ylabel="",
    wrect=15,
    hrect=10,
):
    """Annotates and decorates the plot."""
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    if xticks is not None:
        ax.set_xticks(ticks=xticks)
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, alpha=grid_alpha)
    ax = _decorate_axis(ax, wrect=wrect, hrect=hrect, ticklabelsize=ticklabelsize)
    if legend:
        ax.legend(fontsize=legendsize)
    return ax

def calculate_stats(files: list, algs: list, gamma: float, dirname: str, filename: str, title: str):
    intervals = []
    points = []
    metrics = ["IQM", "Mean", "Optimality Gap"]
    for idx, event_file_path in enumerate(files):
        # Extract scalar values
        scalar_values = [event.value for event in scalar_events]

        # print(stats.describe(scalar_values))
        bs = IIDBootstrap(np.array(scalar_values))
        custom_stats = lambda x: np.array(
            [
                stats.trim_mean(x, 0.25),
                stats.tmean(x),
                gamma - np.mean(np.minimum(x, gamma)),
            ]
        )
        ci = bs.conf_int(custom_stats, 1000)
        intervals.append(ci)
        points.append(
            np.array(
                [
                    stats.trim_mean(scalar_values, 0.25),
                    stats.tmean(scalar_values),
                    gamma - np.mean(np.minimum(scalar_values, gamma)),
                ]
            )
        )
    num_metrics = len(metrics)
    figsize = (3.3 * num_metrics, 0.7 * len(algs))
    fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize)
    color_palette = "colorblind"
    color_palette = sns.color_palette(color_palette, n_colors=len(algs))
    colors = dict(zip(algs, color_palette))
    h = 0.5
    for idx, metric_name in enumerate(metrics):
        for alg_idx, algorithm in enumerate(algs):
            ax = axes[idx]
            cur_int = intervals[alg_idx]
            lower, upper = cur_int[:, idx]
            ax.barh(
                y=alg_idx,
                width=upper - lower,
                height=h,
                left=lower,
                color=colors[algorithm],
                alpha=0.75,
                label=algorithm,
            )
            ax.vlines(
                x=points[alg_idx][idx],
                ymin=alg_idx - (7.5 * h / 16),
                ymax=alg_idx + (6 * h / 16),
                label=algorithm,
                color="k",
                alpha=0.5,
            )
        ax.set_yticks(list(range(len(algs))))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        if idx != 0:
            ax.set_yticks([])
        else:
            ax.set_yticklabels(algs, fontsize="x-large")
        ax.set_title(title, fontsize="xx-large")
        ax.tick_params(axis="both", which="major")
        _decorate_axis(ax, ticklabelsize="xx-large", wrect=5)
        ax.spines["left"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(dirname+f"{filename}_rliable_stats.png")
    return fig, axes


def plot_probability_of_improvement(files: list, algs: list, baseline: str, dirname: str, filename: str):
    fig, ax = plt.subplots()
    color_palette = "colorblind"
    colors = sns.color_palette(color_palette, n_colors=len(files))
    probability_estimates = []
    probability_interval_estimates = []
    for idx, f in enumerate(files):
        prob, ints, gamma = probability_of_improvement(f, baseline)
        probability_estimates.append(prob)
        probability_interval_estimates.append(ints)
        lower, upper = ints
        alg_name = algs[idx]
        h = 0.6
        ax.barh(
            y=idx,
            width=upper - lower,
            height=h,
            left=lower,
            color=colors[idx],
            alpha=0.75,
            label=alg_name,
        )
        ax.vlines(
            x=prob,
            ymin=idx - 7.5 * h / 16,
            ymax=idx + (6 * h / 16),
            color="k",
            alpha=min(0.75 + 0.1, 1.0),
        )
    ax = _decorate_axis(ax)
    ax.set_yticks(range(len(probability_estimates)))
    ax.set_yticklabels(algs, fontsize="large")
    ax.tick_params(axis="both", which="major")
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_label_coords(-0.2, 1.0)
    fig.savefig(dirname+f"{filename}_prob.png")
    return gamma


def probability_of_improvement(file_x: str, file_y: str):
    """Overall Probability of imporvement of algorithm `X` over `Y`.

    Args:
    scores_x: A matrix of size (`num_runs_x` x `num_tasks`) where scores_x[n][m]
        represent the score on run `n` of task `m` for algorithm `X`.
    scores_y: A matrix of size (`num_runs_y` x `num_tasks`) where scores_x[n][m]
        represent the score on run `n` of task `m` for algorithm `Y`.
    Returns:
        P(X_m > Y_m) averaged across tasks.
    """
    event_acc = EventAccumulator(file_x)
    event_acc.Reload()
    scalar_events = event_acc.Scalars("total_return")
    scores_x = np.array([event.value for event in scalar_events])
    scores_x = np.expand_dims(scores_x, axis=1)

    event_acc = EventAccumulator(file_y)
    event_acc.Reload()
    scalar_events = event_acc.Scalars("total_return")
    scores_y = np.array([event.value for event in scalar_events])
    gamma = stats.trim_mean(scores_y, 0.25)
    scores_y = np.expand_dims(scores_y, axis=1)
    num_tasks = scores_x.shape[1]
    task_improvement_probabilities = []

    def _prob_helper(scores_x, scores_y, task):
        if np.array_equal(scores_x[:, task], scores_y[:, task]):
            task_improvement_prob = 0.5
        else:
            task_improvement_prob, _ = stats.mannwhitneyu(
                scores_x[:, task], scores_y[:, task], alternative="greater"
            )
            task_improvement_prob /= scores_x.shape[0] * scores_y.shape[0]
        return task_improvement_prob

    bs = IIDBootstrap(scores_x)
    custom_stats = lambda x: np.array([_prob_helper(x, scores_y, 0)])
    ci = bs.conf_int(custom_stats, 1000)
    for task in range(num_tasks):
        task_improvement_prob = _prob_helper(scores_x, scores_y, task)
        task_improvement_probabilities.append(task_improvement_prob)
    return np.mean(task_improvement_probabilities), ci, gamma


def get_stats(files, algs, opt_file, dirname, filename, title):
    gamma = plot_probability_of_improvement(files=files, algs=algs, baseline=opt_file, dirname=dirname, filename=filename)

    files.append(opt_file)
    algs.append(f"$\pi^*$")
    print(calculate_stats(files=files, algs=algs, gamma=gamma, dirname=dirname, filename=filename, title=title))