import numpy as np
from matplotlib import pyplot as plt


def quadratic(a, b, c, sign=-1):
    return (-b + sign * np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def vector_loop(l1, l2, l3, l4, theta2, crossed=False):
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    k1 = l1 / l2
    k2 = l1 / l4
    k3 = (l2 ** 2 - l3 ** 2 + l4 ** 2 + l1 ** 2) / (2 * l2 * l4)
    k4 = l1 / l3
    k5 = (l4 ** 2 - l1 ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    a = -k1 - k2 * c2 + k3 + c2
    b = -2 * s2
    c = k1 - k2 * c2 + k3 - c2
    d = c2 - k1 + k4 * c2 + k5
    e = -2 * s2
    f = k1 + (k4 - 1) * c2 + k5
    sign = 1 if crossed else -1
    answers = np.array([quadratic(a, b, c, sign=sign), quadratic(d, e, f, sign=sign)])
    # Returns theta4, theta3
    return 2 * np.arctan(answers)


def velocity_analysis(l2, l3, l4, w2, theta2, theta3, theta4):
    answers = np.array(
        [np.sin(theta4 - theta2) / (l3 * np.sin(theta3 - theta4)), np.sin(theta2 - theta3) / (l4 * np.sin(theta4 - theta3))])
    # Returns w3, w4
    return l2 * w2 * answers


def acceleration_analysis(l2, l3, l4, w2, w3, w4, alpha2, theta2, theta3, theta4):
    a = l4 * np.sin(theta4)
    b = l3 * np.sin(theta3)
    c = l2 * alpha2 * np.sin(theta2) + l2 * w2 ** 2 * np.cos(theta2) + l3 * w3 ** 2 * np.cos(theta3) - l4 * w4 ** 2 * np.cos(theta4)
    d = l4 * np.cos(theta4)
    e = l3 * np.cos(theta3)
    f = l2 * alpha2 * np.cos(theta2) - l2 * w2 ** 2 * np.sin(theta2) - l3 * w3 ** 2 * np.sin(theta3) + l4 * w4 ** 2 * np.sin(theta4)
    # returns alpha3, alpha4
    return np.array([c * d - a * f, c * e - b * f]) / (a * e - b * d)


def linear_acceleration(l, w, alpha, theta):
    a_c = w ** 2 / l * -np.array([np.cos(theta), np.sin(theta)])
    a_t = l * alpha * np.array([-np.sin(theta), np.cos(theta)])
    return a_c + a_t


def mechanical_advantage(l1, l2, l3, l4, theta2, r_in, r_out, crossed=False):
    theta4, theta3 = vector_loop(l1, l2, l3, l4, theta2, crossed=crossed)
    _, w4 = velocity_analysis(l2, l3, 1, theta2, theta3, theta4)
    return np.abs(1 / w4 * r_in / r_out)


def slider_crank_vector_loop(l2, l3, l4, theta2, crossed=False):
    if crossed:
        theta3 = np.arcsin((l2 * np.sin(theta2) - l4) / l3)
    else:
        theta3 = np.arcsin(-(l2 * np.sin(theta2) - l4) / l3) + np.pi
    l1 = l2 * np.cos(theta2) - l3 * np.cos(theta3)
    return l1, theta3


def slider_crank_velocity_analysis(l2, l3, w2, theta2, theta3):
    w3 = l2 * w2 / l3 * np.cos(theta2) / np.cos(theta3)
    v_l1 = l3 * w3 * np.sin(theta3) - l2 * w2 * np.sin(theta2)
    return w3, v_l1


def slider_crank_acceleration_analysis(l2, l3, w2, w3, alpha2, theta2, theta3):
    alpha3 = (l2 * alpha2 * np.cos(theta2) - l2 * w2 ** 2 * np.sin(theta2) + l3 * w3 ** 2 * np.sin(theta3)) / (l3 * np.cos(theta3))
    a_l1 = -l2 * alpha2 * np.sin(theta2) - l2 * w2 ** 2 * np.cos(theta2) + l3 * alpha3 * np.sin(theta3) + l3 * w3 ** 2 * np.cos(theta3)
    return alpha3, a_l1


def slider_crank_mechanical_advantage(l2, l3, l4, theta2, r_in, crossed=False):
    w3, _ = slider_crank_velocity_analysis(l2, l3, l4, 1, theta2, crossed)
    return np.abs(1 / w3 * r_in / l3)


def percent_dev_from_constant(f, start, end):
    f_avg = (f(start) + f(end)) / 2
    x = np.arange(start, end, 1.0)
    y = np.abs(f(x) - f_avg) / f_avg
    return np.sum(y) / (end - start) * 100


def plot(f, xfun=None, xmin=0, xmax=360, xstep=1, title='', xlabel='', ylabel='', labels=None, colors=None, sharex=False):
    if labels is None:
        labels = []
    if colors is None:
        colors = []

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    x = np.linspace(xmin, xmax, int((xmax - xmin) / xstep))
    plots = f(x)
    if xfun is not None:
        x = xfun(x)

    try:
        _, n_cols = plots.shape
        if sharex:
            fig, ax = plt.subplots()
            ax.set_xlabel(xlabel)
        for i in range(0, n_cols):
            label = labels[i] if len(labels) > i else ''
            color = colors[i] if len(colors) > i else 'red'
            if sharex:
                if i != 0:
                    ax = ax.twinx()
                ax.set_ylabel(label, color=color)
                ax.plot(x, plots[:, i], color=color)
                ax.tick_params(axis="y", labelcolor=color)
            else:
                plt.plot(x, plots[:, i], color=color, label=label)
    except:
        plt.plot(x, plots, color='red')

    plt.title(title)
    if not sharex:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    else:
        fig.tight_layout()
