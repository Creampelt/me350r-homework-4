import numpy as np
from matplotlib import pyplot as plt
import utils


l2 = 105
l3 = 172
l4 = 27

alpha2_w2_pairs = [(0, 1), (1, 0), (1, 1)]


def f(theta):
    results = []
    theta_rad = np.radians(theta)
    for alpha2, w2 in alpha2_w2_pairs:
        _, theta3 = utils.slider_crank_vector_loop(l2, l3, l4, theta_rad)
        w3, _ = utils.slider_crank_velocity_analysis(l2, l3, w2, theta_rad, theta3)
        _, a_l1 = utils.slider_crank_acceleration_analysis(l2, l3, w2, w3, alpha2, theta_rad, theta3)
        results.append(a_l1)
    return np.array(results).T


utils.plot(f, title="Problem 1", xlabel="Theta_2 (deg)", xmin=15, xmax=60, colors=["red", "blue", "green"],
           ylabel="Output Linear Acceleration (mm/s^2)",
           labels=list(map(lambda pair: f"alpha2 = {pair[0]}, omega2 = {pair[1]}", alpha2_w2_pairs)))
plt.legend()
plt.show()
