import numpy as np
from matplotlib import pyplot as plt
import utils


l1 = 2.22
l2 = 0.86
l3 = 1.85
l4 = 0.86
AP = 1.33

alpha2 = 1
w2 = 1


def f(theta):
    theta_rad = np.radians(theta)
    theta4, theta3 = utils.vector_loop(l1, l2, l3, l4, theta_rad)
    w3, w4 = utils.velocity_analysis(l2, l3, l4, w2, theta_rad, theta3, theta4)
    alpha3, _ = utils.acceleration_analysis(l2, l3, l4, w2, w3, w4, alpha2, theta_rad, theta3, theta4)
    return alpha3


def g(theta):
    theta_rad = np.radians(theta)
    theta4, theta3 = utils.vector_loop(l1, l2, l3, l4, theta_rad)
    w3, w4 = utils.velocity_analysis(l2, l3, l4, w2, theta_rad, theta3, theta4)
    alpha3, _ = utils.acceleration_analysis(l2, l3, l4, w2, w3, w4, alpha2, theta_rad, theta3, theta4)
    return np.linalg.norm(utils.linear_acceleration(AP, w3, alpha3, theta3), axis=0)


print(f(0))


plt.subplot(211)
utils.plot(f, xmin=-180, xmax=180, title="Problem 2a", xlabel="Theta_2 (deg)", ylabel="Angular Accel. of AB (rad/s^2)")
plt.subplot(212)
utils.plot(g, xmin=-180, xmax=180, title="Problem 2b", xlabel="Theta_2 (deg)", ylabel="Linear Accel. of P (units/s^2)")
plt.tight_layout()
plt.show()
