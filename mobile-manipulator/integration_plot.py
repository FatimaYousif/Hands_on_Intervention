import numpy as np
import matplotlib.pyplot as plt

# Load data
base_move = np.load("base_plot_move.npy")
ee_move = np.load("ee_plot_move.npy")
base_rotate = np.load("base_plot_rotate.npy")
ee_rotate = np.load("ee_plot_rotate.npy")
base_both = np.load("base_plot_both.npy")
ee_both = np.load("ee_plot_both.npy")

# Plot
plt.figure(figsize=(8, 6))
plt.plot(base_move[:, 0], base_move[:, 1], label="(Base) move then rotate", color="blue")
plt.plot(ee_move[:, 0], ee_move[:, 1], label="(EE) move then rotate", color="lightblue")
plt.plot(base_rotate[:, 0], base_rotate[:, 1], label="(Base) rotate then move", color="green")
plt.plot(ee_rotate[:, 0], ee_rotate[:, 1], label="(EE) rotate then move", color="lightgreen")
plt.plot(base_both[:, 0], base_both[:, 1], label="(Base) move and rotate", color="red")
plt.plot(ee_both[:, 0], ee_both[:, 1], label="(EE) move and rotate", color="pink")

plt.title("Evolution of Mobile Base and End-effector Position")
plt.xlabel("X[m]")
plt.ylabel("Y[m]")
plt.grid(True)
plt.legend()
plt.show()
