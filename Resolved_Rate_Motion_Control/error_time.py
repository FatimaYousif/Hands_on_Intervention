import numpy as np
import matplotlib.pyplot as plt

# Load saved control error norms
transpose_errors = np.load('transpose_errors.npy')
pinverse_errors = np.load('pinverse_errors.npy')
DLS_errors = np.load('DLS_errors.npy')
tt=np.arange(0, 10, 1.0/60.0)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(tt,transpose_errors, label='Transpose Method')
plt.plot(tt,pinverse_errors, label='PInverse Method')
plt.plot(tt,DLS_errors, label='DLS Method')
plt.xlabel('Time Steps')
plt.ylabel('Control Error Norm')
plt.title('Evolution of Control Error Norm over Time')
plt.legend()
plt.grid(True)
plt.show()
