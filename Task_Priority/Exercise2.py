# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition (3 revolute joint planar manipulator)
d =  np.zeros(3)                            # displacement along Z-axis
q =  np.array([0, np.pi/4, np.pi/4]).reshape(3, 1)                          # rotation around Z-axis (theta)
alpha = np.zeros(3)                        # rotation around X-axis
a =  np.array([0.75, 0.5, 0.5])                       # displacement along X-axis
revolute =  [True, True, True]                # flags specifying the type of joints

#gains
K1 = np.diag([1, 1])
K2 = np.array([1])
vel_threshold = 0.8

# Desired values of task variables
sigma1_d = np.array([0.0, 1.0]).reshape(2,1) # Position of the end-effector
sigma2_d = np.array([[0.0]]) # Position of joint 1

# Simulation params
dt = 1.0/60.0
Tt = 10 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

e1 = []
e2 = []

timestamp = []
last_time = 0

# t_priority = "EE position"
t_priority = "JOINT 1 position"

# Simulation initialization
def init():
    global sigma1_d, last_time
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    last_time = timestamp[-1] if timestamp else 0
    sigma1_d = np.random.uniform(-1, 1, size=(2, 1))
    return line, path, point

# Simulation loop
def simulate(t):
    global q, a, d, alpha, revolute, sigma1_d, sigma2_d
    global PPx, PPy, last_time
    
    # Update robot
    T = kinematics(d, q.flatten(), a, alpha)
    J = jacobian(T, revolute)

    #----------------------------------------------------------------

    # Update control
    if t_priority == "EE position":
  
        # TASK 1
        sigma1 = T[-1][0:2, -1].reshape(2, 1)  # Current position of the end-effector
        err1 = sigma1_d - sigma1  # Error in Cartesian position
        J1 = J[:2, :]  # Jacobian of the first task
        Jinv = DLS(J1, 0.1)
        P1 = np.eye(3) - (np.linalg.pinv(J1) @ J1)  # Null space projector

        # TASK 2
        sigma2 = q[0]  # Current position of joint 1
        err2 = sigma2_d - sigma2  # Error in joint position
        J2 = np.array([1, 0, 0]).reshape(1, 3)  # Jacobian of the second task
        J2bar = J2 @ P1  # Augmented Jacobian

        # Combining tasks
        dq1 = (Jinv @ (err1)).reshape(3, 1)  # Velocity for the first task
        dq12 = dq1 + (DLS(J2bar, 0.1) @ (err2 - J2 @ dq1))  # Velocity for both tasks

    elif t_priority == "JOINT 1 position":
        # TASK 1
        sigma2 = q[0]  # Current position of joint 1
        err2 = sigma2_d - sigma2  # Error in joint position
        J1 = np.array([1, 0, 0]).reshape(1, 3)  # Jacobian of the second task
        Jinv = DLS(J1, 0.1)  # Augmented Jacobian
        P1 = np.eye(3) - (np.linalg.pinv(J1) @ J1)  # Null space projectorJinv

        # TASK 2
        sigma1 = T[-1][0:2, -1].reshape(2, 1)  # Current position of the end-effector
        err1 = sigma1_d - sigma1  # Error in Cartesian position
        J2 = J[:2, :]  # Jacobian of the first task
        J2bar = J2 @ P1

        # Combining tasks
        dq1 = (Jinv @ (err2)).reshape(3, 1)  # Velocity for the first task
        dq12 = dq1 + (DLS(J2bar, 0.1) @ (err1 - J2 @ dq1))  # Velocity for both tasks 
    #----------------------------------------------------------------

    q = q + dq12 * dt # Simulation update

    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma1_d[0], sigma1_d[1])
    
    norm_err1= np.linalg.norm(err1)
    norm_err2= np.linalg.norm(err2)

    e1.append(norm_err1)
    e2.append(norm_err2)

    timestamp.append(t + last_time)


    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# post simulation plotting
fig_joint = plt.figure()
ax = fig_joint.add_subplot(111, autoscale_on=False, xlim=(0, 60), ylim=(-1, 2))
ax.set_title("Task-Priority (two tasks)")
ax.set_xlabel("Time[s]")
ax.set_ylabel("Error")
ax.grid()
plt.plot(timestamp, e1, label="e1 (end-effector position)")
plt.plot(timestamp, e2, label="e2 (joint 1 position)")
ax.legend(loc='upper right')
plt.show()
