import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patch
import matplotlib.transforms as trans
from lab6_robotics import *  # Assuming this module contains required functions and classes

# Robot model
d = np.zeros(3)                             # displacement along Z-axis
q = np.array([0.2, 0.3, 0.5])               # rotation around Z-axis (theta)
alpha = np.zeros(3)                         # displacement along X-axis
a = np.array([0.75, 0.5, 0.3])              # rotation around X-axis 
revolute = [True, True, True]               # flags specifying the type of joints

robot = MobileManipulator(d, q, a, alpha, revolute) # Manipulator object

# Task definition
base_pos = np.array([0.0, 0.0, 0, 0, 0, 0.0])
ee_pos = np.array([1.0, 0.5])

joint = 2 # select joint [0-5]
joint_angle = 0.3 # selected joint angle [-pi, pi]

joint_max = 0.1
joint_min = -0.5
tasks = [ 
    Configuration2D("End-effector position", 5, ee_pos),
]

# Simulation params
dt = 1.0/60.0

# Lists to store data for plots
time_values = []
position_error_values = []
orientation_error_values = []
velocity_values = [[] for _ in range(5)]  # Initialize an empty list for each DOF

# Simulation time
sim_time = 0.0

def click(event): #get position from mouse
    global x, y, ee_pos
    ee_pos[0] = event.xdata
    ee_pos[1] = event.ydata

# Drawing preparation
fig = plt.figure()
fig.canvas.mpl_connect("button_press_event", click)
ax1 = fig.add_subplot(121, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax1.set_title('Simulation')
ax1.set_aspect('equal')
ax1.grid()
ax1.set_xlabel('x[m]')
ax1.set_ylabel('y[m]')
rectangle = patch.Rectangle((-0.25, -0.15), 0.5, 0.3, color='blue', alpha=0.3)
veh = ax1.add_patch(rectangle)
line, = ax1.plot([], [], 'o-', lw=2) # Robot structure
path, = ax1.plot([], [], 'c-', lw=1) # End-effector path
point, = ax1.plot([], [], 'rx') # Target
PPx = []
PPy = []
ax1.legend(loc="upper left")

# Simulation loop
def simulate(t):
    global tasks, robot, time_values, position_error_values, orientation_error_values, velocity_values, sim_time
    
    ### Recursive Task-Priority algorithm
    P = np.eye(5)
    dq = np.zeros([5,1])
    for i, t in enumerate(tasks):
        t.update(robot)
        J_bar = t.getJacobian()@P

        # W = 0.2*np.diag([2,3,4,1,4])  #1st weight
        # W = 0.2*np.diag([4,3,2,1,4])  #2nd weight
        W = 0.2*np.diag([2,2,3,1,4])    #3rd weight
        
        J_bar_inv = DLS(J_bar, 0.05, W)
        if t.isActive():
            dq += J_bar_inv@(t.dxe - t.getJacobian()@dq)
            P  -= np.linalg.pinv(J_bar)@J_bar

    # Update robot
    robot.update(dq, dt)

    # Update drawing
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[0].getDesired()[0], tasks[0].getDesired()[1])
    eta = robot.getBasePose()
    veh.set_transform(trans.Affine2D().rotate(eta[2,0]) + trans.Affine2D().translate(eta[0,0], eta[1,0]) + ax1.transData)

    # Calculating and storing EE configuration task error
    # print(tasks[0].getError().shape)
    # print(tasks[0].getJacobian().shape)

    position_error_norm = np.linalg.norm(tasks[0].getError()[:3])  # Norm of position error
    orientation_error_norm = np.linalg.norm(tasks[0].getError()[3:])  # Norm of orientation error
    position_error_values.append(position_error_norm)
    orientation_error_values.append(orientation_error_norm)

    # velocity output from TP algorithm for each DOF
    for i in range(5):
        velocity_values[i].append(dq[i, 0])

    time_values.append(sim_time)

    sim_time += dt

    return line, veh, path, point

# Simulation initialization
def init():
    global tasks, ee_pos
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    
    # Set initial random position for the end-effector task
    tasks[0].sigma_d = np.random.choice([-2,2])*np.random.rand(6)
    
    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 5, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)

# Show the plots after simulation finishes
plt.show()

# Plotting the results after simulation
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot evolution of end-effector configuration task error
axs[0].plot(time_values, position_error_values, label='Position Error Norm')
axs[0].plot(time_values, orientation_error_values, label='Orientation Error Norm')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Error')
axs[0].set_title('Evolution of End-effector Configuration Task Error')
axs[0].legend()

# Plot evolution of velocity output from TP algorithm for each DOF
for i in range(5):
    axs[1].plot(time_values, velocity_values[i], label=f'DOF {i+1}')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Velocity')
axs[1].set_title('Evolution of Velocity Output from Task-Priority Algorithm')
axs[1].legend()

plt.tight_layout()
plt.show()
