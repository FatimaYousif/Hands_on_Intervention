from lab4_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot model - 3-link manipulator
d = np.zeros(3)  # displacement along Z-axis
theta = np.array([0, np.pi / 4, np.pi / 4]).reshape(3, 1) # rotation around Z-axis (theta)
alpha = np.zeros(3)  # displacement along X-axis
a = np.array([0.75, 0.5, 0.5])  # rotation around X-axis
revolute = [True, True, True]  # flags specifying the type of joints

robot = Manipulator(d, theta, a, alpha, revolute) # Manipulator object

# Task hierarchy definition
# sequence = order of priority
tasks = [ 
             Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2, 1), link=3), 
            #Orientation2D("End-effector orientation", np.array([[np.pi]]), link=2),
            #Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2, 1), 1),
            #Configuration2D("End-effector config", np.array([1.0, 0.5, np.pi]).reshape(3, 1)),
            # JointPosition("Joint position", np.array([0]),0),
        ] 

#-------------------???
# Set K matrix for task 1
K = 2
tasks[0].setKmatrix(K)
# Set up FFV
FFV = 0.0
tasks[0].setFeedForwardVel(FFV)

# Simulation params
dt = 1.0/60.0

#-------------------
timestamp = []
# vel_threshold = 1.5


# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Simulation initialization
def init():
    global tasks
    global last_time
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])

    # Ex 1 - 11th pt  - diff desired pos everytime in the simulation
    temp = np.random.uniform(-1, 1, size=(2, 1))
    tasks[0].setDesired(temp)

    last_time = timestamp[-1] if timestamp else 0
    
    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy,last_time

    ### Recursive Task-Priority algorithm = SLIDE 7
    # A. Initialize null-space projector ----------- =psuedo inv I (size)
    # B. Initialize output vector (joint velocity)
    # C. Loop over tasks
        # D. Update task state
        # E. Compute augmented Jacobian
        # F. Compute task velocity = DLS (otherwise will have problems in singularity)
        # G. Accumulate velocity ( NEW )
        # H. Update null-space projector
    ###

    # line 1 = SLIDE 7
    P = np.eye(3)    # A
    dq = np.zeros((3,1))    # B

    for task in tasks: # line 2  = C
       
        task.update(robot) # D
        
        J_bar = task.getJacobian() @ P  # line 3 # E
                
        J_bar_inv = DLS(J_bar, 0.1)   # F

        dqi = J_bar_inv @ ((task.getKmatrix() @ task.getError()) + task.getFeedForwardVel() - (task.getJacobian() @ dq)) # control equation

        dq += dqi # G

        P  -= np.linalg.pinv(J_bar)@J_bar   # line 5 = H
           
    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)

    # CHANGE THIS FOR DIFF TASKS

    # EE POS
    point.set_data(tasks[0].getDesired()[0], tasks[0].getDesired()[1])
    
    # point.set_data(tasks[1].getDesired()[0], tasks[0].getDesired()[1])

    # ----------------------------
    timestamp.append(t + last_time)

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()


# ----------------------------
# Evolution of joint positions Plotting
fig_joint = plt.figure()
ax = fig_joint.add_subplot(111, autoscale_on=False, xlim=(0, 60), ylim=(-1, 2))
ax.set_title("Task-Priority (two tasks)")
ax.set_xlabel("Time[s]")
ax.set_ylabel("Error")
ax.grid()
plt.plot(timestamp, tasks[0].err_plot, label="e1 ({})".format(tasks[0].name))
plt.plot(timestamp, tasks[1].err_plot, label="e2 ({})".format(tasks[1].name))

ax.legend()

plt.show()

