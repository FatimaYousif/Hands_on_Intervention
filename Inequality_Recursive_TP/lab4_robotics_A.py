from lab2_robotics import *  # Includes numpy import


def jacobianLink(T, revolute, link):  # Needed in Exercise 2
    """
    Function builds a Jacobian for the end-effector of a robot,
    described by a list of kinematic transformations and a list of joint types.

    Arguments:
    T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
    revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    link(integer): index of the link for which the Jacobian is computed

    Returns:
    (Numpy array): end-effector Jacobian
    """
    # Code almost identical to the one from lab2_robotics...

"""
    Class representing a robotic manipulator.
"""


class Manipulator:
    """
    Constructor.

    Arguments:
    d (Numpy array): list of displacements along Z-axis
    theta (Numpy array): list of rotations around Z-axis
    a (Numpy array): list of displacements along X-axis
    alpha (Numpy array): list of rotations around X-axis
    revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    """

    def __init__(self, d, theta, a, alpha, revolute):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute
        self.dof = len(self.revolute)
        self.q = np.zeros(self.dof).reshape(-1, 1)
        self.update(0.0, 0.0)

    """
        Method that updates the state of the robot.

        Arguments:
        dq (Numpy array): a column vector of joint velocities
        dt (double): sampling time
    """

    def update(self, dq, dt):
        self.q += dq * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
        self.T = kinematics(self.d, self.theta, self.a, self.alpha)

    """ 
        Method that returns the characteristic points of the robot.
    """

    def drawing(self):
        return robotPoints2D(self.T)

    """
        Method that returns the end-effector Jacobian.
    """

    def getEEJacobian(self):
        return jacobian(self.T, self.revolute)

    """
        Method that returns the end-effector transformation.
    """

    def getEETransform(self):
        return self.T[-1]

    """
        Method that returns the position of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): position of the joint
    """

    def getJointPos(self, joint):
        return self.q[joint]

    """
        Method that returns number of DOF of the manipulator.
    """

    def getDOF(self):
        return self.dof

"""
    Base class representing an abstract Task.
"""


class Task:
    """
    Constructor.

    Arguments:
    name (string): title of the task
    desired (Numpy array): desired sigma (goal)
    """

    def __init__(self, name, desired):
        self.name = name  # task title
        self.sigma_d = desired  # desired sigma
        self.err_plot = []

    """
        Method updating the task variables (abstract).

        Arguments:
        robot (object of class Manipulator): reference to the manipulator
    """

    def update(self, robot):
        pass

    """ 
        Method setting the desired sigma.

        Arguments:
        value(Numpy array): value of the desired sigma (goal)
    """

    def setDesired(self, value):
        self.sigma_d = value

    """
        Method returning the desired sigma.
    """

    def getDesired(self):
        return self.sigma_d

    """
        Method returning the task Jacobian.
    """

    def getJacobian(self):
        return self.J

    """
        Method returning the task error (tilde sigma).
    """

    def getError(self):
        return self.err

"""
    Subclass of Task, representing the 2D position task.
"""
class Position2D(Task):
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.J = np.zeros(0)
        self.err = np.zeros(desired.shape)  # Initialize with proper dimensions

    def update(self, robot):
        self.J = robot.getEEJacobian()[: len(self.sigma_d), :]  # Update task Jacobian
        self.err = self.getDesired() - robot.getEETransform()[: len(self.sigma_d), -1].reshape(self.sigma_d.shape)  # Update task error
        
        # also updating error plot
        self.err_plot.append(np.linalg.norm(self.err))
        


"""
    Subclass of Task, representing the 2D orientation task.
"""


class Orientation2D(Task):
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.J = np.zeros((len(desired), 3))  # Initialize with proper dimensions
        self.err = np.zeros(desired.shape)  # Initialize with proper dimensions
    

    def update(self, robot):
        self.J = robot.getEEJacobian()[-1, :].reshape((1, 3))  # Update task Jacobian
        T = robot.getEETransform()  # Get Transformations matrix form robot
        yaw = np.arctan2(T[1, 0], T[0, 0]).reshape(self.sigma_d.shape)  # Extract yaw from Transformation matrix
        self.err = self.getDesired() - yaw  # Update task error

        # also updating error plot
        self.err_plot.append(np.linalg.norm(self.err))
"""
    Subclass of Task, representing the 2D configuration task.
"""


class Configuration2D(Task):
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.J = np.zeros((3, 3))
        self.err = np.zeros(desired.shape)

    def update(self, robot):
        self.J = robot.getEEJacobian()[[0, 1, -1], :]
        T = robot.getEETransform()
        currentpos = T[:2, -1]
        currentyaw = np.arctan2(T[1, 0], T[0, 0])
        self.err = self.getDesired() - np.block([currentpos, currentyaw]).reshape(self.sigma_d.shape)
     
        # also updating error plot
        self.err_plot.append(np.linalg.norm(self.err))
""" 
    Subclass of Task, representing the joint position task.
"""


class JointPosition(Task):
    def __init__(self, name, desired, joint=0):
        super().__init__(name, desired)
        self.joint = joint
        self.J = np.zeros((1, 3))
        self.err = np.zeros(desired.shape)

    def update(self, robot):
        self.J[0,self.joint] = 1
        self.err = self.getDesired() - robot.getJointPos(self.joint).reshape(self.sigma_d.shape)

        # also updating error plot
        self.err_plot.append(np.linalg.norm(self.err))