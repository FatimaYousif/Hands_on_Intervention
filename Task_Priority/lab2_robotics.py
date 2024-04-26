import numpy as np # Import Numpy

def DH(d, theta, a, alpha):
    '''
        Function builds elementary Denavit-Hartenberg transformation matrices 
        and returns the transformation matrix resulting from their multiplication.

        Arguments:
        d (double): displacement along Z-axis
        theta (double): rotation around Z-axis
        a (double): displacement along X-axis
        alpha (double): rotation around X-axis

        Returns:
        (Numpy array): composition of elementary DH transformations
    '''
    # 1. Build matrices representing elementary transformations (based on input parameters).
    # 2. Multiply matrices in the correct order (result in T).
    
    # TODO

    T_d= np.array([[1,0,0,0], [0,1,0,0], [ 0,0,1,d], [0,0,0,1]])
    T_theta= np.array([[np.cos(theta), -np.sin(theta), 0, 0] , [np.sin(theta), np.cos(theta), 0, 0], [0,0,1,0], [0,0,0,1]])
    T_a= np.array([[1,0,0,a], [0,1,0,0], [ 0,0,1,0], [0,0,0,1]])
    T_alpha= np.array([[1,0,0,0], [0,np.cos(alpha), -np.sin(alpha),0],[0,np.sin(alpha), np.cos(alpha),0], [0,0,0,1]])

    T= T_d @ T_theta @ T_a @ T_alpha

    return T

def kinematics(d, theta, a, alpha):
    '''
        Functions builds a list of transformation matrices, for a kinematic chain,
        descried by a given set of Denavit-Hartenberg parameters. 
        All transformations are computed from the base frame.

        Arguments:
        d (list of double): list of displacements along Z-axis
        theta (list of double): list of rotations around Z-axis
        a (list of double): list of displacements along X-axis
        alpha (list of double): list of rotations around X-axis

        Returns:
        (list of Numpy array): list of transformations along the kinematic chain (from the base frame)
    '''
    T = [np.eye(4)] # Base transformation
    # For each set of DH parameters:
    # 1. Compute the DH transformation matrix.
    # 2. Compute the resulting accumulated transformation from the base frame.
    # 3. Append the computed transformation to T.

    for i in range(len(d)):
        # 1.
        Ti = DH(d[i], theta[i], a[i], alpha[i])
        # 2. and 3. 
        T.append(np.matmul(T[-1], Ti))
    return T

# Inverse kinematics
def jacobian(T, revolute):
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # 1. Initialize J and O.
    # 2. For each joint of the robot
    #   a. Extract z and o.
    #   b. Check joint type.
    #   c. Modify corresponding column of J.
    # 1.
    J = np.zeros((6,len(T)-1))

    # 2.
    for i in range(0,len(T)-1): 
        
        # 2. a. 
        z = np.array(T[i][0:3,2])
        o = np.array(T[-1][0:3,3] - T[i][0:3,3])

        # 2. b.
        if revolute[i]:

        # 2. c.    
            J[:,i] = np.concatenate((np.cross(z,o),z)).reshape((1,6)) # a column for revolute joint
        else:
            J[:,i] = np.concatenate((np.cross(z,o),np.zeros((3,1)) )).reshape((1,6)) # a column for prismatic joint
    return J

# Damped Least-Squares
def DLS(A, damping):
    '''
        Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    # formula in slide 17
    # ζ = J(𝔮).T (J(𝔮)*J(𝔮).T + λ**2 * I)−1 ·xE

    DLS = A.T @ np.linalg.inv(A @ A.T + damping**2 * np.eye((A @ A.T).shape[0]))

    return DLS # Implement the formula to compute the DLS of matrix A.

# Extract characteristic points of a robot projected on X-Y plane
def robotPoints2D(T):
    '''
        Function extracts the characteristic points of a kinematic chain on a 2D plane,
        based on the list of transformations that describe it.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
    
        Returns:
        (Numpy array): an array of 2D points
    '''
    P = np.zeros((2,len(T)))
    for i in range(len(T)):
        P[:,i] = T[i][0:2,3]
    return P