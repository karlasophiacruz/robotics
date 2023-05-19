"""scara_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor

import numpy as np
from controller import Supervisor  # type: ignore

np.set_printoptions(precision=4, suppress=True)


def c(x):
    return np.cos(x)


def s(x):
    return np.sin(x)


def fkine(q):
    """
    Implement the forward kinematics of the SCARA robot.
    Input: q - joint angles (list of 4 floats)
    Output: T - transformation matrix (4x4 numpy array)
    """
    # q1 = angulo da junta 0
    # q2 = angulo da junta 1
    # q3 = deslocamento do efetuador
    # q4 = angulo do efetuador

    q1, q2, q3, q4 = q
    d0, d1, d2, d3, d4 = 0.65, 0.1, 0.5, 0.5, -0.2
    
    # Transformacao da base em relacao a junta 1 (sem rotacao, deslocamento em z (altura))
    # vermelho da base em relacao ao vermelho de cima
    #T0 = np.array([[1, 0, 0, 0],
     #              [0, 1, 0, 0],
     #              [0, 0, 1, d0],
      #             [0, 0, 0, 1]])
                   
    T0 = np.array([[c(q1), -s(q1), 0, 0],
                   [s(q1), c(q1), 0, 0],
                   [0, 0, 1, d0],
                   [0, 0, 0, 1]])
                   
    # Transformacao da junta 1 em relacao a junta 2 (rotacao em z, deslocamento em z (altura) e em x (comprimento))
    # primeiro verde em relacao ao primeiro amarelo      
    T1 = np.array([[c(q2), -s(q2), 0, d2],
                   [s(q2), c(q2), 0, 0],
                   [0, 0, 1, d1],
                   [0, 0, 0, 1]])

    # Transformacao da junta 2 em relacao a junta 3 (rotacao em z, deslocamento em x (comprimento))
    # primeiro amarelo em relacao ao segundo amarelo  
    #T2 = np.array([[c(q2), -s(q2), 0, d3],
     #              [s(q2), c(q2), 0, 0],
     #              [0, 0, 1, 0],
     #              [0, 0, 0, 1]])
                   
    T2 = np.array([[1, 0, 0, d3],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Transformacao da junta 3 em relacao a garra (rotacao em x, deslocamento em z (altura))
    # segundo amarelo em relacao ao rosa               
    #T3 = np.array([[1, 0, 0, 0],
    #               [0, c(q3), -s(q3), 0],
    #               [0, s(q3), c(q3), -(d4 + q2)],
     #              [0, 0, 0, 1]])
    T3 = np.array([[c(q4), -s(q4), 0, 0],
                   [-s(q4), -c(q4), 0, 0],
                   [0, 0, -1, -q3],
                   [0, 0, 0, 1]])               

    T = np.dot(np.dot(np.dot(T0, T1), T2), T3)
    
    return T


def invkine(x, y, z, phi):
    """
    Implement the inverse kinematics of the SCARA robot.
    Input: x, y, z, phi - desired end-effector pose
    Output: q - joint angles (list of 4 floats)
    """
    a1, a2, d0, d1, d4 = 0.5, 0.5, 0.65, 0.1, 0.2
    
    c2 = (x**2 + y**2 - a1**2 - a2**2) / (2 * a1 * a2)
    
    if c2 < -1 or c2 > 1:
        raise 
    
    q2_l = np.arctan2(np.sqrt(1 - c2**2), c2)
    q2_ll = np.arctan2(-np.sqrt(1 - c2**2), c2)
    
    k1_l = a1 + a2 * c(q2_l)
    k1_ll = a1 + a2 * c(q2_ll)
    
    k2_l = a2 * s(q2_l)
    k2_ll = a2 * s(q2_ll)
    
    q1_l = np.arctan2(y, x) - np.arctan2(k2_l, k1_l)
    q1_ll = np.arctan2(y, x) - np.arctan2(k2_ll, k1_ll)
    
    q3 = d0 + d1 - d4 - z
    
    q4_l = phi - q1_l - q2_l
    q4_ll = phi - q1_ll - q2_ll
    
    return [q1_ll, q2_ll, q3, q4_ll]


class Scara:
    def __init__(self):
        # create the Robot instance.
        self.robot = Supervisor()

        # get the time step of the current world.
        self.timestep = int(self.robot.getBasicTimeStep())

        # get joint motor devices
        self.joints = [self.robot.getDevice("joint%d" % i) for i in range(1, 5)]

        # get duck reference
        self.duck = self.robot.getFromDef("DUCK")

        # get gripper reference
        self.gripper = self.robot.getFromDef("GRIPPER")

        self.grasp = False
        self.grasp_prev = False

    def set_position(self, q):
        """
        Set the joint positions of the SCARA robot.
        Input: q - joint angles (list of 4 floats)
        """
        for joint, value in zip(self.joints, q):
            joint.setPosition(value)

    def is_colliding(self, ds=0.15):
        """
        Check if the gripper is colliding with the duck.
        Input: ds - safety distance (float)
        Output: new_pos - new gripper position (list of 3 floats)
                new_yaw - new gripper yaw (float)
                colliding - True if colliding, False otherwise (bool)
        """
        dp = np.array(self.duck.getPose()).reshape(4, 4)[:-1, -1]
        gt = np.array(self.gripper.getPose()).reshape(4, 4)
        gp = gt[:-1, -1]
        gy = np.arctan2(gt[1, 0], gt[0, 0])
        return (
            (gp + np.array([0.0, 0.0, -0.5 * ds])).tolist(),
            gy,
            (np.linalg.norm(dp - gp) < ds),
        )

    def step(self):
        """
        Perform one simulation step.
        Output: -1 if Webots is stopping the controller, 0 otherwise (int)
        """
        if self.grasp is True and self.grasp_prev is False:
            print("GRASP STARTED")
            self.grasp_prev = True
        elif self.grasp is False and self.grasp_prev is True:
            print("GRASP ENDED")
            self.grasp_prev = False

        if self.grasp:
            new_pos, new_yaw, colliding = self.is_colliding()
            self.duck.resetPhysics()
            if colliding:
                self.duck.getField("translation").setSFVec3f(new_pos)
                self.duck.getField("rotation").setSFRotation([0.0, 0.0, 1.0, new_yaw])

        return self.robot.step(self.timestep)

    def delay(self, ms):
        """
        Delay the simulation for a given time.
        Input: ms - delay time in milliseconds (int)
        """
        counter = ms / self.timestep
        while (counter > 0) and (self.step() != -1):
            counter -= 1

    def hold(self):
        """
        Hold the duck.
        """
        self.grasp = True

    def release(self):
        """
        Release the duck.
        """
        self.grasp = False

    def getDuckPose(self):
        """
        Get the duck pose.
        Output: position - duck position (list of 3 floats)
                yaw - duck yaw (float)
        """
        pose = np.array(self.duck.getPose()).reshape(4, 4)
        position = pose[:-1, -1].tolist()
        yaw = np.arctan2(pose[1, 0], pose[0, 0])
        return position + [yaw]


if __name__ == "__main__":
    scara = Scara()
    
    box_pose = [0.85, -0.3, 0.4]
    # Main loop:
    # Perform simulation steps until Webots is stopping the controller
    
    while scara.step() != -1:
        if not scara.grasp:
            duck_pose = scara.getDuckPose()
            
            desired_angles = invkine(*duck_pose)
            
            scara.set_position(desired_angles)
            
            scara.delay(5000)
            
            scara.hold()
         
        else:
            box_pose = [box_pose[0], 
                        box_pose[1],
                        box_pose[2] + 0.17,
                        -np.pi/2]
                       
            desired_angles = invkine(*box_pose)
            
            scara.set_position(desired_angles)
            
            scara.delay(5000)
            
            scara.release()
            
            # Enter here exit cleanup code.
            break
    
