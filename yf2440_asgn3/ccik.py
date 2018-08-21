#!/usr/bin/env python

import math
import numpy
import rospy
from urdf_parser_py.urdf import URDF
from sensor_msgs.msg import JointState
from cartesian_control.msg import CartesianCommand
import tf
import tf.msg
from std_msgs.msg import Float32
from geometry_msgs.msg import Transform

####################
# Returns the angle-axis representation of the rotation contained in the input matrix
# Use like this:
# angle, axis = rotation_from_matrix(R)
def rotation_from_matrix(matrix):
    R = numpy.array(matrix, dtype=numpy.float64, copy=False)
    R33 = R[:3, :3]
    # axis: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = numpy.linalg.eig(R33.T)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    
    axis = numpy.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = numpy.linalg.eig(R)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    # rotation angle depending on axis
    cosa = (numpy.trace(R33) - 1.0) / 2.0
    if abs(axis[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
    elif abs(axis[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
    angle = math.atan2(sina, cosa)
    return angle, axis
######################
def S_matrix(t):
    S = numpy.zeros((3,3))
    S[0,1] = -t[2]
    S[0,2] =  t[1]
    S[1,0] =  t[2]
    S[1,2] = -t[0]
    S[2,0] = -t[1]
    S[2,1] =  t[0]
    return S
######################    
def cartesian_control (joint_transforms, b_T_ee_current, b_T_ee_desired, secondary_objective, q_current, q0_desired, joint_names, joint_axes, num_joints):
    ee_T_b_current = tf.transformations.inverse_matrix(b_T_ee_current)
    # cumpute delta_x in its own coordinate frame {ee}
    delta_x = numpy.dot(ee_T_b_current, b_T_ee_desired)
    # get the translation part
    b_t_ee = tf.transformations.translation_from_matrix(delta_x)
    # get the rotation part
    b_r_ee = delta_x[:3,:3]
    # convert to anglr-axis representation
    angle, axis = rotation_from_matrix(delta_x)
    # desired change in angle around rach of the canonical x, y and z
    angle_change = numpy.dot(angle,axis)

    #xdot = numpy.append(b_t_ee , angle_change)
    xdot = numpy.zeros((6))
    xdot[:3] = b_t_ee
    xdot[3:] = angle_change

    # convert delta_x to a desired velocity v_ee
    p_gain = 0.5
    v_ee = p_gain * xdot

    # scale the velocity
    if numpy.linalg.norm(xdot) > 1.0:
        xdot /= max(xdot)

    J = numpy.zeros((6, 0))
    for j in range(len(joint_transforms)):
        b_T_j = joint_transforms[j]
        j_T_b = tf.transformations.inverse_matrix(b_T_j)
        j_T_ee = numpy.dot(j_T_b, b_T_ee_current)
       
        ee_T_j = tf.transformations.inverse_matrix(j_T_ee)
        ee_R_j = ee_T_j[:3,:3]

        j_t_ee = tf.transformations.translation_from_matrix(j_T_ee)
        S = S_matrix(j_t_ee)

        Vj = numpy.append(numpy.append(ee_R_j, numpy.dot(-ee_R_j, S), axis=1), numpy.append(numpy.zeros([3,3]), ee_R_j,axis=1), axis=0)

        if joint_axes[j][0] != 0:
            if  joint_axes[j][0] < 0:
                J = numpy.column_stack((J, -Vj[:,3]))
            else:
                J = numpy.column_stack((J, -Vj[:,3]))
        elif joint_axes[j][1] != 0:
            if  joint_axes[j][1] < 0:
                J = numpy.column_stack((J, -Vj[:,4]))
            else:
                J = numpy.column_stack((J, Vj[:,4]))
        else :
            if  joint_axes[j][2] < 0:
                J = numpy.column_stack((J, -Vj[:,5]))
            else:
                J = numpy.column_stack((J, Vj[:,5]))
        
    Jp = numpy.linalg.pinv(J, rcond=1e-2)
    qdot = numpy.zeros(len(joint_transforms))
    qdot = numpy.dot(Jp, v_ee)

    # implements the null-space control
    if secondary_objective == True:
        Jp = numpy.linalg.pinv(J, rcond=0)
        qdot_second = numpy.array(numpy.zeros(num_joints))
        qdot_second[0] = q0_desired - q_current[0]
	qdot_second=numpy.transpose(qdot_second)
        qdot_null = numpy.dot(numpy.identity(num_joints) - numpy.dot(Jp, J), qdot_second)
        qdot = numpy.dot(Jp, v_ee) + qdot_null

    #for j in range(len(qdot)):
        #if abs(qdot[j]) > 1:
            #if qdot[j]<0:
                #qdot[j] = -1
            #else:
                #qdot[j] = 1
    # scale the velocity
    if numpy.linalg.norm(qdot) > 1.0:
        qdot /= max(qdot)

    return (qdot,joint_names)

############################################
def inverse_kinematics (joint_transforms, b_T_ee_current, b_T_ee_desired, joint_names, joint_axes):    
    ee_T_b_current = tf.transformations.inverse_matrix(b_T_ee_current)
    # cumpute delta_x in its own coordinate frame {ee}
    delta_x = numpy.dot(ee_T_b_current, b_T_ee_desired)
    # get the translation part
    b_t_ee = tf.transformations.translation_from_matrix(delta_x)
    # get the rotation part
    b_r_ee = delta_x[:3,:3]
    # convert to anglr-axis representation
    angle, axis = rotation_from_matrix(delta_x)
    # desired change in angle around rach of the canonical x, y and z
    angle_change = numpy.dot(angle,axis)

    #xdot = numpy.append(b_t_ee , angle_change)
    xdot = numpy.zeros((6))
    xdot[:3] = b_t_ee
    xdot[3:] = angle_change

    # convert delta_x to a desired velocity v_ee
    p_gain = 0.5
    v_ee = p_gain * xdot

    J = numpy.zeros((6, 0))
    for j in range(len(joint_transforms)):
        b_T_j = joint_transforms[j]
        j_T_b = tf.transformations.inverse_matrix(b_T_j)
        j_T_ee = numpy.dot(j_T_b, b_T_ee_current)
       
        ee_T_j = tf.transformations.inverse_matrix(j_T_ee)
        ee_R_j = ee_T_j[:3,:3]

        j_t_ee = tf.transformations.translation_from_matrix(j_T_ee)
        S = S_matrix(j_t_ee)

        Vj = numpy.append(numpy.append(ee_R_j, numpy.dot(-ee_R_j, S), axis=1), numpy.append(numpy.zeros([3,3]), ee_R_j,axis=1), axis=0)

        if joint_axes[j][0] != 0:
            if  joint_axes[j][0] < 0:
                J = numpy.column_stack((J, -Vj[:,3]))
            else:
                J = numpy.column_stack((J, Vj[:,3]))
        elif joint_axes[j][1] != 0:
            if  joint_axes[j][1] < 0:
                J = numpy.column_stack((J, -Vj[:,4]))
            else:
                J = numpy.column_stack((J, Vj[:,4]))
        else :
            if  joint_axes[j][2] < 0:
                J = numpy.column_stack((J, -Vj[:,5]))
            else:
                J = numpy.column_stack((J, Vj[:,5]))
        
    Jp = numpy.linalg.pinv(J, rcond=1e-2)
    return (xdot,Jp)
    #return (x_c,Jp)

########################################
class CartesianControl(object):
    def __init__(self):
        #Loads the robot model, which contains the robot's kinematics information
        self.robot = URDF.from_parameter_server() 

        self.joint_transforms = []
        self.q_current = []
        self.q0_desired = 0
        self.x_current = tf.transformations.identity_matrix()
        self.x_current2 = tf.transformations.identity_matrix()
        self.R_base = tf.transformations.identity_matrix() #???
        self.x_desired = tf.transformations.identity_matrix()

        # This is where we hold general info about the joints
        self.num_joints = 0
        self.joint_names = []
        self.joint_axes = []
        # Prepare general information about the robot
        self.get_joint_info()

        #Subscribes to information about what the current joint values are.
        rospy.Subscriber("/joint_states", JointState, self.joint_callback)
 
        rospy.Subscriber("/cartesian_command", CartesianCommand,  self.command_callback)

        rospy.Subscriber("/ik_command", Transform,  self.ik_callback)

        self.pub_jv = rospy.Publisher("/joint_velocities", JointState, queue_size=1)

        self.ik = rospy.Publisher("/joint_command", JointState, queue_size=1)
####################################
    def ik_callback(self, command):
        msg2 = JointState()
        trans = tf.transformations.translation_matrix((command.translation.x,
                                                  command.translation.y,
                                                  command.translation.z))
        rot = tf.transformations.quaternion_matrix((command.rotation.x,
                                                command.rotation.y,
                                                command.rotation.z,
                                                command.rotation.w))

        self.x_desired = numpy.dot(trans,rot)
        root = self.robot.get_root()
        #qc = -2*numpy.pi + 4*numpy.pi*numpy.random.rand(self.num_joints)
        n = 0
        qc = [0] * self.num_joints
        for i in range(self.num_joints):
            qc[i] = numpy.random.uniform(0, math.pi)
        while True:
            T = tf.transformations.identity_matrix()
            self.joint_transforms = []
            self.process_link_recursive2(root, T, self.current_joint_state, qc)
            (delta_x,Jp) = inverse_kinematics(self.joint_transforms, self.x_current2, self.x_desired, self.joint_names, self.joint_axes)
            if numpy.amax(numpy.abs(delta_x)) < 0.01:
                break
            delta_q = numpy.dot(Jp, delta_x)
            for i in range (0, self.num_joints):
                index = self.current_joint_state.name.index(self.joint_names[i])
                qc[index] += delta_q[i] 
            if n > 500:
                #qc = -2*numpy.pi + 4*numpy.pi*numpy.random.rand(self.num_joints)
                for i in range(self.num_joints):
                    qc[i] = numpy.random.uniform(0, math.pi)
                n = 0
            n  = n + 1
        msg2.position = qc
        #msg2.name = self.joint_names      
        self.ik.publish(msg2)
#####################################
    # get joint information
    def get_joint_info(self):
        self.num_joints = 0
        self.joint_names = []
        self.joint_axes = []
        link = self.robot.get_root()
        while True:
            if link not in self.robot.child_map: break            
            (joint_name, next_link) = self.robot.child_map[link][0]
            current_joint = self.robot.joint_map[joint_name]        
            if current_joint.type != 'fixed':
                self.num_joints = self.num_joints + 1
                self.joint_names.append(current_joint.name)
                self.joint_axes.append(current_joint.axis)
            link = next_link
################################    
    # convert x_target to matrix
    def command_callback(self, command):
        msg = JointState()
        trans = tf.transformations.translation_matrix((command.x_target.translation.x,
                                                  command.x_target.translation.y,
                                                  command.x_target.translation.z))
        rot = tf.transformations.quaternion_matrix((command.x_target.rotation.x,
                                                command.x_target.rotation.y,
                                                command.x_target.rotation.z,
                                                command.x_target.rotation.w))
        self.x_desired = numpy.dot(trans,rot)
        root = self.robot.get_root()
        T = tf.transformations.identity_matrix()
        self.joint_transforms = []
        self.q_current = self.current_joint_state.position
        self.process_link_recursive(root, T, self.current_joint_state)
        self.q0_desired = command.q0_target
        if command.secondary_objective == False:
            (dq,joint_names) = cartesian_control(self.joint_transforms, self.x_current, self.x_desired, False, self.q_current, self.q0_desired, self.joint_names, self.joint_axes, self.num_joints)
            msg.velocity = dq
            msg.name = joint_names
        elif command.secondary_objective == True:
            (dq,joint_names) = cartesian_control(self.joint_transforms, self.x_current, self.x_desired, True, self.q_current, self.q0_desired, self.joint_names, self.joint_axes, self.num_joints)
            msg.velocity = dq
            msg.name = joint_names
        else:            
            msg.velocity = numpy.zeros(self.num_joints)
            msg.name = joint_names
        self.pub_jv.publish(msg)
##################################       
    def joint_callback(self, joint_values):
        self.current_joint_state = joint_values
##################################
    def process_link_recursive(self, link, T, joint_values):
        if link not in self.robot.child_map: 
            self.x_current = T
            return 
        for i in range(0,len(self.robot.child_map[link])):
            (joint_name, next_link) = self.robot.child_map[link][i]
            if joint_name not in self.robot.joint_map:
                rospy.logerror("Joint not found in map")
                continue
            current_joint = self.robot.joint_map[joint_name]        
            trans_matrix = tf.transformations.translation_matrix((current_joint.origin.xyz[0], 
                                                                  current_joint.origin.xyz[1],
                                                                  current_joint.origin.xyz[2]))
            rot_matrix = tf.transformations.euler_matrix(current_joint.origin.rpy[0], 
                                                         current_joint.origin.rpy[1],
                                                         current_joint.origin.rpy[2], 'rxyz')
            origin_T = numpy.dot(trans_matrix, rot_matrix)
            current_joint_T = numpy.dot(T, origin_T)
            if current_joint.type != 'fixed':
                if current_joint.name not in joint_values.name:
                    rospy.logerror("Joint not found in list")
                    continue
                # compute transform that aligns rotation axis with corresponding axis
                self.joint_transforms.append(current_joint_T)
                index = joint_values.name.index(current_joint.name)
                angle = joint_values.position[index]
                joint_rot_T = tf.transformations.rotation_matrix(angle, numpy.asarray(current_joint.axis))
                next_link_T = numpy.dot(current_joint_T, joint_rot_T) 
            else:
                next_link_T = current_joint_T

            self.process_link_recursive(next_link, next_link_T, joint_values)
#################################
    def process_link_recursive2(self, link, T, joint_values, qc):
        if link not in self.robot.child_map: 
            self.x_current2 = T
            return 
        for i in range(0,len(self.robot.child_map[link])):
            (joint_name, next_link) = self.robot.child_map[link][i]
            if joint_name not in self.robot.joint_map:
                rospy.logerror("Joint not found in map")
                continue
            current_joint = self.robot.joint_map[joint_name]        
            trans_matrix = tf.transformations.translation_matrix((current_joint.origin.xyz[0], 
                                                                  current_joint.origin.xyz[1],
                                                                  current_joint.origin.xyz[2]))
            rot_matrix = tf.transformations.euler_matrix(current_joint.origin.rpy[0], 
                                                         current_joint.origin.rpy[1],
                                                         current_joint.origin.rpy[2], 'rxyz')
            origin_T = numpy.dot(trans_matrix, rot_matrix)
            current_joint_T = numpy.dot(T, origin_T)
            if current_joint.type != 'fixed':
                if current_joint.name not in joint_values.name:
                    rospy.logerror("Joint not found in list")
                    continue
                # compute transform that aligns rotation axis with corresponding axis
                self.joint_transforms.append(current_joint_T)
                index = joint_values.name.index(current_joint.name)
                #angle = joint_values.position[index]
                angle = qc[index]
                joint_rot_T = tf.transformations.rotation_matrix(angle, numpy.asarray(current_joint.axis))
                next_link_T = numpy.dot(current_joint_T, joint_rot_T) 
            else:
                next_link_T = current_joint_T

            self.process_link_recursive2(next_link, next_link_T, joint_values,qc)

###############################
if __name__ == '__main__':
    rospy.init_node('cartesian_control', anonymous=True)
    cc = CartesianControl()
    rospy.spin()
