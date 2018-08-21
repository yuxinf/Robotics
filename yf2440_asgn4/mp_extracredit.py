#!/usr/bin/env python

import math
import random
import numpy
import sys
import time

from copy import deepcopy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform
import geometry_msgs.msg
from trajectory_msgs.msg import JointTrajectory
import moveit_msgs.msg
import moveit_msgs.srv
import rospy
import tf
from trajectory_msgs.msg import JointTrajectoryPoint
from urdf_parser_py.urdf import URDF

def S_matrix(t):
    S = numpy.zeros((3,3))
    S[0,1] = -t[2]
    S[0,2] =  t[1]
    S[1,0] =  t[2]
    S[1,2] = -t[0]
    S[2,0] = -t[1]
    S[2,1] =  t[0]
    return S

######################################################################################################

# IK for extra credit
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

#####################################################################
def convert_to_message(T):
    t = geometry_msgs.msg.Pose()
    position = tf.transformations.translation_from_matrix(T)
    orientation = tf.transformations.quaternion_from_matrix(T)
    t.position.x = position[0]
    t.position.y = position[1]
    t.position.z = position[2]
    t.orientation.x = orientation[0]
    t.orientation.y = orientation[1]
    t.orientation.z = orientation[2]
    t.orientation.w = orientation[3]        
    return t

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

class MoveArm(object):

    def __init__(self):

        self.robot = URDF.from_parameter_server()

        self.joint_transforms = []
        self.q_current = []
        self.q0_desired = 0
        self.x_current = tf.transformations.identity_matrix()
        self.x_desired = tf.transformations.identity_matrix()

        # This is where we hold general info about the joints
        self.num_joints = 0
        self.joint_names = []
        self.joint_axes = []
        # Prepare general information about the robot
        self.get_joint_info()

        #self.num_joints = 7
        #self.joint_names = ["lwr_arm_0_joint",
			    #"lwr_arm_1_joint",
			    #"lwr_arm_2_joint",
			    #"lwr_arm_3_joint",
			    #"lwr_arm_4_joint",
			    #"lwr_arm_5_joint",
			    #"lwr_arm_6_joint"]

        #self.q_sample = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.q_sample = []
        for i in range(self.num_joints):
            self.q_sample.append(0.1)

        # Subscribes to information about what the current joint values are
        rospy.Subscriber("/joint_states", JointState, self.joint_states_callback)
        # Subscribe to command for motion planning goal
        rospy.Subscriber("/motion_planning_goal", Transform, self.motion_callback)

        # Publish trajectory command
        self.pub_trajectory = rospy.Publisher("/joint_trajectory", JointTrajectory, queue_size=1)

        # Initialize variables
        self.joint_state = JointState()

        # Wait for moveit IK service
        rospy.wait_for_service("compute_ik")
        self.ik_service = rospy.ServiceProxy('compute_ik',  moveit_msgs.srv.GetPositionIK)
        print "IK service ready"

        # Wait for validity check service
        rospy.wait_for_service("check_state_validity")
        self.state_valid_service = rospy.ServiceProxy('check_state_validity', moveit_msgs.srv.GetStateValidity)
        print "State validity service ready"

        # MoveIt parameter
        self.group_name = "lwr_arm"
        #self.group_name = "manipulator"
############################################################################################
    # IK provided
    """ This function will perform IK for a given transform T of the end-effector. It 
    returns a list q[] of 7 values, which are the result positions for the 7 joints of 
    the KUKA arm, ordered from proximal to distal. If no IK solution is found, it 
    returns an empy list.
    """
    def IK(self, T_goal):
        req = moveit_msgs.srv.GetPositionIKRequest()
        req.ik_request.group_name = self.group_name
        req.ik_request.robot_state = moveit_msgs.msg.RobotState()
        req.ik_request.robot_state.joint_state = self.joint_state
        req.ik_request.avoid_collisions = True
        req.ik_request.pose_stamped = geometry_msgs.msg.PoseStamped()
        #req.ik_request.pose_stamped.header.frame_id = "world_link"
        req.ik_request.pose_stamped.header.frame_id = self.robot.get_root()
        req.ik_request.pose_stamped.header.stamp = rospy.get_rostime()
        req.ik_request.pose_stamped.pose = convert_to_message(T_goal)
        req.ik_request.timeout = rospy.Duration(3.0)
        res = self.ik_service(req)
        q = []
        if res.error_code.val == res.error_code.SUCCESS:
            q = res.solution.joint_state.position
        return q
#############################################################################################

    """ This function checks if a set of joint angles q[] creates a valid state, or 
    one that is free of collisions. The values in q[] are assumed to be values for 
    the joints of the KUKA arm, ordered from proximal to distal. 
    """
    #def is_state_valid(self, q):
        #req = moveit_msgs.srv.GetStateValidityRequest()
        #req.group_name = self.group_name
        #req.robot_state = moveit_msgs.msg.RobotState()
        #req.robot_state.joint_state.name = ["lwr_arm_0_joint",
                                            #"lwr_arm_1_joint",
                                            #"lwr_arm_2_joint",
                                            #"lwr_arm_3_joint",
                                            #"lwr_arm_4_joint",
                                            #"lwr_arm_5_joint",
                                            #"lwr_arm_6_joint"]
        #req.robot_state.joint_state.position = q
        #req.robot_state.joint_state.velocity = numpy.zeros(7)
        #req.robot_state.joint_state.effort = numpy.zeros(7)
        #req.robot_state.joint_state.header.stamp = rospy.get_rostime()
        #res = self.state_valid_service(req)
        #return res.valid

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

    def is_state_valid(self, q):
	req = moveit_msgs.srv.GetStateValidityRequest()
	req.group_name = self.group_name
	current_joint_state = deepcopy(self.joint_state)
	current_joint_state.position = list(current_joint_state.position)
	self.joint_state_from_q(current_joint_state, q)
	req.robot_state = moveit_msgs.msg.RobotState()
	req.robot_state.joint_state = current_joint_state
	res = self.state_valid_service(req)
	return res.valid

    def joint_state_from_q(self, joint_state, q):
	for i in range(0,self.num_joints):
            name = self.joint_names[i]
            #name = self.joint_state.name[i]
            if name not in joint_state.name:
		print "ERROR: joint name not found"
	    i = joint_state.name.index(name)
	    joint_state.position[i] = q[i]

    
    # returns a list of joints q[] in a particular order containing just 
    def q_from_joint_state(self, joint_state):
        q = []
	for i in range(0,self.num_joints):
            name = self.joint_names[i]
            #name = self.joint_state.name[i]
            if name not in joint_state.name:
	        print "ERROR: joint name not found"
		return 0
	    index = joint_state.name.index(name)
	    q.append(joint_state.position[index])
	return q

    def test_collision(self, closest_point, target_point):
        step_size = self.q_sample
        T = []
        max_num_points = max(numpy.ceil(numpy.true_divide(abs(numpy.subtract(target_point,closest_point)), step_size)))
        m = numpy.true_divide(numpy.subtract(target_point, closest_point), max_num_points)
        b = closest_point
        for i in range(int(max_num_points)+1):
            T.append(m*i + b)
	for row in T:
	    if self.is_state_valid(row) == False:
	        return False
	return True
	
    def get_closest_point(self, tree, q):
	distances = [numpy.linalg.norm(numpy.subtract(q_pos, q)) for i,q_pos in enumerate(d["position"] for d in tree)]
	min_distance_index = distances.index(min(distances))
	closest_point = tree[min_distance_index].get("position")
	return min_distance_index, closest_point

    def get_point_at_distance(self, closest_point, random_point, K):
        v = numpy.subtract(random_point,closest_point)
	vector = v / numpy.linalg.norm(v)
	vector *= K
	vector = numpy.add(vector, closest_point)
	return vector

    def project_plan(self, q_start, q_goal):
	q_list = self.rrt(q_start, q_goal)
	joint_trajectory = self.create_trajectory(q_list)
	return joint_trajectory

    def create_trajectory(self, q_list):
        joint_trajectory = JointTrajectory()
	for i in range(0, len(q_list)):
            p = []
	    point = JointTrajectoryPoint()
	    point.positions = list(q_list[i])
            for k in range (self.num_joints):
                j = self.joint_names.index(self.joint_state.name[k])
                p.append(point.positions[j])
            point.positions = list(p)
	    joint_trajectory.points.append(point)
	joint_trajectory.joint_names = self.joint_names
        #joint_trajectory.joint_names = self.joint_state.name
	return joint_trajectory

    def rrt(self, q_start, q_goal):
        # create a dictionary of position and a reference to its parent node
        rrt_object = {"position" : q_start, "parent_node" : -1}
	rrt_list = []
	rrt_list.append(rrt_object.copy())
        
        maximum_nodes = 200
	maximum_time_secs = 250
	start = rospy.get_rostime().secs
	now = rospy.get_rostime().secs

        while (len(rrt_list) < maximum_nodes) or ((now - start) < maximum_time_secs):
            # Sample random_point in C-space
            random_point = [0] * self.num_joints
            for i in range(self.num_joints):
                #random_point[i] = numpy.random.uniform(-math.pi, math.pi)
                random_point[i] = numpy.random.uniform(-3.14159, 3.14159)
        
            # Find point closest_point in tree that is closest to random_point
            min_distance_index, closest_point = self.get_closest_point(rrt_list, random_point)

            # Using the closest point above to find the new point that lies a predefined distance
            new_point = self.get_point_at_distance(closest_point, random_point, 0.5)

            # Check whether new brach intersects the obstacles
            if self.test_collision(closest_point, new_point) == True:
                rrt_object.update({"position" : new_point})
	        rrt_object.update({"parent_node": min_distance_index})
                #print(min_distance_index)
	        rrt_list.append(rrt_object.copy())

                if self.test_collision(closest_point, q_goal) == True:
                    rrt_object.update({"parent_node": len(rrt_list)-1})
                    rrt_object.update({"position" : q_goal})
	            rrt_list.append(rrt_object.copy())
                    break
            now = rospy.get_rostime().secs
            #print(rrt_list)

        # trace back
        q_list = [q_goal]
	parent_node = rrt_list[-1].get("parent_node")
        
        
        while True:
            q_list.insert(0, rrt_list[parent_node].get("position"))
            if parent_node <= 0:
	        break
	    else:
		parent_node = rrt_list[parent_node].get("parent_node")

        #print(q_list)
        # shortcutting
        q_list_copy = []
	q_list_copy.append(q_list[0])
        n=0
                
        for i in range(len(q_list)-2):
            if self.test_collision(q_list[n], q_list[i+2]) == False:
	        q_list_copy.append(q_list[i+1])
                n = i +1

	q_list_copy.append(q_list[-1])
	q_list = q_list_copy

        # re-sample
        #step = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        step = []
        for i in range(self.num_joints):
            step.append(0.5)

        T = []
        for i in range(len(q_list)-1):
            max_num_points = max(numpy.ceil(numpy.true_divide(abs(numpy.subtract(q_list[i],q_list[i+1])), step)))
            m = numpy.true_divide(numpy.subtract(q_list[i+1], q_list[i]), max_num_points)
            b = q_list[i]
	    T += ([(m*i+b) for i in range(int(max_num_points)+1)])
        return T

    def motion_callback(self, msg):
        R = tf.transformations.quaternion_matrix((msg.rotation.x,
                                                  msg.rotation.y,
						  msg.rotation.z,
						  msg.rotation.w))
	T = tf.transformations.translation_matrix((msg.translation.x, 
						   msg.translation.y, 
						   msg.translation.z))
	end_goal = numpy.dot(T,R)

###############################################################################################

        # IK provided
        #q_goal = self.IK(end_goal)

###############################################################################################

        #IK for extra credit
        self.x_desired = numpy.dot(T,R)
        root = self.robot.get_root()
        n = 0
        qc = [0] * self.num_joints
        for i in range(self.num_joints):
            #qc[i] = numpy.random.uniform(0, math.pi)
            qc[i] = numpy.random.uniform(-math.pi, math.pi)
        start_time = time.time()
        timeout = 15
        while True:
            T = tf.transformations.identity_matrix()
            self.joint_transforms = []
            self.process_link_recursive2(root, T, self.joint_state, qc)
            (delta_x,Jp) = inverse_kinematics(self.joint_transforms, self.x_current2, self.x_desired, self.joint_names, self.joint_axes)
            if time.time() - start_time > timeout:
                print("IK timeout")
                break
            if numpy.amax(numpy.abs(delta_x)) < 0.01:
                break
            delta_q = numpy.dot(Jp, delta_x)
            for i in range (0, self.num_joints):
                index = self.joint_state.name.index(self.joint_names[i])
                qc[index] += delta_q[i] 
            if n > 20:
                for i in range(self.num_joints):
                    #qc[i] = numpy.random.uniform(0, math.pi)
                    qc[i] = numpy.random.uniform(-math.pi, math.pi)
                n = 0
            n  = n + 1
        if self.is_state_valid(qc) == True:
            q_goal = qc
            for  u in range(len(q_goal)):
                while q_goal[u] > numpy.pi: q_goal[u] = q_goal[u] - 2*numpy.pi
                while q_goal[u] < -numpy.pi: q_goal[u] = q_goal[u] + 2*numpy.pi
        else:
            q_goal = None
            
##############################################################################################

        q_start = self.q_from_joint_state(self.joint_state)
        #if len(q_goal)==0:
        if q_goal==None:
	    print "IK failed"
	    return
	print "IK solved, planning"
	trajectory = self.project_plan(numpy.array(q_start), q_goal)
        if not trajectory.points:
	    print "Motion plan failed"
	else:
	    self.pub_trajectory.publish(trajectory)
        
    def joint_states_callback(self, joint_state):
        self.joint_state = joint_state
        #print(joint_state)

################################################################################################
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
########################################################################################################

if __name__ == '__main__':
    rospy.init_node('move_arm', anonymous=True)
    ma = MoveArm()
    rospy.spin()

