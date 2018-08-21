#!/usr/bin/env python
## HW2

import numpy
from urdf_parser_py.urdf import URDF
from sensor_msgs.msg import JointState
import rospy
import geometry_msgs.msg
import tf
import tf.msg

def convert_to_message(T, slave, master):
    t = geometry_msgs.msg.TransformStamped()
    t.header.frame_id = master
    t.child_frame_id = slave
    t.header.stamp = rospy.Time.now()
    translation = tf.transformations.translation_from_matrix(T)
    q = tf.transformations.quaternion_from_matrix(T)
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]        
    return t

class FK():
    def __init__(self):
        self.pub_tf = rospy.Publisher("/tf", tf.msg.tfMessage, queue_size=1)  
        self.robot = URDF.from_parameter_server()
        rospy.Subscriber("joint_states", JointState, self.callback)

    def callback(self, joint_values):
         answer = tf.msg.tfMessage()
         
         link_name = self.robot.get_root()
         link_names = []
         link_names.append(link_name)
         joints = []

         while True:
            
            if link_name not in self.robot.child_map:
                break

            (joint_name, next_link_name) = self.robot.child_map[link_name][0]
            joints.append(self.robot.joint_map[joint_name])
            link_names.append(next_link_name)
            link_name = next_link_name

            n = len(joints)

            for i in range(0, n):
                O = tf.transformations.translation_matrix(joints[i].origin.xyz)
                 
                j = 0
                for k in joint_values.name:
                    if k == joint_name:
                        pass
                    else:
                        j = j+1

                if joints[i].type == 'revolute':
                    R = tf.transformations.quaternion_matrix(
                        tf.transformations.quaternion_about_axis(
                            joint_values.position[j-1], joints[i].axis))
                else :  
                    R = tf.transformations.identity_matrix()     

                T = tf.transformations.concatenate_matrices(O,R)

                answer.transforms.append(
                    convert_to_message(T, link_names[i+1], link_names[i]))

            self.pub_tf.publish(answer)



if __name__ == '__main__':
    rospy.init_node('fk', anonymous=True)
    fk = FK()
    rospy.spin()


