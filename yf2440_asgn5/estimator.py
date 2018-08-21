#!/usr/bin/env python

import rospy
import math
import numpy
import sys
import random

from state_estimator.msg import SensorData
from geometry_msgs.msg import Pose2D

class now(object):
    pass

class Estimator(object):

    def __init__(self):

        self.timestep = 0.01
        self.covariance = 0.1
        self.lx = []
        self.ly = []
        self.lr = []
        self.lb = []

        rospy.Subscriber("/sensor_data", SensorData, self.callback)

        self.pub = rospy.Publisher("/robot_pose_estimate", Pose2D, queue_size=1)
        self.pub2 = rospy.Publisher("/robot_pose_estimate_2", Pose2D, queue_size=1)

    def callback(self, msg):

        N = len(msg.readings)
        V = msg.vel_trans
        W = msg.vel_ang

        self.lx = []
        self.ly = []
        self.lr = []
        self.lb = []
        for i in range(N):
            self.lx.append(msg.readings[i].landmark.x)
            self.ly.append(msg.readings[i].landmark.y)
            self.lb.append(msg.readings[i].bearing)
            self.lr.append(msg.readings[i].range)

#######################
#Extended Kalman Filter

        angle=true.xyt[2]
        F=numpy.zeros((3,3))
        F[0,0] =1
        F[0,2] =-self.timestep*V*numpy.sin(angle)
        F[1,1] =1
        F[1,2] =self.timestep*V*numpy.cos(angle)
        F[2,2] =1

        xpp=self.timestep*numpy.array([V*numpy.cos(angle),V*numpy.sin(angle),W])
        xp=numpy.array(true.xyt)+ xpp

        Fp=numpy.dot(F,true.P)
        Ppp=numpy.dot(Fp, numpy.transpose(F))
        Pp=Ppp+self.covariance*numpy.identity(3)

        if N > 0:
            y=numpy.empty(2*N)
            H=numpy.empty((2*N,3))
            Hx=numpy.empty(2*N)

            for i in range(N):
                k=2*i
                y[k]=self.lr[i]
                y[k+1]=self.lb[i]
                H[k][0]=(xp[0]-self.lx[i])/numpy.sqrt((xp[0]-self.lx[i])**2+(xp[1]-self.ly[i])**2)
                H[k][1]=(xp[1]-self.ly[i])/numpy.sqrt((xp[0]-self.lx[i])**2+(xp[1]-self.ly[i])**2)
                H[k][2]=0
                H[k+1][0]=-(xp[1]-self.ly[i])/((xp[0]-self.lx[i])**2+(xp[1]-self.ly[i])**2)
                H[k+1][1]=(xp[0]-self.lx[i])/((xp[0]-self.lx[i])**2+(xp[1]-self.ly[i])**2)
                H[k+1][2]=-1
                Hx[k]=numpy.sqrt((xp[0]-self.lx[i])**2+(xp[1]-self.ly[i])**2)
                Hx[k+1]=math.atan2((self.ly[i]-xp[1]),(self.lx[i]-xp[0]))- xp[2]

            mu=y-Hx
            Sp=numpy.dot(H, Pp)
            Spp=numpy.dot(Sp, numpy.transpose(H))
            S=Spp+self.covariance*numpy.identity(2*N)
            Rp=numpy.dot(Pp, numpy.transpose(H))
            R=numpy.dot(Rp, numpy.linalg.inv(S))
            xp=xp+numpy.dot(R, mu)
            Pp=Pp-numpy.dot(numpy.dot(R,H), Pp)

        true.xyt=list(xp)
        true.P=Pp
      
        For_publish=Pose2D()
        For_publish.x=xp[0]
        For_publish.y=xp[1]
        For_publish.theta=xp[2]

        self.pub.publish(For_publish)

################
#Particle Filter

        Wp=numpy.random.normal(0,0.1,(120,3))
        Vp=numpy.random.normal(0,0.1,(120,3))
        Vp[:,0]=Vp[:,0]+V
        Vp[:,1]=Vp[:,1]+V
        Vp[:,2]=Vp[:,2]+W

        for i in range(120):
            Vp[i,0]=Vp[i,0]*numpy.cos(true2.xyt[i,2])
            Vp[i,1]=Vp[i,1]*numpy.sin(true2.xyt[i,2])

        Xp=self.timestep*Vp+true2.xyt+Wp

        if N > 0:
            weight=numpy.ones(120)
            for i in range(N):
                for j in range(120):
                    sqra=Xp[j][0]-self.lx[i]
                    sqrb=Xp[j][1]-self.ly[i]
                    sqr=numpy.sqrt((sqra)**2+(sqrb)**2)
                    weight[j]=weight[j]*numpy.abs(sqr-self.lr[i])
                    atana=self.ly[i]-Xp[j][1]
                    atanb=self.lx[i]-Xp[j][0]
                    atan=math.atan2(atana,atanb)
                    value = atan-Xp[j][2]-self.lb[i]
                    while value > numpy.pi: 
                        value=value-2*numpy.pi
                    while value < -numpy.pi: 
                        value=value+2*numpy.pi
                    weight[j]=weight[j]*numpy.abs(value)

            weight=1/weight
            lottaryS=sum(weight)
            for i in range(120):
                lottaryV=random.uniform(0,lottaryS)
                choice=0
                for j in range(120):
                    choice=choice+weight[j]
                    if choice > lottaryV:
                        true2.xyt[i]=Xp[j]
                        break
        else: 
             true2.xyt = Xp

        For_publish=Pose2D()
        For_publish.x=sum(true2.xyt[:,0])/120
        For_publish.y=sum(true2.xyt[:,1])/120
        For_publish.theta=sum(true2.xyt[:,2])/120

        self.pub2.publish(For_publish)

if __name__ == '__main__':
    true = now()
    true.xyt = [0,0,0]
    true.P = numpy.zeros((3,3))
    true2 = now()
    true2.xyt=numpy.zeros((120,3))
    rospy.init_node('estimate', anonymous=True)
    estimator = Estimator()
    rospy.spin()


        
