#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 22:56:03 2020

@author: Ethan.Qu
"""


import numpy as np
import matplotlib.pyplot as plt


#返回值为c1,c2,c3... & label1, label2, label3....
#First Group 
x1 = np.random.normal(1,12,10000)
y1 = np.random.normal(1,12,10000)

x1 = x1.reshape(1,-1)
y1 = y1.reshape(1,-1)

c1_1 = np.concatenate((x1,y1))

c1_1 = c1_1[:,c1_1[0]>1]

c1_1 = c1_1[:,c1_1[1]>1]


x2 = np.random.normal(48,12,10000)
y2 = np.random.normal(48,12,10000)

x2 = x2.reshape(1,-1)
y2 = y2.reshape(1,-1)

c1_2 = np.concatenate((x2,y2))

c1_2 = c1_2[:,c1_2[0]<48]

c1_2 = c1_2[:,c1_2[1]<48]

# plt.xlim(xmax = 50, xmin = 0)
# plt.ylim(ymax = 50, ymin = 0)

# plt.scatter(c1_1[0],c1_1[1],s = np.pi/3, c = '#DC143C')

# plt.scatter(c1_2[0],c1_2[1],s = np.pi/3, c = '#00CED1')

# plt.show()

c1 = np.hstack((c1_1,c1_2))

label_1 = -np.ones(len(c1_1[0]))
label_2 = np.ones(len(c1_2[0]))

label_1 = label_1.reshape(1,-1)
label_2 = label_2.reshape(1,-1)

label1  = np.hstack((label_1,label_2))


#Second Group
x1 = np.random.normal(20,2,2500)
y1 = np.random.normal(30,2,2500)

x1 = x1.reshape(1,-1)
y1 = y1.reshape(1,-1)

c2_1 = np.concatenate((x1,y1))


x2 = np.random.normal(33,5.5,2500)
y2 = np.random.normal(12,5.5,2500)

x2 = x2.reshape(1,-1)
y2 = y2.reshape(1,-1)

c2_2 = np.concatenate((x2,y2))


# plt.xlim(xmax = 50, xmin = 0)
# plt.ylim(ymax = 50, ymin = 0)

# plt.scatter(c2_1[0],c2_1[1],s = np.pi/3, c = '#DC143C')

# plt.scatter(c2_2[0],c2_2[1],s = np.pi/3, c = '#00CED1')

# plt.show()

c2 = np.hstack((c2_1,c2_2))

label_1 = -np.ones(len(c2_1[0]))
label_2 = np.ones(len(c2_2[0]))

label_1 = label_1.reshape(1,-1)
label_2 = label_2.reshape(1,-1)

label2  = np.hstack((label_1,label_2))

#Third Group

x1 = np.random.normal(21.5,2,5000)
y1 = np.random.normal(25,7,5000)

x1 = x1.reshape(1,-1)
y1 = y1.reshape(1,-1)

c3_1 = np.concatenate((x1,y1))

c3_1 = c3_1[:,c3_1[0]<=21.5]


x2 = np.random.normal(23.5,2,5000)
y2 = np.random.normal(25,7,5000)

x2 = x2.reshape(1,-1)
y2 = y2.reshape(1,-1)

c3_2 = np.concatenate((x2,y2))

c3_2 = c3_2[:,c3_2[0]>=23.5]



# plt.xlim(xmax = 50, xmin = 0)
# plt.ylim(ymax = 50, ymin = 0)

# plt.scatter(c3_1[0],c3_1[1],s = np.pi/3, c = '#DC143C')
#
# plt.scatter(c3_2[0],c3_2[1],s = np.pi/3, c = '#00CED1')

# plt.show()

c3 = np.hstack((c3_1,c3_2))

label_1 = -np.ones(len(c3_1[0]))
label_2 = np.ones(len(c3_2[0]))

label_1 = label_1.reshape(1,-1)
label_2 = label_2.reshape(1,-1)

label3  = np.hstack((label_1,label_2))

#Fourth Group

x1 = np.random.normal(18,4.5,20000)
y1 = np.random.normal(28,2.2,20000)

x1 = x1.reshape(1,-1)
y1 = y1.reshape(1,-1)

c4_1 = np.concatenate((x1,y1))


x2 = np.random.normal(32,4.5,20000)
y2 = np.random.normal(22,2.2,20000)

x2 = x2.reshape(1,-1)
y2 = y2.reshape(1,-1)

c4_2 = np.concatenate((x2,y2))



# plt.xlim(xmax = 50, xmin = 0)
# plt.ylim(ymax = 50, ymin = 0)
#
# plt.scatter(c4_1[0],c4_1[1],s = np.pi/3, c = '#DC143C')
#
# plt.scatter(c4_2[0],c4_2[1],s = np.pi/3, c = '#00CED1')

# plt.show()
c4 = np.hstack((c4_1,c4_2))
# print(np.shape(c4) )

label_1 = -np.ones(len(c4_1[0]))
label_2 = np.ones(len(c4_2[0]))

label_1 = label_1.reshape(1,-1)
label_2 = label_2.reshape(1,-1)

label4  = np.hstack((label_1,label_2))
# print(np.shape(label4))







