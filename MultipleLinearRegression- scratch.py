from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
import math
import csv

'''
    Mulitple Linear Regression with  from Scratch
     --------------------------------------------------
    Mulitple Linear Regression with one variable with Least Square Error (LSE)

    Model: y = B0 + B1.x1 + B2.x2 + .......+ Bn.xn
    This model uses matrix multiplication to calculate the B vector which is the coefficient for x's.
    
    R^2 score: 1 - [ SUM( (yi - y_pred)**2)  / (SUM (yi- y_mean)**2 ) ]
 
        
    Root Mean square error RMSE = SQRT( SUM[ (yi_predict - yi)**2 ]/ (total no of points) )

'''

def read_data(fobj):
    reader=csv.DictReader(fobj, delimiter=',')
    math=[]
    read=[]
    write=[]
    for line in reader:
        math.append(int(line['Math']))
        read.append(int(line['Reading']))
        write.append(int(line['Writing']))
    return math, read, write

def get_test_train_data(x_data, y_data, per):
    # validation if size of x_data  y_dta is same or not
    if len(x_data)!= len(y_data):
        print 'error in length of features & label data'
        return [[]],[[]],[[]],[[]]
    else:
        # splitting data into 'per' percentage and (100-per) percentage
        split= int(per*len(x_data))
        print 'split=', split
        x_train= x_data[0:split]
        y_train= y_data[0:split]

        x_test= x_data[split:]
        y_test= y_data[split:]

        return x_train, y_train, x_test, y_test

def cost_fun(x, y, B ):
    '''
            we will not calculate len of x as it will return 
    '''
    m= len(y)
    J = np.sum( (x.dot(B) - y) **2 )/ (2*m)
    return J

def Gradient_descent(x, y, B, alpha, itr):
    cost_values=[0] * itr
    m= len(y)
    for i in range(itr):
        
        h= x.dot(B)
        loss= h-y
        gradient= x.T.dot(loss) / m
        B= B - alpha * gradient
        cost= cost_fun(x,y,B)
        cost_values[i]=cost
        
    return B, cost_values

def RMSE(y, y_pred):
    rms_error= np.sqrt(sum( (y-y_pred)**2 )/len(y))
    return rms_error

def r2_score(y, y_pred):
    mean_y=np.mean(y)
    t_mean_err= sum( (y-mean_y)**2 )
    t_pred_err= sum( (y-y_pred)**2 )
    r2= 1-(t_pred_err/t_mean_err)
    return r2
    
    
#Main
with open('student.csv') as f:
    math_data, read_data, write_data = read_data(f)
    f.close()
    
#math_data = math_data[:8]
#read_data = read_data[:8]
#write_data = write_data[:8]

# creating np array x0 as [0,0,0,0,.... , upto no of sample data, in this case - to length of data]
x0= np.ones(len(math_data))

#print math_data
#print read_data
#print write_data
#print x0
'''
   A point to note here.
   m, r = normal py array with math marks for each student and reading marks.
   But we need a numpy array of 2 features and m data sets (=m , which is actually length of numpy array)
   Initially We have,
   m_data= [m1,,m1,m3,.....,mn]
   r_data= [r1,r2,r3,......., rn]
   Our numpy array will look like , num_arr= [ [m1,m2,m3,..,mn], [r1,r2,r3,...rn] ]
   but we need                       num_arr=[ [m1,r1], [m2,r2], [m3,r3],...... n terms ]
   we are normally converting m and r to numpy and then taking transpose it . It will get the required numpy array
'''
x_data= np.array([x0, math_data, read_data]).T
y_data= np.array(write_data)
B= np.array([0,0,0])
#print 'size=', y_data.shape


x_train, y_train, x_test, y_test = get_test_train_data(x_data, y_data,1)

#print 'x_train=\n',x_train
#print 'y_train=\n',y_train
#print 'B=\n',B
#print 'x_test=\n',x_test
#print 'y_test=\n',y_test
#print 'type x=', type(x_train)
#print 'type y=', type(y_train)

# no of ietration upto which gradient calculation should be done
itr= 1000
newB, cost_values= Gradient_descent(x_train, y_train, B, 0.0001, itr)

#predicting on train data to check others parameters like RMSE and r2_score
y_pred= x_train.dot(newB)



itr= [i for i in range(itr)]

'''
# Code to plot Cost Function Vs Iteration graph
cost_fig=plt.figure(0)
plt.plot(itr, cost_values, c='red')
plt.xlabel('No of Iteration')
plt.ylabel('Cost values')
plt.title('Cost Function Vs Iteration')
cost_fig.show()

# Code to plot scattered points in 3d axes
point_fig= plt.figure(1)
ax= Axes3D(point_fig)
ax.scatter( math_data, read_data, write_data ,c='red') 
#point_fig.show()
'''

print 'training complete'
print RMSE(y_train, y_pred)
print r2_score(y_train, y_pred)
    

