from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
import csv

'''
    Mulitple Linear Regression with  from Scratch
     --------------------------------------------------
    Mulitple Linear Regression with one variable with Least Square Error (LSE)

    Model: y = B0 + B1.x1 + B2.x2 + .......+ Bn.xn
    This model uses matrix multiplication to calculate the B vector which is the coefficient for x's.
    
    R^2 error: 1 - [ SUM(yi - y_mean)**2 / (SUM (yi- y_pred))**2 ]
 
    
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


#Main
with open('student.csv') as f:
    math, read, write=read_data(f)
    
    m_data= math[:8]
    r_data= read[:8]
    w_data= write[:8]
    #print m_data, r_data, w_data
    '''
       A point to note here.
       m, r = noraml py array with math marks for each student and reading marks.
       But we need a numpy array of 2 features and m data sets (=m , which is actually length of numpy array)
       Initially We have,
       m_data= [m1,,m1,m3,.....,mn]
       r_data= [r1,r2,r3,......., rn]
       Our numpy array will look like , num_arr= [ [m1,m2,m3,..,mn], [r1,r2,r3,...rn] ]
       but we need                       num_arr=[ [m1,r1], [m2,r2], [m3,r3],...... n terms ]
       we are normally converting m and r to numpy and then taking transpose it . It will get the numpy array as
    '''
    x_data= np.array([m_data, r_data]).T
    y_data= np.array(w_data)
    print 'size=', y_data.shape
    
    
    x_train, y_train, x_test, y_test= get_test_train_data(x_data, y_data,0.7)
    
    print 'x_train=\n',x_train
    print 'y_train=\n',y_train
    print 'x_test=\n',x_test
    print 'y_test=\n',y_test
    print 'type x=', type(x_train)
    print 'type y=', type(y_train)
    


    reg=linear_model.LinearRegression()
    reg= reg.fit(x_train, y_train)
    print 'training complete'
    y_pred=  reg.predict(x_train)
    print 'y_pred=', y_pred
    

