from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math

'''
    Linear Regression with One Variable from Scratch
     --------------------------------------------------
    Linear Regression with one variable with Least Square Error (LSE)

    Model: y = B0 + B1.x
    B1= SUM [ (xi - x_mean)* (yi - y_mean) ] / SUM[ (xi-x_mean)**2 ]
    R^2 error: 1 - [ SUM(yi - y_mean)**2 / (SUM (yi- y_pred))**2 ]
 
    B0: y_mean - B1. x_mean
    Root Mean square error RMSE = SQRT( SUM[ (yi_predict - yi)**2 ]/ (total no of points) )

'''

def mean(x):
    '''  returns mean of list x
    '''
    return sum(x)/(len(x))

def fnd_B1(x_data, y_data, x_mean, y_mean):
    num=0
    den=0
    m=len(x_data)
    for i in range(m):
        num+= ((x_data[i] - x_mean) *(y_data[i] - y_mean))
        den+= (x_data[i]- x_mean)**2
    B1= num/den
    print 'num=', num
    print 'den=', den
    return B1

def fnd_B0(y_mean, B1, x_mean):
    B0 = y_mean - B1*x_mean
    return B0

def RMSE(B0, B1, x_data, y_data):
    m= len(x_data)
    rmse=0
    for i in range(m):
        y_pred= B0 + B1* x_data[i]
        rmse+= (y_pred - y_data[i])**2
    return math.sqrt(rmse/m)

def R_square_error(x_data, y_data, x_mean, y_mean, B0, B1):
    r2=0
    t_mean_err=0
    t_pred_err=0
    for i in range(len(x_data)):
        y_pred= B0 + B1* x_data[i]
        t_mean_err += ( (y_data[i] - y_mean)**2 )
        t_pred_err += ( (y_data[i]- y_pred)**2 )
    r2= 1- (t_pred_err / t_mean_err )
    return r2
    
def MyLinearRegression(x_data, y_data):
    x_mean = mean(x_data)
    y_mean = mean(y_data)
    B1 = fnd_B1(x_data, y_data, x_mean, y_mean)
    B0 = fnd_B0(y_mean, B1, x_mean)
    return x_mean, y_mean, B1, B0

def PlotGraph(B0, B1, x_data, y_data):
    
    # plotting given data points
    plt.scatter(x_data, y_data,c= 'blue', label= 'Original_data')

    ''' finding the point upto where line should be plotted (line should cover all ponits in graph)
        mn = lowestpoint + lowestpoint/2
        mx = highestpoint + highestpoint/2
        It guarantees that line will cover all lowest & highest points
    '''
    mn= min(x_data)
    mx= max(x_data)
    
    x= np.linspace(mn-(mn/2),mx+(mx/2),1000)

    # Hypothesis line
    y= B0 + B1* x

    #  plotting regression line
    plt.plot(x,y,c='red', label='Regression line')
    plt.title('MyLinearRegression- LSE')
    plt.xlabel('x_data')
    plt.ylabel('y_data')
    plt.legend()
    plt.show()

'''
    # code for finding B0 And B1 by alternative way using correlation and covariance
'''

def fnd_mean_deviation(lst):
    '''
      returns list of num
      
    '''    
    mean_value= mean(lst)
    return [val-mean_value for val in lst ]

def dot_product(x,y):
    dp=0
    for u,v in zip(x,y):
        dp+= u*v
    return dp


def fnd_sum_of_square(lst):
    return sum([ val*val for val in lst])

def fnd_variance(lst):
    length= len(lst)
    deviation= fnd_mean_deviation(lst)
    return (fnd_sum_of_square(deviation)/length)

def stdev(x):
    return math.sqrt(fnd_variance(x))

def covariance(x_data, y_data):
    m= len(x_data)
    # take care while finding sample covariance, do (m-1) instead of m
    return dot_product( fnd_mean_deviation(x_data), fnd_mean_deviation(y_data) )/(m)  

def correlation(x_data, y_data):
    return covariance(x_data, y_data) / ( stdev(x_data) * stdev(y_data) )


def Alt_B0_B1(x_data, y_data):
    B1= ( correlation(x_data, y_data)* stdev(y_data) ) / ( stdev(x_data) )
    B0= mean(y_data)- B1*mean(x_data)
    return B1, B0
'''
def output(B1, B0):
    x=[6,7,8,9,10,11]
    y=[]
    for i in x:
        y_pred= B0 + B1* i
        y.append(y_pred)
    return y
'''    
#main
x_data=[1,2,4,3,5]
y_data=[1,3,3,2,5]
#x_data = [49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#y_data = [68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]


x_mean, y_mean, B1, B0 = MyLinearRegression(x_data, y_data)
print 'General way=', x_mean, y_mean, B1, B0
print 'RMSError=', RMSE(B0, B1, x_data, y_data)
print 'R^2 error=', R_square_error(x_data, y_data,x_mean, y_mean, B0, B1)


B1, B0 = Alt_B0_B1(x_data, y_data)
print 'Alternative way of finding B0 & B1', B1, B0
PlotGraph(B0, B1, x_data, y_data)

#print output(B1, B0)








