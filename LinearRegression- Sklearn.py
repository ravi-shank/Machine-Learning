from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error



#main
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

        x_test= y_data[split:]
        y_test= y_data[split:]

        return x_train, y_train, x_test, y_test
    


x_data=[1,2,4,3,5,6,7,8,9,10,11]
y_data=[1,3,3,2,5,7,6,8,9,12,13]

x_train, y_train, x_test, y_test = get_test_train_data(x_data, y_data, 0.7)

x_train = [49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
y_train = [68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]

x_train = [1,2,4,3,5]
y_train = [1,3,3,2,5]

x_test =[6,7,8,9,10]

'''
print 'x_train\n', x_train
print 'y_train\n', y_train
print 'x_test\n', x_test
print 'y_test\n', y_test
'''
x_train= np.array(x_train).reshape(-1,1)
y_train= np.array(y_train).reshape(-1,1)
x_test =np.array(x_test).reshape(-1,1)

#print type(x_train)
#print 'shape=',x_train.shape

#print x_train, len(x_train)

reg= linear_model.LinearRegression()
reg= reg.fit(x_train, y_train)
print 'Training Complete'

#for calculation of rmse and r^2 error , we are predicting on training data
y_pred= reg.predict(x_train)

print 'RMSE=', np.sqrt( mean_squared_error(y_train, y_pred) )

print 'r^2 error=', reg.score(x_train, y_train)

plt.scatter(x_train, y_train,c= 'blue', label= 'Original_data')
plt.plot(x_train, y_pred, c='red' , label= 'regression line')
plt.title('MyLinearRegression- LSE')
plt.xlabel('x_data')
plt.ylabel('y_data')
plt.legend()
plt.show()
