from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
  h(theta)= X.dot(theta)
'''




def sigmoid(z):
        return (1.0 / (1+np.exp(-z)) )

def CostFunction(  X, y, theta ):
        m=len(y)
        h= sigmoid(X.dot(theta))
        J= (1/m) * ( -y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h))  )
        return J

def GradientAscent( X, y, theta, alpha ):
        m,n= X.shape
        #theta= theta.reshape((n,1))
        #y=y.reshape((m,1))
        h= sigmoid(X.dot(theta))
        return ( (1/m)* (X.T.dot(h-y))  )

def LogisticRegression(X, y, theta, alpha, itr):
        m= len(y)
        cost_values = [0]*itr
        for i in range(itr):
                
                h = sigmoid(X.dot(theta))
                gradient =  GradientAscent(X, y, theta, alpha )
                theta = theta - alpha * gradient 
                cost_values[i]= CostFunction(X, y, theta )

        return theta, cost_values

def predict(X_test, Y_test, theta):
        m= len(X_test)
        y_pred=[]
        score=0
        for i in xrange(m):
                pred=round( sigmoid(X_test[i].dot(theta)) )
                if pred==Y_test[i]:
                        score+=1
        accuracy=(score/m)*100
        return accuracy
                


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
df = pd.read_csv('grade.csv')
row,col = df.shape
no_of_clases = 3
#since we have 4 features only so we will reset col with 4
print 'Data Set size=', df.size

# building unit matrix of size row,col
X_data= df[['grade1', 'grade2']]
X_data= np.array(X_data)

Y_data= df['label']
Y_data= np.array(Y_data)

##print 'X=\n', X_data.shape
##print 'y=\n', Y_data.shape
##print 'Printing Finished\n'

X_train, Y_train, X_test, Y_test = get_test_train_data(X_data, Y_data, 0.5)
print 'Test train data split complete'

init_theta= [0, 0]
alpha = 0.1
itr= 1000
print 'Initial Theta=', init_theta

new_theta, cost_values=  LogisticRegression(X_train, Y_train, init_theta, alpha, itr)
print 'New Theta=\n', new_theta
#print 'Cost function=\n', cost_values[:10]
it = [i for i in range(itr)]
plt.plot(it,cost_values )
plt.xlabel('Iteration')
plt.ylabel('cost_values')
plt.title('Iteration Vs Cost function')
plt.show()
accuracy = predict(X_test, Y_test, new_theta)
print 'Accuracy=', accuracy,'%'
