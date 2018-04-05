from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
'''
  h(theta)= X.dot(theta)
'''



def get_test_train_data(x_data, y_data, per):
    # validation if size of x_data  y_dta is same or not
    if len(y_data) != len(x_data):
        print 'error in length of features & label data'
        return [[]],[[]],[[]],[[]]
    else:
        # splitting data into 'per' percentage and (100-per) percentage
        split= int(per*len(x_data))

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
print 'Data Set size=', df.shape

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


clf = linear_model.LogisticRegression()
clf.fit(X_train, Y_train)

print 'Model trained'
pred= clf.predict(np.array(Y_test).reshape(-1,2))
#print 'Predicted Values=', pred
print 'Accuracy=', clf.score(X_test, Y_test) *100, '%'
