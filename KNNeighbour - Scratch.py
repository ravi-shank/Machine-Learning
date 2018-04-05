from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math
plt.style.use('ggplot')
'''

KNN Classification algorithm from scratch where we will be given a dataset of different class and
some few data set will be given which we have to classify based.
Data file : Point.csv
Features:
1. x is  x cordinate
2. y is y cordinate
3. cls is class to which the point belongs to


'''


def read_train_data(file_name):
        df= pd.read_csv(file_name)
        x = df['x']
        y = df['y']
        cls = df['cls']
        #print df.head(5)
        return x,y,cls

def read_test_data(file_name):
        df= pd.read_csv(file_name)
        x = df['x']
        y = df['y']
        cls = df['cls']
        #print df.head(5)        
        return x,y,cls

def plot_point(x_train_data, y_train_data, cls_train_data, x_test_data, y_test_data, predict_cls):
        
        x0=[]
        y0=[]
        x1=[]
        y1=[]
        for i in xrange(len(x_train_data)):
                if cls_train_data[i]==0:
                        x0.append(x_train_data[i])
                        y0.append(y_train_data[i])
                else:
                        x1.append(x_train_data[i])
                        y1.append(y_train_data[i])
                        
        before= plt.figure("Before classfication")
        plt.scatter(x0, y0, s= 30,  c='red', label = "Train class 0" )
        plt.scatter(x1, y1, s= 30,  c='black', label = "Train class 1")
        plt.scatter([x for x in x_test_data ], [y for y in y_test_data ], s= 30,  c='green', label = "To predict")
        plt.legend(loc='lower right')
        plt.title('Graph before Classification')

        # code to plot after classifcation
        after= plt.figure("After classfication")
        x2=[]
        y2=[]
        x3=[]
        y3=[]
        for i in xrange(len(x_test_data)):
                if predict_cls[i]==0:
                        x2.append(x_test_data[i])
                        y2.append(y_test_data[i])
                else:
                        x3.append(x_test_data[i])
                        y3.append(y_test_data[i])
                        
        plt.scatter(x0, y0, s= 30,  c='red', label = "Train class 0" )
        plt.scatter(x1, y1, s= 30,  c='black', label = "Train class 1") 
        plt.scatter(x2, y2, s= 30,  c='red', marker= "x", label = "Pred class 0")
        plt.scatter(x3, y3, s= 30,  c='black', marker= "*", label = "Pred class 1")
        plt.legend(loc='lower right')
        plt.title('Graph After Classification')
        plt.show() 

def euclidean_distance(x1, x2):
        '''
          Here x1 and x2 wil be a list
          Ex: x1=[1,2,1,.....]
              x2=[3,4,5,.....]
          We don't know whether points x1 and x2 are 2D or 3D points or high dimensional points
          So we will iterate over the whole length and calculate the euclidean distance.
          
        '''
        
        dist=0
        for i in range(len(x1)):
                dist = dist+  ( (x1[i] - x2[i])**2 )
        return math.sqrt(dist)
          


def common_class(dist):
        dic={}
        for lst in dist:
                try:
                        dic[lst[1]]+=1
                except KeyError:
                        dic[lst[1]]=1
                        
        cls= sorted(dic.items(),key= lambda x: x[1],  reverse=True)
        cls = cls[0][0]
        
        return cls

        
        
def K_nearest_neighbour(train_point, test_point, cls, k=3):
        '''
         dist list will be list of list in the form [distance_i,cls]
         dist= [ [2.34, 0], [12.3, 0], [34.34, 1],....... ]
        '''
        dist=[]
        for i,train_point in enumerate(x_train):
                euc_dis = euclidean_distance(train_point, test_point)
                dist.append([euc_dis, cls[i]])
        
        
        # finding k nearest point by sotring the euclidean distance
        dist= sorted(dist)[:k]
        pred_cls= common_class(dist)
        
        return pred_cls
                
        
        
        
        
def predict(x_train, x_test, cls):

        '''
        We will take eack point from x_test and find the predicted class 
        '''
        predict_cls=[]
        for i, point in enumerate(x_test):
                pred_class= K_nearest_neighbour(x_train, point, cls)
                predict_cls.append(pred_class)
                
        return predict_cls
        
def get_accuracy(pred,cls):
        count =0
        for i in range(len(pred)):
                if pred[i]==cls[i]:
                        count+=1
        return count/len(pred)
                
        

x_train_data, y_train_data, cls_train_data = read_train_data('KNN_train.csv')
x_test_data, y_test_data , cls_test_data = read_test_data('KNN_test.csv')



#Plotting training point on graph
#plot_point_classification(x_data, y_data, cls_data, x_test, y_test)

x_train = [[u,v] for u,v in zip(x_train_data, y_train_data)]
x_test  = [[u,v] for u,v in zip(x_test_data, y_test_data)]

predict_cls = predict(x_train, x_test, cls_train_data)

plot_point(x_train_data, y_train_data, cls_train_data, x_test_data, y_test_data, predict_cls)

accuracy= get_accuracy(predict_cls, cls_test_data)
print 'Accuracy= ', accuracy*100,'%'
#print 'Predicted class='
#for c in predict_cls:
#        print c,





        


