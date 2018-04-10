from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math
plt.style.use('ggplot')

np.random.seed(10)
'''

K-means clustering algorithm from scratch where we will be given a dataset having x and cordinate .
Data file : kmean_data.csv
Features:
1. x is  x cordinate
2. y is y cordinate


'''


def read_data(file_name):
        df= pd.read_csv(file_name)
        x = df['x']
        y = df['y']

        return x,y


def plot_point_before_clustering(x_data, y_data, C_x, C_y):
        
        #plotting points                
        before= plt.figure("Before Clustering")
        plt.scatter( x_data, y_data, s= 10,  c='red', label = "Unclustered Points" )

        #plotting Initial cluster centroids
        plt.scatter( C_x, C_y, s= 100,  c='black', marker= "*" , label="Initial cluster Centre" )
        plt.title('Graph before Clustering')
        plt.legend(loc='upper right')
        plt.show()

def plot_point_after_clustering( point, Centroids_cord , Clusters, k):
        col=['r', 'g', 'b' ]
        after = plt.figure("After Clustering")
        for i in xrange(k):
                pts=[ point[j] for j in xrange(len(point)) if Clusters[j]==i]
                plt.scatter( [p[0] for p in pts] , [p[1] for p in pts] , s= 10,  c=col[i] )
                
        plt.scatter([p[0] for p in Centroids_cord], [p[1] for p in Centroids_cord], s= 100,  c='black', marker= "*" , label="Cluster Centre")
        plt.title('Grapg after Clustering')
        plt.legend(loc='upper right')
        plt.show()

def plot_error(error_arr):
        err= plt.figure("Iteration Vs Error Graph")
        plt.plot( [ i+1 for i in range(len(error_arr))], error_arr, c='magenta' )
        plt.xlabel("Iterations")
        plt.ylabel("Error value")
        plt.title("Iteration Vs Error Graph")
        plt.show()
        
        

def euclidean_distance(p1, p2):
        '''
          Generic
          Here p1 and p2 wil be a list
          Ex: p1=[1,2,1,.....]
              p2=[3,4,5,.....]
              
          We don't know whether points x1 and x2 are 2D or 3D points or high dimensional points
          So we will iterate over the whole length and calculate the euclidean distance.
          
        '''

        dist=0
        for i in range(len(p1)):
                dist = dist+  ( (p1[i] - p2[i])**2 )
        return math.sqrt(dist)
          
def centroid_dist(Centroids_cord_old, Centroids_cord ):
        
        k= len(Centroids_cord)
        dist=0
        for i in xrange(k):
                dist=dist +  euclidean_distance( Centroids_cord_old[i], Centroids_cord[i] )
        return dist
        
def get_cluster_id(dist):
        idx= sorted(dist, key= lambda x: x[0])[0][1]
##        mn= dist[0]
##        idx= dist[1]
##        for i in xrange(1:len(dist)):
##                if dist[i][0]< mn:
##                        mn= dist[i][0]
##                        idx= dist[i][1]
        return idx

def get_new_cord(k, Clusters, point):
        # here k is cluster id for which we are finding mean centroid
        cordinates=[]
        for i in xrange(len(Clusters)):
                if Clusters[i]==k:
                        cordinates.append(point[i])
        cord=[]
        n=len(cordinates)
        for i in xrange(len(cordinates[0])):
                cord.append( sum ( x[i] for x in cordinates )/n )
        return cord
        
def K_means_clustering(point, k):
        # Taking Initital k Centroids randomly
        C_x= np.random.randint(min(x_data)+20, max(x_data)-20, size= k)
        C_y= np.random.randint(min(x_data)+20, max(x_data)-20, size= k)

        
        #centroids_cord will contains co-ordinates of all k centriods
        # Ex- [ [1,2], [4,5], [12,23] ..... k items]
        Centroids_cord_old=[ [0,0] for i in xrange(k)]
        Centroids_cord = [ [u,v] for u,v in zip(C_x, C_y) ]

        error_arr=[]

        # this will contain cluster id  for each point in point array
        # example Clusters = [ 1, 0, 2, 1, 0 ]
        # means point 0 belogs to cluster 1, point 1 belogs to cluster 0, etc.
        
        Clusters= [ 0 for i in xrange(len(point))]
        
        error = centroid_dist( Centroids_cord_old, Centroids_cord )
        error_arr.append(error)
        
        counter=0
        while error!=0:
                counter+=1
                
                for i in xrange(len(point)):
                        dist=[]
                        for j in xrange(k):
                                # keeping euclidean_dist and cluster id
                                d= [euclidean_distance( point[i], Centroids_cord[j] ), j]
                                dist.append(d)
                        idx = get_cluster_id(dist)
                        Clusters[i]= idx

                # saving new centroid to old centroid
                for i in xrange(k):
                        Centroids_cord_old[i] = Centroids_cord[i]
                
                # finding new clusters centroid
                for i in xrange(k):
                        Centroids_cord[i]= get_new_cord(i, Clusters, point)
                

                error = centroid_dist( Centroids_cord_old, Centroids_cord )
                error_arr.append(error)
                
        print 'Kmeans finished'
        print 'error array=',error_arr
        plot_point_before_clustering(x_data, y_data, C_x, C_y)
        plot_point_after_clustering( point, Centroids_cord, Clusters, k )
        plot_error(error_arr)
        
#main
x_data, y_data = read_data('Kmean_data.csv')

point = [ [u,v] for u,v in zip(x_data, y_data) ]

K_means_clustering(point, 3)






        


