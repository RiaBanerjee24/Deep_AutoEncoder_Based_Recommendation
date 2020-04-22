import pandas as pd
import numpy as np
import csv

def createMatrix(path):
    df = pd.read_csv(path,sep='\t',names=['user_id','item_id','rating','timestamp'])
    matrix = []

    #user = count = df.user_id.unique()
    for uid in range(1,944): #will run for 943 times
        i_id = df.iloc[:,1][df.iloc[:,0] == uid]
        u_rating = df.iloc[:,2][df.iloc[:,0]==uid]
        row = np.zeros(1682)
        row[i_id-1] = u_rating
        matrix.append(row)
    #print(type(matrix)) # <class 'list'>
    #print(type(matrix[0])) # <class 'numpy.ndarray'>

    # print(len(matrix)) #943 (users)
    # print(len(matrix[0])) #1682 (movies)

    return matrix

def setPath():
    train_matrix = createMatrix(path='ml-100k/u1.base')
    test_matrix = createMatrix(path='ml-100k/u1.test')
    df1 = pd.DataFrame(train_matrix)
    df1.to_csv('train_matrix',index=False,header=False)
    df1 = pd.DataFrame(test_matrix)
    df1.to_csv('test_matrix', index=False, header=False)

setPath()