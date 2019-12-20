#encoding:utf-8
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as pyplot


def closed_form_1():

    dataset = pd.read_csv("./data/climate_change_1.csv")
    X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])
    
    X = np.column_stack((X,np.ones(len(X))))
   
    y = dataset.get("Temp")

    X_train = X[:284]
    X_test = X[284:]
    y_train = y[:284]
    y_test = y[284:]
    
    X_train=np.mat(X_train)
    y_train = np.mat(y_train).T
    xTx = X_train.T*X_train
    w = 0
    if np.linalg.det(xTx)==0.0:
        print("xTx 不可逆")
    else:
        print(np.linalg.det(xTx))
        w = np.ravel(xTx.I*(X_train.T*y_train))
     
    coef_=w[:-1]
    intercept_=w[-1]
      
    X_train=X_train[:,0:8]
    X_test = X_test[:,0:8]
    d = 0
    for i in range(8):
        d += coef_[i]*X_train[:,i]
    y_train_pred = d+intercept_
      
    s = 0
    for i in range(8):
        s += coef_[i]*X_test[:,i]
    y_test_pred = s+intercept_
      
      
    X_train = np.ravel(X_train).reshape(-1,8)
    y_train = np.ravel(y_train)
    y_train_pred = np.ravel(y_train_pred)
      
    print("Coefficient: ",coef_)
    print("Intercept: ",intercept_)
    print("the model is： y = ",coef_,"* X +(",intercept_,")")
      
    y_train_avg = np.average(y_train)
      
    print((np.sum((y_train-y_train_pred)**2)))
     
    print((np.sum((y_train-y_train_avg)**2)))
    R2_train = 1-(np.average((y_train-y_train_pred)**2))/(np.average((y_train-y_train_avg)**2))
    print("R2 in Train ： ",R2_train)
      
    y_test_avg = np.average(y_test)
    R2_test = 1-(np.average((y_test-y_test_pred)**2))/(np.average((y_test-y_test_avg)**2))
    print("R2 in Test ： ",R2_test)
     
    dataset = pd.read_csv("./data/climate_change_2.csv")
    X_2 = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols","NO"])
    
    y_2 = dataset.get("Temp")
    X_2 = np.column_stack((X,np.ones(len(X_2))))
   
    X_2=np.mat(X_2)
    y_2 = np.mat(y_2).T
    d=0
    for i in range(8):
        d += coef_[i]*X_2[:,i]
    y_2_pred = d+intercept_
      
    X_2 = np.ravel(X_2).reshape(-1,8)
    y_2 = np.ravel(y_2)
    y_2_pred = np.ravel(y_2_pred)
    y_2_avg = np.average(y_2)
    R2_2 = 1-(np.average((y_2-y_2_pred)**2))/(np.average((y_2-y_2_avg)**2))
    print("R2 in csv_2 ： ",R2_2)



def closed_form_2():

    dataset = pd.read_csv("./data/climate_change_1.csv")
    X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])

    y = dataset.get("Temp")
  
    X = np.column_stack((X,np.ones(len(X))))

    for lamida in [10,1,0.1,0.01,0.001]:
        X_train = X[:284]
        X_test = X[284:]
        y_train = y[:284]
        y_test = y[284:]
    
        X_train=np.mat(X_train)
        y_train = np.mat(y_train).T
        xTx = X_train.T*X_train
        w = 0
        print("="*25+"L2 Redulatization (lamida is "+str(lamida)+")"+"="*25)
        I_m= np.eye(X_train.shape[1])
        if np.linalg.det(xTx+lamida*I_m)==0.0:
            print("xTx 不可逆")
        else:
            print(np.linalg.det(xTx+lamida*I_m))
            w= (xTx+lamida*I_m).I*(X_train.T*y_train)
        wights = np.ravel(w)    
        y_train_pred = np.ravel(np.mat(X_train)*np.mat(w))
        y_test_pred = np.ravel(np.mat(X_test)*np.mat(w))
        coef_=wights[:-1]
        intercept_=wights[-1]

        X_train = np.ravel(X_train).reshape(-1,9)
        y_train = np.ravel(y_train)
        
        print("Coefficient: ",coef_)
        print("Intercept: ",intercept_)
        print("the model is： y = ",coef_,"* X +(",intercept_,")")
        y_train_avg = np.average(y_train)
    
        R2_train = 1-(np.average((y_train-y_train_pred)**2))/(np.average((y_train-y_train_avg)**2))
        print("R2 in Train ： ",R2_train)
     
        y_test_avg = np.average(y_test)
        R2_test = 1-(np.average((y_test-y_test_pred)**2))/(np.average((y_test-y_test_avg)**2))
        print("R2 in Test ： ",R2_test)

closed_form_1()
closed_form_2()
    
    
    
    
