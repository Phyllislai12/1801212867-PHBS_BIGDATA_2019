#encoding:utf-8
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model

# [Problem 3.1] remove features with high linearity according to VIF
# VIF variance inflation factor, which measures the linearity between features 
def vif(X, thres=1000):#The threshold is set to 1000
    col = list(range(X.shape[1]))
    dropped = True#If drop == True, then all calculated vifs are within the threshold and the traversal stops
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:,col].values, ix) for ix in range(X.iloc[:,col].shape[1])]#Calculate the VIF values for eight variables
        maxvif = max(vif)
        maxix = vif.index(maxvif)
        if maxvif > thres:
            print('delete=',X.columns[col[maxix]],'  ', 'vif=',maxvif )
            del col[maxix]
            dropped = True
    print('Remain Variables:', list(X.columns[col]))
    print('VIF:', vif)
    return list(X.columns[col]) 

'''
[Problem 3.2]The following program is a model for simple linear regression based on the above results
'''
def closed_form_1():
    #Create a general linear fit with intercept terms
    dataset = pd.read_csv("./data/climate_change_1.csv")
    X = dataset.get(['MEI', 'CO2', 'CFC-11', 'CFC-12', 'Aerosols'])#The result of the above selection
    
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
        print("xTx irreversible")
    else:
        w = np.ravel(xTx.I*(X_train.T*y_train))
     
    coef_=w[:-1]
    intercept_=w[-1]
      
    X_train=X_train[:,0:5]
    X_test = X_test[:,0:5]
    d = 0
    for i in range(5):
        d += coef_[i]*X_train[:,i]
    y_train_pred = d+intercept_
      
    s = 0
    for i in range(5):
        s += coef_[i]*X_test[:,i]
    y_test_pred = s+intercept_
      
      
    X_train = np.ravel(X_train).reshape(-1,5)
    y_train = np.ravel(y_train)
    y_train_pred = np.ravel(y_train_pred)
      
    print("Coefficient: ",coef_)
    print("Intercept: ",intercept_)
    print("the model is： y = ",coef_,"* X +(",intercept_,")")
      
    y_train_avg = np.average(y_train)
    
    R2_train = np.sum((y_train_pred-y_train_avg)**2)/(np.sum((y_train-y_train_avg)**2))
    print("R2 in Train ： ",R2_train)
      
    y_test_avg = np.average(y_test)
    R2_test = np.sum((y_test_pred-y_test_avg)**2)/(np.sum((y_test-y_test_avg)**2))
    print("R2 in Test ： ",R2_test)
    
      
   

closed_form_1()













        
            

        
    

    
