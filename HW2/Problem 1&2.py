#encoding:utf-8
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import sklearn


#[Problem 1.1 & 1.2]

def closed_form_1():
    #Create a general linear fit with intercept terms
    dataset = pd.read_csv("./data/climate_change_1.csv")
    X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])
    
    X = np.column_stack((X,np.ones(len(X))))
    z = np.linalg.matrix_rank(X)
    print("The rank of the X matrix of climate1 is：",z)
    print("The conditions for the X matrix of climate1 are：",np.linalg.cond(X))
    y = dataset.get("Temp")
    if z != X.shape[1]:
        print("climate_1 matrix is unfilled rank, so it cannot be used for linear models")
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
    R2_train = np.sum((y_train_pred-y_train_avg)**2)/(np.sum((y_train-y_train_avg)**2))
    print("R2 in Train ： ",R2_train)
      
    y_test_avg = np.average(y_test)
    R2_test = np.sum((y_test_pred-y_test_avg)**2)/(np.sum((y_test-y_test_avg)**2))
    print("R2 in Test ： ",R2_test)
     
    dataset = pd.read_csv("./data/climate_change_2.csv")
    X_2 = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols","NO"])
    X_2 = np.column_stack((X,np.ones(len(X_2))))
    
    z = np.linalg.matrix_rank(X_2)
    print("The rank of the X matrix of climate2 is：",z)
    print("The conditions for the X matrix of climate2 are：",np.linalg.cond(X_2))
    if z != X_2.shape[1]:
        print("climate_2 matrix is unfilled rank, so it cannot be used for linear models")
        
#[Problem 1.3]:According to the coefficient result, the most significant is Aerosols, others is TSI>MEI>N2O
#[Problem 1.4]The applicable conditions of ordinary linear regression
#1. The number of samples should be greater than the number of features
#2. The determinant of XT*X is not equal to 0
#3. X is a non-singular matrix

#[Problem2.1]
#Loss function for linear model with L1 regularization:JR(w)=0.5*|y-Xw|**2 +λ∑|wi| 
#Loss function for linear model with L2 regularization:JR(w)=0.5*|y-Xw|**2 +0.5* λ|w|**2  

#[Problem2.2&2.4]
def closed_form_2():
    #Create L2 regular linear fitting with intercept term, i.e., ridge regression
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
            print("xTx irreversible")
        else:
           
            w= (xTx+lamida*I_m).I*(X_train.T*y_train)

        wights = np.ravel(w)    
        y_train_pred = np.ravel(np.mat(X_train)*np.mat(w))
        y_test_pred = np.ravel(np.mat(X_test)*np.mat(w))
        coef_=wights[:-1]
        intercept_=wights[-1]
        
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
      
   
        y_train_avg = np.average(y_train)
        R2_train = np.sum((y_train_pred-y_train_avg)**2)/(np.sum((y_train-y_train_avg)**2))
        print("R2 in Train ： ",R2_train)
      
        y_test_avg = np.average(y_test)
        R2_test = np.sum((y_test_pred-y_test_avg)**2)/(np.sum((y_test-y_test_avg)**2))
        print("R2 in Test ： ",R2_test)

        print("Coefficient: ",coef_)
        print("Intercept: ",intercept_)
        #The following is the formula of linear fitting:
        print("the model is： y = ",coef_,"* X +(",intercept_,")")
    
#Cross-validation selects parameters
def build_model(x,y):
    kfold = KFold(n_splits=5).split(x, y)
    model = Ridge(normalize=True)
    lamibda_range = [10,1,0.1,0.01,0.001]
    grid_param = {"alpha":lamibda_range}
    
    grid = GridSearchCV(estimator=model,param_grid=grid_param,cv=kfold,scoring="r2")
    grid.fit(x,y)
    print(grid.best_params_)
    return grid.best_params_


closed_form_1()
closed_form_2()

dataset = pd.read_csv("./data/climate_change_1.csv")
X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])
y = dataset.get("Temp")
  
build_model(X,y)
    
    
# [Problem2.3]
# Regularization eliminates collinearity between features by increasing penalty functions. It can be understood as adding an L2 regular term to the linear regression loss function to limit the theta. By determining the value of lamida
# can balance the model between bias and variance. Adding lamibda* identity matrix to the xTx matrix can make the determinant of the matrix whose determinant is close to 0 not equal to 0.
# is judged by r2, and fits best when lambda is 10.
# But it doesn't make sense to simply judge this, so take the cross validation approach, such as build_model
# Because in the actual training, the fitting degree of the training results to the training set is usually good (the initial condition is sensitive), but the fitting degree to the data outside the training set is usually not so satisfactory. So we usually don't
# will not use all data sets for training, but will separate a part (this part does not participate in training) to test the parameters generated by the training set, and relatively objectively judge the consistency of these parameters to the data outside the training set.
# This idea is called Cross Validation.
# As the title suggests, specifying the training set and test machine can skew the resulting model.
    
