#encoding:utf-8
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as pyplot


def closed_form_1():
    dataset = pd.read_csv("./data/climate_change_1.csv")
    X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])
    
    #X = np.column_stack((X,np.ones(len(X))))
    y = dataset.get("Temp")
#     pyplot.scatter(X["Aerosols"],y)
#     pyplot.show()
    X_train = X[:284]
    X_test = X[284:]
    y_train = y[:284]
    y_test = y[284:]
    from sklearn.feature_selection import SelectFromModel
    regr = linear_model.LinearRegression()
#     selectFromModel = SelectFromModel(regr,threshold = 0.005)
#     selectFromModel.fit(X_train,y_train)
#     #a = selectFromModel.score(X_test, y_test)
#     print(X.columns[selectFromModel.get_support()])
    regr.fit(X_train,y_train)
    print('coefficients(b1,b2...):',regr.coef_)
    print('intercept(b0):',regr.intercept_)
    y_train_pred = regr.predict(X_train)
       
    R2_1 = regr.score(X_train, y_train)
    print(R2_1)
     
       
    R2_2 = regr.score(X_test, y_test)
    print(R2_2)
    
#     X_train=np.mat(X_train)
#     y_train = np.mat(y_train).T
#     xTx = X_train.T*X_train
#     w = 0
#     if np.linalg.det(xTx)==0.0:
#         print("xTx 不可逆")
#     else:
#         w = np.ravel(xTx.I*(X_train.T*y_train))
#     
#     coef_=w[:-1]
#     intercept_=w[-1]
#      
#     X_train=X_train[:,0:8]
#     X_test = X_test[:,0:8]
#     d = 0
#     for i in range(8):
#         d += coef_[i]*X_train[:,i]
#     y_train_pred = d+intercept_
#      
#     s = 0
#     for i in range(8):
#         s += coef_[i]*X_test[:,i]
#     y_test_pred = s+intercept_
#      
#      
#     X_train = np.ravel(X_train).reshape(-1,8)
#     y_train = np.ravel(y_train)
#     y_train_pred = np.ravel(y_train_pred)
#      
#     print("Coefficient: ",coef_)
#     print("Intercept: ",intercept_)
#     print("the model is： y = ",coef_,"* X +(",intercept_,")")
#      
#     y_train_avg = np.average(y_train)
#      
#     print((np.sum((y_train-y_train_pred)**2)))
#     
#     print((np.sum((y_train-y_train_avg)**2)))
#     R2_train = 1-(np.average((y_train-y_train_pred)**2))/(np.average((y_train-y_train_avg)**2))
#     print("R2 in Train ： ",R2_train)
#      
#     y_test_avg = np.average(y_test)
#     R2_test = 1-(np.average((y_test-y_test_pred)**2))/(np.average((y_test-y_test_avg)**2))
#     print("R2 in Test ： ",R2_test)
#     
    dataset = pd.read_csv("./data/climate_change_2.csv")
    X_2 = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols","NO"])
    #X_2 = np.column_stack((X_2,np.ones(len(X_2))))
    y_2 = dataset.get("Temp")
    regr_2 = linear_model.LinearRegression()
    regr_2.fit(X_2, y_2)
    print(regr_2.coef_)
    print(regr_2.intercept_)
    
#     X_2 = np.column_stack((X,np.ones(len(X_2))))

#     y_2 = dataset.get("Temp")
#      
#     X_2=np.mat(X_2)
#     y_2 = np.mat(y_2).T
#     d=0
#     for i in range(8):
#         d += coef_[i]*X_2[:,i]
#     y_2_pred = d+intercept_
#      
#     X_2 = np.ravel(X_2).reshape(-1,8)
#     y_2 = np.ravel(y_2)
#     y_2_pred = np.ravel(y_2_pred)
#     y_2_avg = np.average(y_2)
#     R2_2 = 1-(np.average((y_2-y_2_pred)**2))/(np.average((y_2-y_2_avg)**2))
#     print("R2 in csv_2 ： ",R2_2)

closed_form_1()
# the most significant is Aerosols, others is TSI>MEI>N2O

def closed_form_2():
    dataset = pd.read_csv("./data/climate_change_1.csv")
    X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])

    y = dataset.get("Temp")
    #X = np.column_stack((X,np.ones(len(X))))
    X_train = X[:284]
    X_test = X[284:]
    y_train = y[:284]
    y_test = y[284:]
    
    lamida = 0.1
    print("="*25+"L2 Redulatization (lamida is "+str(lamida)+")"+"="*25)
    clf=linear_model.Ridge(alpha=lamida,max_iter = 20000)
    clf.fit(X_train,y_train)
    y_train_pred = clf.predict(X_train)
    R2_1 = clf.score(X_train, y_train)
    print('coefficients(b1,b2...):',clf.coef_)
    print('intercept(b0):',clf.intercept_)
    print("lamida = ",lamida,", train set R2 = ",R2_1)
    R1_1 = clf.score(X_test,y_test)
    print("lamida = ",lamida,", test set R2 = ",R1_1)

    print("="*25+"start change L2 Redulatization (lamida is 10,1,0.1,0.01,0.001)"+"="*25)
    clf = linear_model.RidgeCV(alphas = [10,1,0.1,0.01,0.001])
    clf.fit(X_train,y_train)
    print('coefficients(b1,b2...):',clf.coef_)
    print('intercept(b0):',clf.intercept_)
    R2_1 = clf.score(X_train, y_train)
    print("lamida = ",clf.alpha_, ", train set R2 = ",R2_1)
    
    R2_1 = clf.score(X_test, y_test)
    print("lamida = ",clf.alpha_, ", test set R2 = ",R2_1)
    
    
#     X_train=np.mat(X_train)
#     y_train = np.mat(y_train).T
#     xTx = X_train.T*X_train
#     w = 0
#     if np.linalg.det(xTx)==0.0:
#         print("xTx 不可逆")
#     else:
#         I_m= np.eye(9, 9, k=0, dtype= float)
#         w = np.ravel((xTx+lamida*I_m).I*(X_train.T*y_train))
#         
#     coef_=w[:-1]
#     intercept_=w[-1]
#     
#     X_train=X_train[:,0:8]
#     X_test = X_test[:,0:8]
#     d = 0
#     for i in range(8):
#         d += coef_[i]*X_train[:,i]
#     y_train_pred = d+intercept_
#     
#     s = 0
#     for i in range(8):
#         s += coef_[i]*X_test[:,i]
#     y_test_pred = s+intercept_
#     
#     
#     X_train = np.ravel(X_train).reshape(-1,8)
#     y_train = np.ravel(y_train)
#     y_train_pred = np.ravel(y_train_pred)
#     
#     print("Coefficient: ",coef_)
#     print("Intercept: ",intercept_)
#     print("the model is： y = ",coef_,"* X +(",intercept_,")")
#     
#     y_train_avg = np.average(y_train)
#     
#     print((np.sum((y_train-y_train_pred)**2)))
#    
#     print((np.sum((y_train-y_train_avg)**2)))
#     R2_train = 1-(np.average((y_train-y_train_pred)**2))/(np.average((y_train-y_train_avg)**2))
#     print("R2 in Train ： ",R2_train)
#     
#     y_test_avg = np.average(y_test)
#     R2_test = 1-(np.average((y_test-y_test_pred)**2))/(np.average((y_test-y_test_avg)**2))
#     print("R2 in Test ： ",R2_test)

closed_form_2()
    
    
    
    
