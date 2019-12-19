#encoding:utf-8
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as pyplot

# Problem 1&2

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
    
    
    
# Problem 3    
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model

 
def vif(X, thres=10.0):
    col = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:,col].values, ix) for ix in range(X.iloc[:,col].shape[1])]
        maxvif = max(vif)
        maxix = vif.index(maxvif)
        if maxvif > thres:
            del col[maxix]
            print('delete=',X.columns[col[maxix]],'  ', 'vif=',maxvif )
            dropped = True
    print('Remain Variables:', list(X.columns[col]))
    print('VIF:', vif)
    return list(X.columns[col]) 

dataset = pd.read_csv("./data/climate_change_1.csv")
X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])

y = dataset.get("Temp")

X_train = X[:284]
X_test = X[284:]
y_train = y[:284]
y_test = y[284:]
d = vif(X_train)
print(d)

X = dataset.get( ['MEI', 'CFC-12', 'Aerosols'])
y = dataset.get("Temp")
X_train = X[:284]
X_test = X[284:]
y_train = y[:284]
y_test = y[284:]

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
print('coefficients(b1,b2...):',regr.coef_)
print('intercept(b0):',regr.intercept_)
y_train_pred = regr.predict(X_train)
       
R2_1 = regr.score(X_train, y_train)
print(R2_1)
R2_2 = regr.score(X_test, y_test)
print(R2_2)




# Problem 4
#encoding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def costFunc(X,Y,theta):
    #cost func
    inner=np.power((X*theta.T)-Y,2)
    return np.sum(inner)/(2*len(X))

def gradientDescent(X,Y,theta,alpha,iters):
    temp = np.mat(np.zeros(theta.shape))
    cost = np.zeros(iters)
    thetaNums = int(theta.shape[1])
    
    for i in range(iters):
        error = (X*theta.T-Y)
        for j in range(thetaNums):
            derivativeInner = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j]-(alpha*np.sum(derivativeInner)/len(X))
        theta = temp
        cost[i]=costFunc(X,Y,theta)
    return theta,cost


dataset = pd.read_csv("./data/climate_change_1.csv")
X = dataset.get(["MEI","CO2","CH4","N2O","CFC-11","CFC-12","TSI","Aerosols"])

y = dataset.get("Temp")
X = np.column_stack((np.ones(len(X)),X))
X_train = X[:284]
X_test = X[284:]
y_train = y[:284]
y_test = y[284:]

X_train = np.mat(X_train)  
Y_train = np.mat(y_train).T

for i in range(1,9):
    X_train[:,i] = (X_train[:,i] - min(X_train[:,i])) / (max(X_train[:,i]) - min(X_train[:,i]))

theta_n = (X_train.T*X_train).I*X_train.T*Y_train
print("theta =",theta_n)
theta = np.mat([0,0,0,0,0,0,0,0,0])
iters = 100000
alpha = 0.001

finalTheta,cost = gradientDescent(X_train,Y_train,theta,alpha,iters)
print("final theta ",finalTheta)
print("cost ",cost)

# x_x0=[]*8
# for i in range(8):
#     x_x0[i] = np.linspace(X_train[:,i].min(),X_train[:,i].max(),100)
# 
# x_x0 = np.meshgrid(x_x0)
# f = 0
# for i in range(9):
#     if i == 0:
#         f += finalTheta[0,0]
#     else:
#         f += finalTheta[0,i]*x_x0[i]
fig, bx = plt.subplots(figsize=(8,6))
bx.plot(np.arange(iters), cost, 'r') 
bx.set_xlabel('Iterations') 
bx.set_ylabel('Cost') 
bx.set_title('Error vs. Training Epoch') 
plt.show()
