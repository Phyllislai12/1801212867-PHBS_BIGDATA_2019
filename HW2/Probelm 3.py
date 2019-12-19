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












        
            

        
    

    
