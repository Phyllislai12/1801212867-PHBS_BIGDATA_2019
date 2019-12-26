#encoding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# The goal of optimization is to minimize the loss function. The gradient direction of the function represents the direction where the value of the function grows fastest.
# So the opposite direction is the direction in which the function decreases the fastest. The optimal idea for gradient descent is to use the negative gradient of the current position
# Direction as the search direction, also known as the "steepest descent method." Gradient descent method is an iterative algorithm, each step needs to solve the objective function and gradient vector.
# steps:
# 1. Find the gradient,
# 2. Move in the direction opposite to the gradient, as follows, where, is the step size. If the step size is small enough, it can be guaranteed that every iteration is decreasing, but it may cause the convergence to be too slow. If the step size is too large, it cannot guarantee that every iteration is decreasing, nor can it guarantee the convergence.
# 3. Loop iteration step 2, until the value changes so that the difference between the two iterations is small enough, such as 0.00000001. In other words, until the value calculated by the two iterations is basically unchanged, then the local minimum value has been reached.
# 4. In this case, the output is the value that causes the function to be minimal
# implementation process:
# loss function

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
            temp[0,j] = theta[0,j]-(alpha*np.sum(derivativeInner)/len(X))#Compute the theta matrix
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

# feature scaling

for i in range(1,9):
    X_train[:,i] = (X_train[:,i] - min(X_train[:,i])) / (max(X_train[:,i]) - min(X_train[:,i]))

theta_n = (X_train.T*X_train).I*X_train.T*Y_train
print("theta =",theta_n)
theta = np.mat([0,0,0,0,0,0,0,0,0])
iters = 100000 # The number of iterations
alpha = 0.001 # learning rate

finalTheta,cost = gradientDescent(X_train,Y_train,theta,alpha,iters)
print("final theta ",finalTheta)
print("cost ",cost)

fig, bx = plt.subplots(figsize=(8,6))
bx.plot(np.arange(iters), cost, 'r') 
bx.set_xlabel('Iterations') 
bx.set_ylabel('Cost') 
bx.set_title('Error vs. Training Epoch') 
plt.show()
# As the number of iterations increases, the loss function becomes smaller and smaller, the trend becomes more and more stable, and the optimal solution is approached at the same time
    
# When the data is iterated for 32 times, it will produce a data of 30 length. This data is too accurate to report errors, but it will be confused later（数据在迭代32次的时候会产生一个30长度的数据，这个数据超精度了，不会报错，但是后面乱算了）
