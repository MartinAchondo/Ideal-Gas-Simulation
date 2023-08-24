import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from scipy import interpolate


class Regression():

    def __init__(self):
        self.regr = linear_model.LinearRegression()


    def get_linear_regression(self, L_train, X_test_input, f: list=[]):
        x = np.array(L_train[0])
        y = np.array(L_train[1])
        if len(f)==0:
            self.X_train = np.array(x).reshape(-1,1)
            self.X_test = np.array(X_test_input).reshape(-1,1)
        else:
            n = len(f)
            self.X_train = np.zeros((len(x),n))
            for func,j in zip(f,range(len(x))):
                self.X_train[:,j] = func(x)
            self.X_test = np.zeros((len(X_test_input),n))
            for func,j in zip(f,range(len(X_test_input))):
                self.X_test[:,j] = func(np.array(X_test_input))
        self.Y_train = np.array(y).reshape(-1,1)

        self.regr.fit(self.X_train,self.Y_train)

        self.Y_pred = self.regr.predict(self.X_test)

        print("Coefficients: \n", self.regr.coef_)

        return self.Y_pred


class Interpolation():

    def __init__(self):
        self.interpolate = interpolate

    def get_interpoltion(self,x,y):

        y_tck = self.interpolate.splrep(x,y,s=0)
        x_fit = np.arange(np.min(x),np.max(x))
        y_new = self.interpolate.splev(x_fit,y_tck)

        return x_fit,y_new
