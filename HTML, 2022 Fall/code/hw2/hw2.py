import numpy as np
import random
import time
import math
import tqdm

#========================================Basic Function======================================
def sign(arr):
	for i in range(arr.shape[0]):
		if arr[i] < 0:
			arr[i] = -1
		else:
			arr[i] = 1

	return arr

def sigmoid(arr):
	for i in range(arr.shape[0]):
		arr[i] = 1 / (1 + math.exp(-arr[i]))
	return arr

def E_sqr(x, y, w):
	y_hat = np.dot(x, w)
	return np.sum((y - y_hat)**2) / y.shape[0]

def E_01(x, y, w):
	y_hat = sign(np.dot(x, w))
	return np.sum(y != y_hat) / y.shape[0]

#========================================Linear Regression====================================
def LinearRegression(x, y):
	x_pi = np.linalg.pinv(x)
	w_lin = np.dot(x_pi, y)
	#print(w_lin, w_lin.shape)
	return w_lin

#========================================Logistic Regression==================================
def gradient(x, y, w):
	err = np.zeros(x.shape[1])
	for i in range(x.shape[0]):
		err += sigmoid(-y[i] * np.dot(w.T, x[i])) * (-y[i] * x[i])
	#print(err, err.shape)
	return err / x.shape[0]

def LogisticRegression(x, y):
	 lr = 0.1
	 T = 500
	 w = np.zeros((x.shape[1], 1))
	 for i in range(T):
	 	err = gradient(x, y, w)
	 	w = w - lr * np.reshape(err, w.shape)
	 return w

#========================================Generate Data========================================
def flipCoin():
	return random.choice([-1, 1])

def generateData(num):
	x = np.zeros((num, 2))
	y = np.zeros(num)

	for i in range(num):
		y[i] = flipCoin()
		if y[i] == 1:
			x[i] = np.random.multivariate_normal([2, 3], [[0.6, 0], [0, 0.6]])
		elif y[i] == -1:
			x[i] = np.random.multivariate_normal([0, 4], [[0.4, 0], [0, 0.4]])
	
	x = np.hstack((np.ones((num, 1)), x))

	return x, y

def outlierData(num):
	x = np.zeros((num, 2))
	y = np.ones(num)
	for i in range(num):
		x[i] = np.random.multivariate_normal([6, 0], [[0.3, 0], [0, 0.1]])

	x = np.hstack((np.ones((num, 1)), x))

	return x, y

#=============================================================================================

def main():
	T = 100
	err_sqr = 0
	err_01_train_lin = 0
	err_01_test_lin = 0
	err_01_test_log = 0
	err_01_out_lin = 0
	err_01_out_log = 0

	p13, p14, p15, p16 = True, False, False, False

	for i in tqdm.trange(T):
		random.seed(time.time())
		x_Train, y_Train = generateData(200)
		x_Test, y_Test = generateData(5000)
		if p13 or p14 or p15:
			w_lin = LinearRegression(x_Train, y_Train)
			
			if p13:
				err_sqr += E_sqr(x_Test, y_Test, w_lin)
			
			if p14 or p15:
				err_01_train_lin += E_01(x_Train, y_Train, w_lin)
				err_01_test_lin  += E_01(x_Test, y_Test, w_lin)
		
		if p15:
			w_log = LogisticRegression(x_Train, y_Train)
			w_log = np.reshape(w_log, w_lin.shape)
			err_01_test_log  += E_01(x_Test, y_Test, w_log)
		
		if p16:
			x_out, y_out = outlierData(20)
			x_Train = np.vstack((x_Train, x_out))
			y_Train = np.concatenate((y_Train, y_out))

			w_lin = LinearRegression(x_Train, y_Train)
			w_log = LogisticRegression(x_Train, y_Train)
			w_log = np.reshape(w_log, w_lin.shape)
			err_01_out_lin += E_01(x_Test, y_Test, w_lin)
			err_01_out_log += E_01(x_Test, y_Test, w_log)		

	if p13:
		print("Problem 13:", err_sqr / T)
	if p14:
		print("Problem 14:", abs(err_01_train_lin / T - err_01_test_lin / T))
	if p15:
		print("Problem 15:", err_01_test_lin / T, err_01_test_log / T)
	if p16:
		print("Problem 16:", err_01_out_lin / T, err_01_out_log / T)
	
	return

if __name__ == '__main__':
	main()