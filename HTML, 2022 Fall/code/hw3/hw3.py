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

def E_01(x, y, w):
	y_hat = sign(np.dot(x, w))
	return np.sum(y != y_hat) / y.shape[0]

#========================================Linear Regression====================================
def LinearRegression(x, y):
	x_pi = np.linalg.pinv(x)
	w_lin = np.dot(x_pi, y)
	#print(w_lin, w_lin.shape)
	return w_lin

#========================================Transform============================================
def homogeneousOrderTransform(data, Q):
	newData = np.zeros((data.shape[0], data.shape[1] * Q))
	for q in range(Q):
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				newData[i][q * data.shape[1] + j] = data[i][j] ** (q + 1)

	newData = np.hstack((np.ones((newData.shape[0], 1)), newData))

	return newData

def fullOrderTransform(data):
	newData = np.zeros((data.shape[0], 10 + 45 + 10))
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			newData[i][j] = data[i][j]

	pos = data.shape[1]
	for k in range(data.shape[1]):
		for l in range(k, data.shape[1]):
			for i in range(data.shape[0]):
				newData[i][pos] = data[i][k] * data[i][l]
			pos += 1

	newData = np.hstack((np.ones((newData.shape[0], 1)), newData))

	return newData

def lowerDimensionTransform(data, Q):
	newData = np.zeros((data.shape[0], Q))
	for i in range(data.shape[0]):
		for j in range(Q):
			newData[i][j] = data[i][j]

	newData = np.hstack((np.ones((newData.shape[0], 1)), newData))
	return newData

def randomDimensionTransform(data, sel):
	newData = np.zeros((data.shape[0], len(sel)))
	for i in range(data.shape[0]):
		for j in range(len(sel)):
			newData[i][j] = data[i][sel[j]]

	newData = np.hstack((np.ones((newData.shape[0], 1)), newData))
	return newData
#=============================================================================================

def main():
	trainData = np.loadtxt('hw3_train.dat')
	testData = np.loadtxt('hw3_test.dat')
	x_train, y_train = np.hsplit(trainData, [-1])
	x_test, y_test = np.hsplit(testData, [-1])

	p12, p13, p14, p15, p16 = True, True, True, True, True

	if p12:
		x_train_HOT = homogeneousOrderTransform(x_train, 2)
		x_test_HOT = homogeneousOrderTransform(x_test, 2)
		w = LinearRegression(x_train_HOT, y_train)
		print("Problem 12:", abs(E_01(x_train_HOT, y_train, w) - E_01(x_test_HOT, y_test, w)))

	if p13:
		x_train_HOT = homogeneousOrderTransform(x_train, 8)
		x_test_HOT = homogeneousOrderTransform(x_test, 8)
		w = LinearRegression(x_train_HOT, y_train)
		print("Problem 13:", abs(E_01(x_train_HOT, y_train, w) - E_01(x_test_HOT, y_test, w)))

	if p14:
		x_train_FOT = fullOrderTransform(x_train)
		x_test_FOT = fullOrderTransform(x_test)
		w = LinearRegression(x_train_FOT, y_train)
		print("Problem 14:", abs(E_01(x_train_FOT, y_train, w) - E_01(x_test_FOT, y_test, w)))

	if p15:
		minErr, minDim = 1000, 0
		for i in range(x_train.shape[1]):
			x_train_LDT = lowerDimensionTransform(x_train, i+1)
			x_test_LDT = lowerDimensionTransform(x_test, i+1)
			w = LinearRegression(x_train_LDT, y_train)
			err = abs(E_01(x_train_LDT, y_train, w) - E_01(x_test_LDT, y_test, w))
			#print("Dim:", i+1, "Err:", err)
			if err < minErr:
				minErr = err
				minDim = i + 1

		print("Problem 15:", minDim)

	if p16:
		t = 200
		errSum = 0
		for i in tqdm.trange(t):
			random.seed(time.time())
			sel = random.sample(range(x_train.shape[1]), 5)
			x_train_RDT = randomDimensionTransform(x_train, sel)
			x_test_RDT = randomDimensionTransform(x_test, sel)
			w = LinearRegression(x_train_RDT, y_train)
			errSum += abs(E_01(x_train_RDT, y_train, w) - E_01(x_test_RDT, y_test, w))

		print("Problem 16:", errSum / t)

if __name__ == '__main__':
	main()