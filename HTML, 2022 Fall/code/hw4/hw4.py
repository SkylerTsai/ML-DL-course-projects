from liblinear.liblinearutil import *
import numpy as np
import math

#========================================basic function=======================================

def E_01(y, predict):
	return np.sum(y != predict) / y.shape[0]

def calLambda(lambdaValue):
	return - round(math.log(float(lambdaValue) * 2, 10), 0)

#========================================Transform============================================

def ThirdOrderPolynomialTransform(data):
	data = np.hstack((np.ones((data.shape[0], 1)), data))
	#print(transformedData.shape) # 1 + 6= 7

	transformedData = np.ones((data.shape[0], 1))

	for i in range(data.shape[1]):
		for j in range(i, data.shape[1]):
			for k in range(max(1, j), data.shape[1]):
				newData = np.zeros((data.shape[0], 1))
				
				for l in range(data.shape[0]):
					newData[l][0] = data[l][i] * data[l][j] * data[l][k]
				
				transformedData = np.hstack((transformedData, newData))
	
	#print(transformedData.shape[0]) # 1 + (6) + (6 + 15) + (6 + 15 * 2 + 20) = 84 

	return transformedData

#========================================V folds==============================================

def VfoldsSplit(x_folds, y_folds, val):
	x_spl_val = x_folds[val]
	x_spl_train = [x_folds[i] for i in range(len(x_folds)) if i != val]
	x_spl_train = np.reshape(x_spl_train, (-1, x_folds[0].shape[1]))
	y_spl_val = y_folds[val]
	y_spl_train = [y_folds[i] for i in range(len(y_folds)) if i != val]
	y_spl_train = np.reshape(y_spl_train, -1)

	return x_spl_train, x_spl_val, y_spl_train, y_spl_val

#========================================liblinear============================================

def FindParam(x_train, y_train, x_val, y_val, x_test, y_test, params):
	prob = problem(y_train, x_train)
	minEval, minPar, Eout, lambdaValue = 1, 0, 1, ''
	for par in params:
		param = parameter(par)
		model = train(prob, param)
		p_lab, p_acc, p_val = predict(y_train, x_train, model, '-q')
		#print('Ein', E_01(y_train, p_lab))
		
		p_lab, p_acc, p_val = predict(y_val, x_val, model, '-q')
		#print('Eval', E_01(y_val, p_lab))

		if E_01(y_val, p_lab) <= minEval:
			minEval = E_01(y_val, p_lab)
			minPar = par
			lambdaValue = par[8:-15]
		
		p_lab, p_acc, p_val = predict(y_test, x_test, model, '-q')
		#print('Eout', E_01(y_test, p_lab))

	param = parameter(minPar)
	model = train(prob, param)
	p_lab, p_acc, p_val = predict(y_test, x_test, model, '-q')
	Eout = E_01(y_test, p_lab)

	return minEval, lambdaValue, Eout

#=============================================================================================

def main():
	trainData = np.loadtxt('hw4_train.dat')
	testData = np.loadtxt('hw4_test.dat')
	x_train, y_train = np.hsplit(trainData, [-1])
	x_test, y_test = np.hsplit(testData, [-1])

	x_train = ThirdOrderPolynomialTransform(x_train)
	y_train = np.reshape(y_train, -1)
	x_test = ThirdOrderPolynomialTransform(x_test)
	y_test = np.reshape(y_test, -1)

	p12, p13, p14, p15, p16 = True, True, True, True, True
	pStr = ['-s 0 -c 5000 -e 0.000001 -q',
	        '-s 0 -c 50 -e 0.000001 -q',
	        '-s 0 -c 0.5 -e 0.000001 -q',
	        '-s 0 -c 0.005 -e 0.000001 -q',
	        '-s 0 -c 0.00005 -e 0.000001 -q']
	
	if p12:
		minEout, lambdaValue, Eout = FindParam(x_test, y_test, x_test, y_test, x_test, y_test, pStr)
		print('Problem 12', 'lambda =', calLambda(lambdaValue), 'min Eout =', minEout)

	if p13:
		minEin, lambdaValue, Ein = FindParam(x_train, y_train, x_train, y_train, x_train, y_train, pStr)
		print('Problem 13', 'lambda =', calLambda(lambdaValue), 'min Ein =', minEin)

	if p14 or p15:
		x_spl_train, x_spl_val = x_train[:120], x_train[120:]
		y_spl_train, y_spl_val = y_train[:120], y_train[120:]
		minEval, lambdaValue, Eout = FindParam(x_spl_train, y_spl_train, x_spl_val, y_spl_val, x_test, y_test, pStr)
		
		if p14:	
			print('Problem 14', 'lambda =', calLambda(lambdaValue), 'Eout =', Eout)

		if p15:
			prob = problem(y_train, x_train)
			param = parameter('-s 0 -c ' + lambdaValue + ' -e 0.000001 -q')
			model = train(prob, param)
			p_lab, p_acc, p_val = predict(y_test, x_test, model, '-q')
			Eout = E_01(y_test, p_lab)
			print('Problem 15', 'lambda =', calLambda(lambdaValue), 'Eout =', Eout)

	if p16:
		x_folds = [x_train[:40], x_train[40:80], x_train[80:120], x_train[120:160], x_train[160:]]
		y_folds = [y_train[:40], y_train[40:80], y_train[80:120], y_train[120:160], y_train[160:]]
		minErr, lambdaValue = 1, ''

		for par in pStr:
			E_val = 0
			for val in range(len(x_folds)):
				x_spl_train, x_spl_val, y_spl_train, y_spl_val = VfoldsSplit(x_folds, y_folds, val)
				prob = problem(y_spl_train, x_spl_train)
				param = parameter(par)
				model = train(prob, param)
				p_lab, p_acc, p_val = predict(y_spl_val, x_spl_val, model, '-q')
				E_val += E_01(y_spl_val, p_lab) / len(x_folds)

			if E_val <= minErr:
				minErr = E_val
				lambdaValue = par[8:-15]

		print('Problem 16', 'lambda =', calLambda(lambdaValue), 'Ecv =', minErr)
			

if __name__ == '__main__':
	main()