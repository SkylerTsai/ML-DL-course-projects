import numpy as np
import math
import tqdm

#========================================basic function=======================================

def E_01(y, predict):
	predict = np.reshape(predict, y.shape)
	return np.sum(y != predict) / y.shape[0]

def predict(x, G, alpha):
	pred = np.zeros(x.shape[0])

	for t in range(len(G)):
		for i in range(x.shape[0]):
			pred[i] += alpha[t] * G[t].pred(x[i])

	return np.sign(pred)

#========================================AdaBoost-Stump=======================================

class Stump:
	s, i, theta = 0, 0, 0

	def pred(self, x):
		return self.s * np.sign(x[self.i] - self.theta)

def DecisionStump(x, y, weight):
	stump = Stump()
	minErr = 1

	for i in range(x.shape[1]):
		x_temp = [x[j][i] for j in range(x.shape[0])]
		xyw = np.hstack((np.reshape(x_temp, y.shape), y, np.reshape(weight, y.shape)))
		xyw_sort = xyw[xyw[:, 0].argsort()]

		negativeSum, positiveSum = [0], [0]
		for j in range(xyw_sort.shape[0]):
			if xyw_sort[j][1] == 1:
				positiveSum.append(positiveSum[-1])
				negativeSum.append(negativeSum[-1] + xyw_sort[j][2])
			elif xyw_sort[j][1] == -1:
				positiveSum.append(positiveSum[-1] + xyw_sort[j][2])
				negativeSum.append(negativeSum[-1])

		for j in range(len(negativeSum) - 1):
			npErr = negativeSum[j] + positiveSum[-1] - positiveSum[j]			
			pnErr = positiveSum[j] + negativeSum[-1] - negativeSum[j]

			if min(npErr, pnErr) < minErr:
				minErr = min(npErr, pnErr)
				if npErr < pnErr:
					stump.s = 1
				else:
					stump.s = -1

				stump.i = i

				if j == 0:
					stump.theta = xyw_sort[0][0] - 1
				else:
					stump.theta = (xyw_sort[j - 1][0] + xyw_sort[j][0]) / 2

	return stump

def epsilonCal(x, y, g, weight):
	errSum = 0
	
	for i in range(x.shape[0]):
		if g.pred(x[i]) != y[i]:
			errSum += weight[i]

	return errSum / np.sum(weight)

def weightUpdate(x, y, g, weight, scal):
	for i in range(weight.shape[0]):
		if g.pred(x[i]) == y[i]:
			weight[i] /= scal
		elif g.pred(x[i]) != y[i]:
			weight[i] *= scal

	return weight

def AdaptiveBoosting(x, y, T = 500):
	weight = np.ones(x.shape[0]) / x.shape[0]
	G = []
	alpha = []

	for t in tqdm.trange(T):
		g = DecisionStump(x, y, weight)
		#print('g', g.s, g.i, g.theta)
		eps = epsilonCal(x, y, g, weight)
		scal = math.sqrt((1 - eps) / eps)
		#print('eps', eps, 'scal', scal)
		weight = weightUpdate(x, y, g, weight, scal)
		#print('weight', weight[:10])

		G.append(g)
		alpha.append(math.log(scal))
		#print('alpha', alpha[-1])

	return G, alpha


#=============================================================================================
def main():
	trainData = np.loadtxt('hw6_train.dat')
	testData = np.loadtxt('hw6_test.dat')
	x_train, y_train = np.hsplit(trainData, [-1])
	x_test, y_test = np.hsplit(testData, [-1])

	p11, p12, p13, p14, p15, p16 = True, True, True, True, True, True

	G, alpha = AdaptiveBoosting(x_train, y_train)
	pred = predict(x_train, G, alpha)
	print(E_01(y_train, pred))
	
	if p11:
		pred = predict(x_train, G[:1], alpha[:1])
		print('Problem 11:', 'Ein(g_1) =', E_01(y_train, pred))

	if p12:
		maxEin = 0
		for i in range(len(G)):
			pred = predict(x_train, G[i : i+1], alpha[i : i+1])
			if E_01(y_train, pred) > maxEin:
				maxEin = E_01(y_train, pred)
		print('Problem 12:', 'max Ein(g_t) =', maxEin)

	if p13:
		minT = 0
		pred = np.zeros(x_train.shape[0]) 

		for t in range(len(G)):
			for i in range(x_train.shape[0]):
				pred[i] += alpha[t] * G[t].pred(x_train[i])
				if E_01(y_train, np.sign(pred)) <= 0.05:
					minT = t + 1
					break

			if minT != 0:
				break

		print('Problem 13:', 'min Ein(G_t) < 0.05, t =', minT)

	if p14:
		pred = predict(x_test, G[:1], alpha[:1])
		print('Problem 14:', 'Eout(g_1) =', E_01(y_test, pred))

	if p15:
		pred = np.zeros(x_test.shape[0])

		for t in range(len(G)):
			for i in range(x_test.shape[0]):
				pred[i] += G[t].pred(x_test[i])

		print('Problem 15:', 'Eout(G_uniform) =', E_01(y_test, np.sign(pred)))

	if p16:
		pred = predict(x_test, G, alpha)
		print('Problem 16:', 'Eout(G_500) =', E_01(y_test, pred))

if __name__ == '__main__':
	main()