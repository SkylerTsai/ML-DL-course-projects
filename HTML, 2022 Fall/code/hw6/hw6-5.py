import numpy as np
import math
import tqdm

#========================================basic function=======================================

def E_01(y, predict):
	predict = np.reshape(predict, y.shape)
	return np.sum(y != predict) / y.shape[0]

def predict(x, y, G, alpha):
	pred = np.zeros(x.shape[0])
	Ein = []

#	print('=====Start=====')
	for t in range(len(G)):
		for i in range(x.shape[0]):
			pred[i] += alpha[t] * G[t].pred(x[i])
		
		Ein.append(E_01(y, np.sign(pred)))
		
		if(len(Ein) > 1 and Ein[-1] > Ein[-2] and Ein[-2] > 0):
			for i in range(len(G)):
				print(G[i].s, G[i].theta, alpha[i])

			print('found!!!!!!!!')
			print('Ein', Ein)
			return True
	
#	print('======End======')
	return False

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

	for t in range(T):
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
	N = 6
	x = np.reshape(np.arange(N), (-1, 1))
	for i in range(2**N):
		y = np.reshape(np.zeros(N), (-1, 1))
		for j in range(N):
			if (i >> j) & 1 == 1:
				y[j] = -1
			else:
				y[j] = 1

		G, alpha = AdaptiveBoosting(x, y, 5)

		if(predict(x, y, G, alpha)):
			print('y', y)
			

if __name__ == '__main__':
	main()