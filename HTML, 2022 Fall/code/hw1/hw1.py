import numpy as np
import random
import time

def sign(x):
	if x > 0:
		return 1
	else:
		return -1

def PLA(x, y):
	w = np.zeros(x.shape[1])
	t = 0
	random.seed(time.time())

	while True:
		rd = random.randint(0, x.shape[0]-1)
		pred = sign((w.T).dot(x[rd]))
		if pred == y[rd][0]:
			if t > 5 * x.shape[0]:
				break
		else:
			w += x[rd] * y[rd]
		t += 1

	return w

def preprocess(x):
	prob = 16
	print("The result of problem", prob, "is:")

	if prob == 16:
		x = np.hstack((np.zeros((x.shape[0], 1)), x))
	else:
		x = np.hstack((np.ones((x.shape[0], 1)), x))

	if prob == 14:
		x *= 2
	elif prob == 15:
		for i in range(x.shape[0]):
			norm = np.linalg.norm(x[i])
			x[i] /= norm
	
	#print(x)

	return x

def main():
	data = np.loadtxt('hw1_train.dat')

	x, y = np.hsplit(data, [-1])
	x = preprocess(x)
	#print(x.shape, y.shape)

	t = 1000
	tot = 0
	for i in range(t):
		w = PLA(x, y)
		tot += w.dot(w)
	print(tot / t)
	
	return

if __name__ == '__main__':
	main()