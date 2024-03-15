from libsvm.svmutil import *
import numpy as np
import math
import tqdm

#========================================basic function=======================================

def E_01(y, predict):
	y = np.reshape(y, -1)
	predict = np.reshape(predict, -1)
	#print('E_01:', np.sum(y != predict), '/', y.shape[0])
	return np.sum(y != predict) / y.shape[0]

#========================================Transform============================================

def OneLabelTransform(data, target):
	temp = data.copy()
	for i in range(len(temp)):
		if temp[i] == target:
			temp[i] = 1
		else:
			temp[i] = -1

	return np.array(temp).astype(np.int)

#========================================liblinear============================================

def MySVM(x_train, y_train, x_test, y_test, par, ReturnNode=False):
	prob = svm_problem(y_train, x_train)
	param = svm_parameter(par)
	model = svm_train(prob, param)
	p_label, p_acc, p_val = svm_predict(y_test, x_test, model, '-q')
	
	if ReturnNode:
		return E_01(y_test, p_label), model.nSV[0] + model.nSV[1]
	else:
		return E_01(y_test, p_label)

#=============================================================================================

def main():
	y_train, x_train = svm_read_problem('satimage.scale')
	y_test, x_test = svm_read_problem('satimage.scale.t')

	p11, p12, p13, p14, p15, p16 = True, True, True, True, True, True

	if p11:
		y_train_5 = OneLabelTransform(y_train, 5)

		prob = svm_problem(y_train_5, x_train)
		param = svm_parameter('-s 0 -t 0 -c 10 -q')
		model = svm_train(prob, param)
		p_label, p_acc, p_val = svm_predict(y_train_5, x_train, model, '-q')

		w = np.zeros(len(x_train) + 1)

		for i in range(model.l):
			for node in  model.SV[i]:
				if node.index == -1:
					break
				w[node.index] += model.sv_coef[0][i] * node.value

		print('Problem 11:', '|w| =', math.sqrt(np.dot(w, w)))

	if p12 or p13:
		label, Ein, NodeNumber = [], [], []
		for i in range(2, 7):
			y_train_i = OneLabelTransform(y_train, i)
			par = '-s 0 -t 1 -d 3 -c 10 -g 1 -r 1 -q'
			ein, nn = MySVM(x_train, y_train_i, x_train, y_train_i, par, ReturnNode=True)

			label.append(i)
			Ein.append(ein)
			NodeNumber.append(nn)

		if p12:
			ans = np.argmax(Ein)
			print('Problem 12:', 'OVA class =', label[ans], 'with Ein', Ein[ans])
		
		if p13:
			ans = np.argmax(NodeNumber)
			print('Problem 13:', 'Node number =', NodeNumber[ans])

	if p14:
		y_train_1 = OneLabelTransform(y_train, 1)
		y_test_1 = OneLabelTransform(y_test, 1)
		C, Eout = [0.01, 0.1, 1, 10, 100], []

		for c in C:
			par = '-s 0 -t 2 -g 10 -c ' + str(c) + ' -q'
			eout = MySVM(x_train, y_train_1, x_test, y_test_1, par)
			Eout.append(eout)

		ans = np.argmin(Eout)
		print('Problem 14:', 'C =', C[ans], 'with Eout', Eout[ans])

	if p15:
		y_train_1 = OneLabelTransform(y_train, 1)
		y_test_1 = OneLabelTransform(y_test, 1)
		G, Eout = [0.1, 1, 10, 100, 1000], []

		for g in G:
			par = '-s 0 -t 2 -c 0.1 -g ' + str(g) + ' -q'
			eout = MySVM(x_train, y_train_1, x_test, y_test_1, par)
			Eout.append(eout)

		ans = np.argmin(Eout)
		print('Problem 15:', 'Gamma =', G[ans], 'with Eout', Eout[ans])

	if p16:
		T = 1000
		y_train_1 = OneLabelTransform(y_train, 1)
		G = [0.1, 1, 10, 100, 1000]
		score = np.zeros(len(G))

		for i in tqdm.trange(T):
			rd_val = np.random.choice(len(x_train), 200)
			rd_tra = np.delete(np.array(np.arange(len(x_train))), rd_val)
			x_val, y_val = np.array(x_train)[rd_val], np.array(y_train_1)[rd_val]
			x_tra, y_tra = np.array(x_train)[rd_tra], np.array(y_train_1)[rd_tra]

			Eval = []

			for g in G:
				par = '-s 0 -t 2 -c 0.1 -g ' + str(g) + ' -q'
				ev = MySVM(x_tra, y_tra, x_val, y_val, par)
				Eval.append(ev)

			best = np.argmin(Eval)
			score[best] += 1

			if(i > 0 and i % 100 == 0):
				print(score)

		ans = np.argmax(score)
		print('Problem 16:', 'Gamma =', G[ans], 'for selected', score[ans], 'times')



if __name__ == '__main__':
	main()