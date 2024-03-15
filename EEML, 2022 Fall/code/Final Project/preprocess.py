import sys
import csv
import argparse
import numpy as np
from lab_dict import label_type

normalize = True

def get_data(train_file):
	data = []

	with open(train_file, newline = '') as csvfile:
		rows = csv.reader(csvfile)
		for row in rows:
			data.append(row)

	return data

def save_data(feature, data, path='preprocessed_train.csv'):
	out = []
	out.append(feature)
	for i in range(len(data)):
		out.append(data[i])

	print("out", len(out), len(out[0]))
	
	with open(path, 'w', newline = '') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(out)

	return

def create_01(data):
	temp = np.zeros((len(data)-1, len(label_type[data[0]]))).astype(np.float32)
	#print(temp.shape)
	for i in range(len(label_type[data[0]])):
#		print((label_type[data[0]])[i])

		for j in range(1, len(data)):
			if data[j] == (label_type[data[0]])[i]:
				temp[j-1][i] = 1

	if np.sum(temp) != len(data) - 1 and data[0] != 'CentralAir':
		print(data[0], "error")
		print(np.sum(temp))
		for i in range(len(temp)):
			if np.sum(temp[i]) != 1:
				print("error on", i, data[i + 1], len(data[i]))
				# break
	
	return temp

def str2float(data):
	temp = np.zeros((len(data) - 1, 1)).astype(np.float32)
	
	for i in range(1, len(data)):
		try:
			temp[i - 1] = data[i]
		except:
			temp[i - 1] = np.nan

	n_mean = np.nanmean(temp)
	for i in range(1, len(data)):
		if np.isnan(temp[i - 1]):
			temp[i - 1] = n_mean

#	print(temp)

	return temp

def get_grade(data):
	temp = np.zeros((len(data) - 1, 1)).astype(np.float32)
	grade = 1
	d = 1 / (len(label_type[data[0]]) - 1)

	for i in range(len(label_type[data[0]])):
		for j in range(1, len(data)):
			if data[j] == (label_type[data[0]])[i]:
				temp[j - 1] = grade

		grade -= d

	return temp


def transform(data):
	new_data = np.array((len(data) - 1, 0))
	temp = []

	for i, label in enumerate(data[0]):
		if label_type[label] == None:
			temp = str2float([data[j][i] for j in range(len(data))])
#		elif (label_type[label])[0] == 'Ex':
#			temp = get_grade([data[j][i] for j in range(len(data))])
		else:
			temp = create_01([data[j][i] for j in range(len(data))])
			
#		print("temp", temp.shape)

		if i == 0:
			new_data = np.reshape(temp, (len(temp), 1))
		else:
			new_data = np.hstack((new_data, temp))
#		print("new", new_data.shape)

	return new_data

def get_feature_name(features):
	temp = []
	for feature in features:
		if label_type[feature] == None:
			temp.append(feature)
#		elif (label_type[feature])[0] == 'Ex':
#			temp.append(feature)
		else:
			for sub_feature in label_type[feature]:
				temp.append(feature + "_" + sub_feature)

#	print(len(temp), temp)

	return temp

def main():
	# Get path of train/test data
	parser = argparse.ArgumentParser()
	path_prefix = "./"
	parser.add_argument("train", help="path of train data")
	parser.add_argument("test", help="path of test data")
	args = parser.parse_args()
	print(f"args: {[(x, getattr(args, x)) for x in vars(args)]}")

	# Preprocess train data
	train_data = get_data(args.train)

	new_feature_name = get_feature_name(train_data[0])
	train_data = transform(train_data)

	t_min, t_max = np.amin(train_data, axis = 0), np.amax(train_data, axis = 0)
	t_min[0], t_min[-1], t_max[0], t_max[-1] = 0, 0, 1, 1
	for i in range(len(t_max)):
		if t_max[i] == 0:
			t_max[i] = 1 

	if normalize:
		train_data = (train_data - t_min) / (t_max - t_min)

	save_data(new_feature_name, train_data)

	data_id, train_data, train_label = np.hsplit(train_data, [1, -1])
	print(train_data.shape, train_label.shape)

	np.save("X_train", train_data)
	np.save("y_train", train_label)
	
	# Preprocess test data
	test_data = get_data(args.test)

	new_feature_name = get_feature_name(test_data[0])
	test_data = transform(test_data)

	if normalize:
		test_data = (test_data - t_min[:-1]) / (t_max[:-1] - t_min[:-1])

	save_data(new_feature_name, test_data, path="preprocessed_test.csv")

	data_id, test_data = np.hsplit(test_data, [1])
	print(test_data.shape)

	np.save("X_test", test_data)

	return

if __name__ == "__main__":
	main()