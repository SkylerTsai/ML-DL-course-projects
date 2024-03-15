import csv
import numpy as np
from random import randrange
from sklearn.linear_model import SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, classification_report

param_grid = {
#	'alpha': [1e-2, 1e-3, 1e-4, 1e-5], 
#	'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
#	'n_estimators': range(10000, 50001, 10000), 
#	'learning_rate': [1e-2, 1e-3, 1e-4],
#	'max_features': ['sqrt', 'log2', 0.1], 
	'C': [1, 2, 3, 4, 5],
	'epsilon': [0, 1, 2, 3, 4, 5], 
}

def save_data(data, path='output.csv'):
	out = []
	out.append(['Id', 'SalePrice'])
	for i in range(len(data)):
		out.append([int(i+1461), data[i]])
	
	with open(path, 'w', newline = '') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(out)

	return

def RMSE(predict, Y):
	# return np.sqrt(np.mean(np.log((Y / predict))**2))
	return np.sqrt(np.mean((Y - predict)**2))

score = make_scorer(RMSE, greater_is_better = False)

def add_dim(x, size):
	td = np.array(x)
	for i in range(1, int(size / td.shape[1])):
		x = np.hstack((td**(i+1), x))

	return x

def get_SGD_model(rs):
	model = SGDRegressor(loss = 'squared_epsilon_insensitive', 
						 epsilon = 1e-2, 
						 l1_ratio = 0.15, 
						 penalty = 'elasticnet', 
						 fit_intercept = False, 
						 alpha = 1e-2, 
						 max_iter = 10000, 
						 tol = 1e-4, 
						 shuffle = True, 
						 random_state = rs, 
						 learning_rate = 'invscaling',
						 early_stopping = False,
						 validation_fraction = 0.0001, 
						 n_iter_no_change = 10, 
						 verbose = 0, 
						)
	return model

def get_Ridge_model(rs):
	model = Ridge(alpha = 1, 
				  fit_intercept = False, 
				  tol = 1e-3, 
				  random_state = rs, 
				 )
	return model

def get_Lasso_model(rs):
	model = Lasso(alpha = 1,
				  fit_intercept = False, 
				  tol = 1e-4, 
				  random_state = rs,
				  max_iter = 10000,  
				 )
	return model

def get_ElasticNet_model(rs):
	model = ElasticNet(alpha = 0.9, 
					   fit_intercept = False, 
					   tol = 1e-4, 
					   random_state = rs,
					   max_iter = 2000
					  )
	return model

def get_SVR_model(rs):
	model = LinearSVR(epsilon = 0,
					  tol = 1e-4, 
					  C = 1, 
					  loss='epsilon_insensitive',
					  fit_intercept = False, 
					  verbose = 0, 
					  random_state = rs, 
					  max_iter = 1e6, 
					 )
	return model

def get_ABR_model(rs):
	model = AdaBoostRegressor(n_estimators = 3000,
							  learning_rate = 1e-2,
							  random_state = rs, 
							 )
	return model

def get_GBR_model(rs):
	model = GradientBoostingRegressor(loss = 'ls', 
									  learning_rate = 1e-3, 
									  n_estimators = 10000, 
									  subsample = 0.2, 
									  criterion = 'friedman_mse', 
									  max_features = 'sqrt', 
									  verbose = 0, 
#                                     validation_fraction = 0.01, 
#                                     n_iter_no_change = 100, 
									  tol = 1e-4,
									  random_state = rs, 
									 )
	return model

def get_gridsearchCV(rs):
	model = get_SGD_model(rs)
#	model = get_Ridge_model(rs)
#	model = get_SVR_model(rs)
#   model = get_ABR_model(rs)
#	model = get_GBR_model(rs)

	grid = GridSearchCV(estimator = model, 
						param_grid = param_grid,
						scoring = score, 
						cv = 5,  
						n_jobs = 16, 
						verbose = 1, 
						return_train_score = True
					   )
	return grid


def print_result(ttype, model, data, label, y_min, y_max):
	predict = model.predict(data)
	# predict = np.exp(predict)
	predict = np.clip(predict, y_min, y_max)
	rmse    = RMSE(predict, label.squeeze())
	print(ttype, rmse)

	return rmse

def main():
	X = np.load('X_train_115.npy').astype(np.float)
	y = np.load('y_train_115.npy').astype(np.float)
#   print(X, X.shape)
#   print(y, y.shape)
	y_min, y_max = np.log(10000), np.log(1000000)
#   print(y_min, y_max)
	
	dim = 2
	p = 0.2
	rs = 87
	rs = randrange(0, 2e9, 1)

	add_dim(X, len(X[0]) * dim)
	X = np.hstack((X, np.ones((len(X), 1))))

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = p, random_state = rs)
	y_train = y_train.squeeze()

	model_types = ['GBR', 'Ridge', 'Lasso', 'ElasticNet', 'SVR', 'ABR']
	# model_types = ['GBR', 'Ridge', 'Lasso', 'ElasticNet', 'SVR', 'ABR', 'SGD']
	# model_types = ['gridsearcv']
	sub_model, weights = [], []

	for i, mtype in enumerate(model_types):
		sub_model += [[mtype, None]]

		if mtype == 'gridsearcv':
			sub_model[i][1] = get_gridsearchCV(rs)
		elif mtype == 'GBR':
			sub_model[i][1] = get_GBR_model(rs)
		elif mtype == 'Ridge':
			sub_model[i][1] = get_Ridge_model(rs)
		elif mtype == 'Lasso':
			sub_model[i][1] = get_Lasso_model(rs)
		elif mtype == 'ElasticNet':
			sub_model[i][1] = get_ElasticNet_model(rs)
		elif mtype == 'SVR':
			sub_model[i][1] = get_SVR_model(rs)
		elif mtype == 'ABR':
			sub_model[i][1] = get_ABR_model(rs)
		elif mtype == 'SGD':
			sub_model[i][1] = get_SGD_model(rs)

		sub_model[i][1].fit(X_train, y_train)
		print('Model', mtype)
		print_result('Tra', sub_model[i][1], X_train, y_train, y_min, y_max)
		rmse = print_result('Val', sub_model[i][1], X_test , y_test , y_min, y_max)
		weights += [np.power(rmse, -2)]
		# weights += [rmse]

	print(weights)
	model = VotingRegressor(sub_model, weights = None, n_jobs = 8, verbose = 1)

	model.fit(X_train, y_train.squeeze())
	# model.fit(X_train, np.log(y_train.squeeze()) )

	print('Model VotingRegressor')
	print_result('Tra', model, X_train, y_train, y_min, y_max)
	print_result('Val', model, X_test , y_test , y_min, y_max)
	print('random_state', rs)
#   print(model.best_params_)

	test = np.load('X_test_115.npy').astype(np.float)
	add_dim(test, len(test[0]) * dim)
	test = np.hstack((test, np.ones((len(test), 1))))
	predict = model.predict(test)
	predict = np.exp(np.clip(predict, y_min, y_max))

	save_data(predict)

	return

if __name__ == '__main__':
	main()