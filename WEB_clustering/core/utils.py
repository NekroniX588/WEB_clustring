# -*- coding: utf-8 -*-
import numpy as np

def get_F_example(data, a, target=[]):#add attribute of norm on consts
	if len(target)==0:
		F = []
		for cur_point in data:
			# cur_point = np.append(cur_point, np.sum([( np.linalg.norm(point[1:] - cur_point[1:]) + a)**(-2) for point in data]))
			cur_point = np.append(cur_point, np.sum([a/(np.linalg.norm(point[1:] - cur_point[1:])**2 + a) for point in data])/(data.shape[0]))
			# cur_point = np.append(cur_point, np.sum([(np.linalg.norm(point[1:] - cur_point[1:])**2 + a)**(-1) for point in data]))
			F.append(cur_point)
		return F
	else:
		return np.sum([a/(np.linalg.norm(point[1:] - target[1:])**2 + a) for point in data])/len(data)

def get_F(data, a, target=[]):
	if len(target)==0:
		F = []
		for i, cur_point in enumerate(data):
		    line = [( np.linalg.norm(point[1:] - cur_point[1:])**2 + a)**(-1) for point in data]
		    cur_point = np.append(cur_point, np.sum(line[:i] + line[i+1:])/(data.shape[0]))
		    F.append(cur_point)
		return F
	else:
		return np.sum([(np.linalg.norm(point[1:] - target[1:]) + a)**(-2) for point in data])