from sklearn.model_selection import train_test_split
import os
import sys
import csv
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer
import scipy.special as sc
from nltk.tokenize import RegexpTokenizer
import re
import numpy as np
from sklearn.svm import SVR
import time
from scipy.sparse import csr_matrix
from sklearn import metrics
import copy
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from scipy.sparse import hstack
from sklearn import linear_model
from scipy.stats import pearsonr, spearmanr, kendalltau

import torch
print(os.curdir)
csv.field_size_limit(sys.maxsize)

start = time.time()

sec = sys.argv[1]

df = pd.read_csv("sorted_"+sec+".csv")

bv = sys.argv[2]

hist = sys.argv[3]

train_text, rem_text, train_hist, rem_hist, train_labels, rem_labels = train_test_split(df['mda'],
    df['prev_'+bv], 
    df[bv],
    shuffle=False,
    train_size=0.8) 

valid_text, test_text, valid_hist, test_hist, valid_labels, test_labels = train_test_split(
	rem_text,
	rem_hist,
	rem_labels,
    shuffle=False,
	test_size=0.5
)

X_train = train_text.astype('U').values
X_valid = valid_text.astype('U').values
X_test = test_text.astype('U').values

y_train = train_labels.to_numpy()
y_valid = valid_labels.to_numpy()
y_test = test_labels.to_numpy()

hist_train = train_hist.to_numpy()
hist_valid = valid_hist.to_numpy()
hist_test = test_hist.to_numpy()

'''
X_vect = y_train
xmax, xmin = X_vect.max(), X_vect.min()
X_vect = (X_vect-xmin)/(xmax-xmin) 
y_train = X_vect 

X_vect = y_test
X_vect = (X_vect-xmin)/(xmax-xmin)  
y_test = X_vect 

X_vect = hist_train
xmax, xmin = X_vect.max(), X_vect.min()  
X_vect = (X_vect-xmin)/(xmax-xmin) 
hist_train = X_vect  

X_vect = hist_test
X_vect = (X_vect-xmin)/(xmax-xmin) 
hist_test = X_vect  
'''

vect = CountVectorizer()
X_train = vect.fit_transform(X_train)
X_train = np.log1p(X_train)
X_valid = vect.transform(X_valid)
X_valid = np.log1p(X_valid)
X_test = vect.transform(X_test)
X_test = np.log1p(X_test)

#print("LENGTHS: ", X_train.shape[0], X_test.shape[0], len(y_train), len(y_test))
#sys.exit(0)
#print("X_train: ", X_train.todense().shape[0], X_test.todense().shape[0])


a = pd.DataFrame(hist_train)
a = a.reset_index(drop=True)
b = pd.DataFrame(hist_valid)
b = b.reset_index(drop=True)
c = pd.DataFrame(hist_test)
c = c.reset_index(drop=True)

d = pd.DataFrame(X_train.todense())
d = d.reset_index(drop=True)
e = pd.DataFrame(X_valid.todense())
e = e.reset_index(drop=True)
f = pd.DataFrame(X_test.todense())
f = f.reset_index(drop=True)


#print(vect.get_feature_names())
#print("X_train: ", type(X_train), np.isnan(X_train.todense().data).any())
#print("X_test: ", type(X_test), np.isnan(X_test.data).any())

#X_train1 = pd.DataFrame(X_train.toarray(), columns=vect.get_feature_names())
#X_train2 = pd.concat([X_train1, pd.DataFrame(hist_train)], axis=1)

#X_test1 = pd.DataFrame(X_test.toarray(), columns=vect.get_feature_names()) 
#X_test2 = pd.concat([X_test1, pd.DataFrame(hist_test)], axis=1)   

#X_test_concat = csr_matrix(pd.concat([pd.DataFrame(X_test.todense()), pd.DataFrame(hist_test)], axis=1))
#X_train_concat = csr_matrix(pd.concat([pd.DataFrame(X_train.todense()), pd.DataFrame(hist_train)], axis=1))
#X_test_concat = csr_matrix(pd.concat([pd.DataFrame(X_test.todense()), pd.DataFrame(hist_test)], axis=0))
if hist == "hist":
    X_train_concat = csr_matrix(pd.concat([d,a], axis=1))
    X_valid_concat = csr_matrix(pd.concat([e,b], axis=1))
    X_test_concat = csr_matrix(pd.concat([f,c], axis=1))
elif hist == "nohist":
    X_train_concat = csr_matrix(d)
    X_valid_concat = csr_matrix(e)
    X_test_concat = csr_matrix(f)

#print("X_train: ", X_train)
#print("X_train_concat: ", X_train_concat)
X_train_concat = X_train_concat.toarray().tolist()
X_valid_concat = X_valid_concat.toarray().tolist()
X_test_concat = X_test_concat.toarray().tolist()
#print("train con: ", X_train_concat, X_train_concat.shape)
#print("test con: ", len(X_test_concat))
#print(torch.tensor(X_train_concat[11]).unsqueeze(0))
#print(torch.tensor(X_valid_concat[11]).unsqueeze(0).dtype)
#print(torch.tensor(X_test_concat[11]).unsqueeze(0).dtype)
#print(np.isnan(X_train_concat.data).any())
#print(np.isnan(X_test_concat.data).any())

train_hist = torch.tensor(train_hist.tolist())
train_y = torch.tensor(train_labels.tolist())

valid_hist = torch.tensor(valid_hist.tolist())
valid_y = torch.tensor(valid_labels.tolist())

test_hist = torch.tensor(test_hist.tolist())
test_y = torch.tensor(test_labels.tolist())

print(train_hist[10])
print(train_hist.dtype)
print(train_hist.shape)


#numeric
#X_train = np.asarray(hist_train).reshape(-1, 1)
#X_test = np.asarray(hist_test).reshape(-1, 1)

lr = LinearRegression()
kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
svr = SVR(kernel='rbf', C=0.1, epsilon=0.0001) #linear')

valid_mses = []
test_mses = []
methods = [ 'svr', 'lr', 'kr']
models_list = [ svr, lr, kr]

for model in models_list:
    model.fit(X_train_concat, y_train.reshape(-1,1))

    preds = model.predict(X_valid_concat)
    mse = mean_squared_error(y_valid.reshape(-1,1), preds)
    valid_mses.append(mse)
    #print(model, mse, " valid")

    preds = model.predict(X_test_concat)
    mse = mean_squared_error(y_test.reshape(-1,1), preds)
    test_mses.append(mse)
    #print(model, mse, " test")

print(sec+"-"+bv+"-"+hist)
print(str(test_mses[valid_mses.index(min(valid_mses))])+"---"+
methods[valid_mses.index(min(valid_mses))]+"---"+str(min(valid_mses)))
print("Total execution time: ", time.time() - start)