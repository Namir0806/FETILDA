import os
import sys
import csv
import pandas as pd

df = pd.DataFrame()

sec1_roa = []
sec1_eps = []
sec1_roa_hist = []
sec1_eps_hist = []
sec7_roa = []
sec7_eps = []
sec7_roa_hist = []
sec7_eps_hist = []

method = ['bert_510', 'finbert_510', 'lf_510', 'lf4094_4094']
method_type = ['mean', 'max', 'bilstm']

for i in method:
    for j in method_type:
        fname = 'mse_' + i + '_sec1A_roa_' + j + '.txt'
        if os.path.exists(fname):
            df1 = pd.read_csv(fname, header=None, nrows=1, index_col=None)
            sec1_roa.append(df1[0][0])
        else:
            sec1_roa.append("")

for i in method:
    for j in method_type:
        fname = 'mse_' + i + '_sec1A_eps_' + j + '.txt'
        if os.path.exists(fname):
            df1 = pd.read_csv(fname, header=None, nrows=1, index_col=None)
            sec1_eps.append(df1[0][0])
        else:
            sec1_eps.append("")

for i in method:
    for j in method_type:
        fname = 'mse_' + i + '_sec1A_roa_' + j + '_hist.txt'
        if os.path.exists(fname):
            df1 = pd.read_csv(fname, header=None, nrows=1, index_col=None)
            sec1_roa_hist.append(df1[0][0])
        else:
            sec1_roa_hist.append("")

for i in method:
    for j in method_type:
        fname = 'mse_' + i + '_sec1A_eps_' + j + '_hist.txt'
        if os.path.exists(fname):
            df1 = pd.read_csv(fname, header=None, nrows=1, index_col=None)
            sec1_eps_hist.append(df1[0][0])
        else:
            sec1_eps_hist.append("")

for i in method:
    for j in method_type:
        fname = 'mse_' + i + '_sec7_roa_' + j + '.txt'
        if os.path.exists(fname):
            df1 = pd.read_csv(fname, header=None, nrows=1, index_col=None)
            sec7_roa.append(df1[0][0])
        else:
            sec7_roa.append("")

for i in method:
    for j in method_type:
        fname = 'mse_' + i + '_sec7_eps_' + j + '.txt'
        if os.path.exists(fname):
            df1 = pd.read_csv(fname, header=None, nrows=1, index_col=None)
            sec7_eps.append(df1[0][0])
        else:
            sec7_eps.append("")

for i in method:
    for j in method_type:
        fname = 'mse_' + i + '_sec7_roa_' + j + '_hist.txt'
        if os.path.exists(fname):
            df1 = pd.read_csv(fname, header=None, nrows=1, index_col=None)
            sec7_roa_hist.append(df1[0][0])
        else:
            sec7_roa_hist.append("")

for i in method:
    for j in method_type:
        fname = 'mse_' + i + '_sec7_eps_' + j + '_hist.txt'
        if os.path.exists(fname):
            df1 = pd.read_csv(fname, header=None, nrows=1, index_col=None)
            sec7_eps_hist.append(df1[0][0])
        else:
            sec7_eps_hist.append("")


df['sec1_roa']= sec1_roa
df['sec1_eps']= sec1_eps
df['sec1_roa_hist']= sec1_roa_hist
df['sec1_eps_hist']= sec1_eps_hist
df['sec7_roa']= sec7_roa
df['sec7_eps']= sec7_eps
df['sec7_roa_hist']= sec7_roa_hist
df['sec7_eps_hist']= sec7_eps_hist


df.to_csv("results_table_0.007.csv", index=True)
