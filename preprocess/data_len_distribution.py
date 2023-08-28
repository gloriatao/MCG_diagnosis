# sort data
import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import torch
from matplotlib import cm
viridis = cm.get_cmap('jet', 256)

labelpath = '/media/cygzz/data/rtao/projects/MCG-NC/data/data001/Labels.xlsx'
datapath = '/media/cygzz/data/rtao/projects/MCG-NC/data/data001/2023-7-21MCG DATA'
outpath = '/media/cygzz/data/rtao/projects/MCG-NC/data/data001/vis/sheet1'
out_pickle = '/media/cygzz/data/rtao/projects/MCG-NC/data/data001/data_pickle'
MCG_data = dict()
xls = pd.ExcelFile(labelpath)
df1 = pd.read_excel(xls, 's1')
df2 = pd.read_excel(xls, 's2')
df3 = pd.read_excel(xls, 's3')

QRSs, Ts = [], []
#----------sheet1------------
one_df = df1
id = one_df['心磁图编号']
is_ischemia = one_df['是否缺血']
hospital = one_df['医院来源']
diagosis = one_df['诊断来源']
loc_q_peak = one_df['loc_q_peak']
loc_r_peak = one_df['loc_r_peak']
loc_s_peak = one_df['loc_s_peak']
loc_t_onset = one_df['loc_t_onset']
loc_t_peak = one_df['loc_t_peak']
loc_t_end = one_df['loc_t_end']
for i in range(id.shape[0]):    
    QRSs.append(loc_s_peak.values[i]-loc_q_peak.values[i])
    Ts.append(loc_t_end.values[i]-loc_t_onset.values[i])
    
    if loc_s_peak.values[i]-loc_q_peak.values[i] > 100:
        print(id[i],'-long QRS-' , loc_s_peak.values[i]-loc_q_peak.values[i])
        
    if loc_s_peak.values[i]-loc_q_peak.values[i] < 50:
        print(id[i],'-short QRS-' , loc_s_peak.values[i]-loc_q_peak.values[i])
                
    if loc_t_end.values[i]-loc_t_onset.values[i] > 200:
        print(id[i],'-long T-' , loc_t_end.values[i]-loc_t_onset.values[i])
        
    if loc_t_end.values[i]-loc_t_onset.values[i] < 60:
        print(id[i],'-short T-' , loc_t_end.values[i]-loc_t_onset.values[i])
    
print('QRS_stas:', max(QRSs), min(QRSs), np.mean(np.array(QRSs)))
print('Ts_stas:', max(Ts), min(Ts), np.mean(np.array(Ts)))

# QRS_stas: 135 37 55.12894560107455 ---100 [20]
# Ts_stas: 301 42 156.97447951645398 ---200 [40]
    

