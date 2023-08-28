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

def read_txt(data_path):
    input = pd.read_csv(data_path, sep='\t', header=None,)
    waveforms = input.values[:, 1:]
    waveforms_2d = waveforms.reshape(waveforms.shape[0],6,6)
    return waveforms, waveforms_2d

def plot_waveform(waveforms, time_stamp, outpath): 
    waveforms = waveforms.T  
    x = np.arange(waveforms.shape[-1])
    peak = waveforms[:,time_stamp[0]]
    index = np.argsort(peak)
    fig, axs = plt.subplots(1, 1)
    for j in range(36):
        x = np.arange(waveforms.shape[-1])
        a = 46
        b = j+5
        cm =(viridis(b/a)[0],viridis(b/a)[1],viridis(b/a)[2], 0.7)
        axs.plot(x, waveforms[index[j], :], color=cm)
        # axs.set_axis_off()
        fig.tight_layout()
        
    axs.plot(time_stamp[0], min(waveforms[:,time_stamp[0]]), 'bo')
    axs.plot(time_stamp[1], max(waveforms[:,time_stamp[1]]), 'bo')
    axs.plot(time_stamp[2], min(waveforms[:,time_stamp[2]]), 'bo')
    axs.plot(time_stamp[3], max(waveforms[:,time_stamp[3]]), 'bo')
    
    axs.plot(time_stamp[4], max(waveforms[:,time_stamp[4]]), 'go')
    axs.plot(time_stamp[5], max(waveforms[:,time_stamp[5]]), 'go')
        
    plt.savefig(outpath)
            
    plt.clf()
    plt.close()

labelpath = 'projects/MCG-NC/data/data001/Labels.xlsx'
datapath = 'projects/MCG-NC/data/data001/2023-7-21MCG DATA'
outpath = 'projects/MCG-NC/data/data001/vis/sheet1'
out_pickle = 'projects/MCG-NC/data/data001/data_pickle'

MCG_data = dict()

xls = pd.ExcelFile(labelpath)
df1 = pd.read_excel(xls, 's1')
df2 = pd.read_excel(xls, 's2')
df3 = pd.read_excel(xls, 's3')

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
    print('All_data:', i)
    one_data = dict()
    one_id = id.values[i]
    one_data['id'] = one_id
    one_data['is_ischemia'] = is_ischemia.values[i]
    one_data['hospital'] = hospital.values[i]
    one_data['diagosis'] = diagosis.values[i]
    one_data['loc_q_peak'] = loc_q_peak.values[i]
    one_data['loc_r_peak'] = loc_r_peak.values[i]
    one_data['loc_s_peak'] = loc_s_peak.values[i]
    one_data['loc_t_onset'] = loc_t_onset.values[i]
    one_data['loc_t_peak'] = loc_t_peak.values[i]
    one_data['loc_t_end'] = loc_t_end.values[i]
    
    one_data['ischemia_intensity'] = None  
    one_data['ischemia_area'] = None
    one_data['xin_jian'] = None
    one_data['qian_bi'] = None
    one_data['jiange_bi']= None
    one_data['xia_bi'] = None
    one_data['ce_bi'] = None

    one_data['LAD'] = None
    one_data['LCX'] = None
    one_data['RCA'] = None    
    
    
    data_path = os.path.join(datapath, one_id+'.txt')
    waveforms, waveforms_2d = read_txt(data_path)
    
    one_data['waveforms'] = waveforms 
    # one_data['waveforms_2d'] = waveforms_2d 
    
    MCG_data[one_id] = one_data
    
    # time_stamp = [int(loc_q_peak.values[i]),int(loc_r_peak.values[i]), 
    #               int(loc_s_peak.values[i]),int(loc_t_peak.values[i]),
    #               int(loc_t_onset.values[i]), int(loc_t_end.values[i])]
    
    # out_path_one = os.path.join(outpath, one_id + '.png')
    # plot_waveform(waveforms, time_stamp, out_path_one)
    

#----------sheet2------------
SPECT_data = dict()
one_df = df2
id = one_df['心磁图编号']

ischemia_intensity = one_df['缺血程度']
ischemia_area = one_df['缺血面积/%']
xin_jian = one_df['心尖']
qian_bi = one_df['前壁']
jiange_bi= one_df['间隔壁']
xia_bi = one_df['下壁']
ce_bi = one_df['侧壁']
SPECT_ID = []
for i in range(id.shape[0]):
    print('SPECT_data:', i)
    one_id = id.values[i] 
    one_data = MCG_data[one_id]  
    one_data['ischemia_intensity'] = ischemia_intensity.values[i] 
    one_data['ischemia_area'] = ischemia_area.values[i] 
    one_data['xin_jian'] = xin_jian.values[i] 
    one_data['qian_bi'] = qian_bi.values[i] 
    one_data['jiange_bi'] = jiange_bi.values[i] 
    one_data['xia_bi'] = xia_bi.values[i] 
    one_data['ce_bi'] = ce_bi.values[i] 
    SPECT_data[one_id] = one_data
    SPECT_ID.append(one_id)
    
# #----------sheet3------------
CTA_data = dict()
one_df = df3
id = one_df['心磁图编号']
LAD = one_df['LAD-左前降支']
LCX = one_df['LCX-左回旋支']
RCA = one_df['RCA-右冠脉支']
CTA_ID = []
for i in range(id.shape[0]):
    print('CTA_data:', i)
    one_id = id.values[i] 
    one_data = MCG_data[one_id]  
    one_data['LAD'] = LAD.values[i] 
    one_data['LCX'] = LCX.values[i] 
    one_data['RCA'] = RCA.values[i] 
    CTA_data[one_id] = one_data
    CTA_ID.append(one_id)
    
# #----------remain------------   
id = df1['心磁图编号'] 
Others_dict = dict()
for i in range(id.shape[0]):
    print('Other:', i)
    one_id = id.values[i] 
    one_data = MCG_data[one_id]
    if (one_id in CTA_ID) or (one_id in SPECT_ID):
        pass
    else:
        Others_dict[one_id] = one_data
        

with open('projects/MCG-NC/data/data001/data_pickle/Others.pickle', 'wb') as f:
    pickle.dump(Others_dict, f)
f.close()

with open('projects/MCG-NC/data/data001/data_pickle/CTA.pickle', 'wb') as f:
    pickle.dump(CTA_data, f)
f.close()

with open('projects/MCG-NC/data/data001/data_pickle/SPECT.pickle', 'wb') as f:
    pickle.dump(SPECT_data, f)
f.close()

with open('projects/MCG-NC/data/data001/data_pickle/All.pickle', 'wb') as f:
    pickle.dump(MCG_data, f)
f.close()

    

