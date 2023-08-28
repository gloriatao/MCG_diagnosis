import os
import random
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch.nn.functional as F

def resize3D(image, size=None, scale=None, mode='trilinear'):
    image = torch.tensor(image, dtype=torch.float32)
    image = image[None,None,:,:,:]
    if scale:
        image = F.interpolate(image, scale_factor=scale, mode=mode)
    elif size:
        image = F.interpolate(image, size=size, mode=mode)
    else:
        print('wrong params')
    image = image.squeeze().numpy()
    return image


class load_val(Dataset):
    def __init__(self, fold=0, type=None):
        self.fold_index = fold
        _, val_list_NI, _, val_list_SPECT, _, val_list_CTA = self.load_data(fold)
        self.type = type
        
        self.val_list_NI = val_list_NI
        self.val_list_SPECT = val_list_SPECT
        self.val_list_CTA = val_list_CTA
        
        if type == 'CTA':
            self.val_list = val_list_CTA
        elif type == 'SPECT':
            self.val_list = val_list_SPECT
        elif type == 'NI':
            self.val_list = val_list_NI    
        self.nSamples = len(self.val_list)
        print("done init")

    def load_data(self, fold):
        Non_Ischemia_path = '/media/cygzz/data/rtao/projects/MCG-NC/data/data001/data_pickle/Non_Ischemia.pickle'
        Ischamia_CTA_path = '/media/cygzz/data/rtao/projects/MCG-NC/data/data001/data_pickle/Ischamia_CTA.pickle'
        Ischamia_SPECT_path = '/media/cygzz/data/rtao/projects/MCG-NC/data/data001/data_pickle/Ischamia_SPECT.pickle'
        fold_path = '/media/cygzz/data/rtao/projects/MCG-NC/data/data001/data_pickle/5fold_CV_index0802.pickle'
        with open(Non_Ischemia_path, 'rb') as f:  # run this to load from pickle
            Non_Ischemia_data = pickle.load(f)
        f.close()
        with open(Ischamia_CTA_path, 'rb') as f:  # run this to load from pickle
            Ischamia_CTA_data = pickle.load(f)
        f.close()
        with open(Ischamia_SPECT_path, 'rb') as f:  # run this to load from pickle
            Ischamia_SPECT_data = pickle.load(f)
        f.close()
        with open(fold_path, 'rb') as f:  # run this to load from pickle
            fold_index_all = pickle.load(f)
        f.close()
        train_index_NI = fold_index_all['Non_Ischemia_fold'][fold]['train_index']
        test_index_NI = list(fold_index_all['Non_Ischemia_fold'][fold]['test_index'])
        
        train_index_SPECT = fold_index_all['Ischemia_SPECT_fold'][fold]['train_index']
        train_index_SPECT = train_index_SPECT + train_index_SPECT + train_index_SPECT # over-sampling
        test_index_SPECT = list(fold_index_all['Ischemia_SPECT_fold'][fold]['test_index'])
        
        train_index_CTA = fold_index_all['Ischemia_CTA_fold'][fold]['train_index']
        test_index_CTA = list(fold_index_all['Ischemia_CTA_fold'][fold]['test_index'])
        
        train_list_NI, val_list_NI = [], []
        for i in (train_index_NI):
            train_list_NI.append(Non_Ischemia_data[list(Non_Ischemia_data.keys())[i]])
        # print('Train: done loading Non_Ischemia cases:', len(train_index_NI))
        for i in (test_index_NI):
            val_list_NI.append(Non_Ischemia_data[list(Non_Ischemia_data.keys())[i]])
        # print('Val: done loading Non_Ischemia cases:', len(test_index_NI))
        
 
        train_list_SPECT, val_list_SPECT = [], []
        for i in (train_index_SPECT):
            train_list_SPECT.append(Ischamia_SPECT_data[list(Ischamia_SPECT_data.keys())[i]])
        # print('Train: done loading Ischemia_SPECT cases:', len(train_index_SPECT))
        for i in (test_index_SPECT):
            val_list_SPECT.append(Ischamia_SPECT_data[list(Ischamia_SPECT_data.keys())[i]])
        # print('Val: done loading Ischemia_SPECT cases:', len(test_index_SPECT)) 
        

        train_list_CTA, val_list_CTA = [], []
        for i in (train_index_CTA):
            train_list_CTA.append(Ischamia_CTA_data[list(Ischamia_CTA_data.keys())[i]])
        # print('Train: done loading Ischemia_CTA cases:', len(train_index_CTA))
        for i in (test_index_CTA):
            val_list_CTA.append(Ischamia_CTA_data[list(Ischamia_CTA_data.keys())[i]])
        # print('Val: done loading Ischemia_CTA cases:', len(test_index_CTA)) 

        return train_list_NI, val_list_NI, train_list_SPECT, val_list_SPECT, train_list_CTA, val_list_CTA

    def __len__(self):
        return self.nSamples

    def plot_waveform(self, waveforms, time_stamp, outpath):
        from matplotlib import cm
        import matplotlib.pyplot as plt
        viridis = cm.get_cmap('jet', 256) 
        
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
            
        axs.plot(time_stamp[0], min(waveforms[:,time_stamp[0]]), 'ro')
        axs.plot(time_stamp[1], max(waveforms[:,time_stamp[1]]), 'go')
        axs.plot(time_stamp[2], min(waveforms[:,time_stamp[2]]), 'ro')
        axs.plot(time_stamp[3], max(waveforms[:,time_stamp[3]]), 'go')
        
        axs.plot(time_stamp[4], max(waveforms[:,time_stamp[4]]), 'go')
        axs.plot(time_stamp[5], max(waveforms[:,time_stamp[5]]), 'bo')
            
        plt.savefig(outpath)
                
        plt.clf()
        plt.close()

    def preprocessing(self, mcg_data):
        qrs_len, t_len = 100, 200
        waveforms = mcg_data['waveforms']
        # nomalize        
        tmp = list(np.multiply(waveforms, waveforms).sum(axis=1))
        mglb = np.array([np.sqrt(t) for t in tmp])
        waveforms = (waveforms-mglb.min())/mglb.max()
    
        q_loc = mcg_data['loc_q_peak'] - 10
        r_loc = mcg_data['loc_r_peak']
        s_loc = mcg_data['loc_s_peak'] + 10
        t_onset_loc = mcg_data['loc_t_onset']
        t_offset_loc = mcg_data['loc_t_end']
        t_loc = mcg_data['loc_t_peak']     
        # vis
        # tstamp = [q_loc, r_loc, s_loc, t_onset_loc, t_offset_loc, t_loc]
        # self.plot_waveform(waveforms, tstamp, '/media/cygzz/data/rtao/projects/MCG-NC/debug/aug.png')
        waveforms3d = waveforms.reshape(mcg_data['waveforms'].shape[0],6,6)
        qrs = waveforms3d[q_loc:s_loc, :, :]
        t = waveforms3d[t_onset_loc:t_offset_loc, :, :]
        # scale
        qrs_scale = resize3D(qrs, size=(qrs_len, 6, 6), mode='trilinear')
        t_scale = resize3D(t, size=(t_len, 6, 6), mode='trilinear')
        
        return qrs_scale, t_scale

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        data_mcg = self.val_list[index]       
        qrs_input, t_input, is_ischemia = [], [], []
        ischemia_intensity, ischemia_area = [],[]
        xin_jian, qian_bi, jiange_bi, xia_bi, ce_bi, id = [], [], [], [], [], []
        LAD, LCX, RCA = [], [], []
        # NI
        if self.type == 'NI':                   
            QRS, T = self.preprocessing(data_mcg)
            qrs_input.append(QRS)
            t_input.append(T)
            is_ischemia=0
            ischemia_intensity=-np.inf
            ischemia_area=-np.inf
            xin_jian=-np.inf
            qian_bi=-np.inf
            jiange_bi=-np.inf
            xia_bi=-np.inf
            ce_bi=-np.inf
            id=data_mcg['id']
            LAD=-np.inf
            LCX=-np.inf
            RCA=-np.inf
            
        # SPECT
        if self.type == 'SPECT':                  
            QRS, T = self.preprocessing(data_mcg)
            qrs_input.append(QRS)
            t_input.append(T)
            is_ischemia=1
            ischemia_intensity=data_mcg['ischemia_intensity']/3.0
            ischemia_area=data_mcg['ischemia_area']/100.0
            xin_jian=data_mcg['xin_jian']
            qian_bi=data_mcg['qian_bi']
            jiange_bi=data_mcg['jiange_bi']
            xia_bi=data_mcg['xia_bi']
            ce_bi=data_mcg['ce_bi']
            id=data_mcg['id']
            LAD=-np.inf
            LCX=-np.inf
            RCA=-np.inf       

        # CTA
        if self.type == 'CTA':        
            QRS, T = self.preprocessing(data_mcg)
            qrs_input.append(QRS)
            t_input.append(T)
            is_ischemia=1
            ischemia_intensity=-np.inf
            ischemia_area=-np.inf
            xin_jian=-np.inf
            qian_bi=-np.inf
            jiange_bi=-np.inf
            xia_bi=-np.inf
            ce_bi=-np.inf
            id=data_mcg['id']          
            LAD=data_mcg['LAD']
            LCX=data_mcg['LCX']
            RCA=data_mcg['RCA']         

        input =  {'QRS':qrs_input, 'T':t_input, 'is_ischemia':is_ischemia, 
                  'ischemia_intensity':ischemia_intensity, 'ischemia_area':ischemia_area,
                  'xin_jian':xin_jian, 'qian_bi':qian_bi,'jiange_bi':jiange_bi,
                  'xia_bi':xia_bi, 'ce_bi':ce_bi, 'LAD':LAD, 'LCX':LCX,
                  'RCA':RCA, 'id':id,}   

        input['QRS'] = torch.tensor(input['QRS']).permute(0,2,3,1)
        input['T'] = torch.tensor(input['T']).permute(0,2,3,1)
        return input



def test():
    loader = load_val(type='NI')
    loader.__getitem__(1)

# test()