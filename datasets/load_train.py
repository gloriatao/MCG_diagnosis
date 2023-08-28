import os
import cv2
import random
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch.nn.functional as F

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2
import matplotlib.pyplot as plt
colormap = plt.get_cmap('jet')


def get_spatial_transform(patch_size, angle=45):
    tr_transforms = []
    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i//2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- angle / 360. * 2 * np.pi, angle / 360. * 2 * np.pi),
            angle_y=(- 10 / 360. * 2 * np.pi, -10 / 360. * 2 * np.pi),
            angle_z=(- 10 / 360. * 2 * np.pi, 10 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='mirror',
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=0, order_data=3,
            random_crop=False,
            p_rot_per_sample=0.5, p_el_per_sample=0.5, p_scale_per_sample=0.5
        )
    )
    # now we mirror along all axes
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


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


class load_train(Dataset):
    def __init__(self, fold=0, bs=2):
        self.fold_index = fold
        train_list_NI, val_list_NI, train_list_SPECT, val_list_SPECT, train_list_CTA, val_list_CTA = self.load_data(fold)
        
        self.train_list_NI = train_list_NI
        self.val_list_NI = val_list_NI
        self.train_list_SPECT = train_list_SPECT
        self.val_list_SPECT = val_list_SPECT
        self.train_list_CTA = train_list_CTA
        self.val_list_CTA = val_list_CTA
        self.bs = bs
        
        self.transform_spatial_t = get_spatial_transform((200, 6, 6), angle=30)
        # self.transform_spatial_qrs = get_spatial_transform((100, 6, 6), angle=10)
        
        self.nSamples = len(self.train_list_SPECT)
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
        print('Train: done loading Non_Ischemia cases:', len(train_index_NI))
        for i in (test_index_NI):
            val_list_NI.append(Non_Ischemia_data[list(Non_Ischemia_data.keys())[i]])
        print('Val: done loading Non_Ischemia cases:', len(test_index_NI))
        
 
        train_list_SPECT, val_list_SPECT = [], []
        for i in (train_index_SPECT):
            train_list_SPECT.append(Ischamia_SPECT_data[list(Ischamia_SPECT_data.keys())[i]])
        print('Train: done loading Ischemia_SPECT cases:', len(train_index_SPECT))
        for i in (test_index_SPECT):
            val_list_SPECT.append(Ischamia_SPECT_data[list(Ischamia_SPECT_data.keys())[i]])
        print('Val: done loading Ischemia_SPECT cases:', len(test_index_SPECT)) 
        

        train_list_CTA, val_list_CTA = [], []
        for i in (train_index_CTA):
            train_list_CTA.append(Ischamia_CTA_data[list(Ischamia_CTA_data.keys())[i]])
        print('Train: done loading Ischemia_CTA cases:', len(train_index_CTA))
        for i in (test_index_CTA):
            val_list_CTA.append(Ischamia_CTA_data[list(Ischamia_CTA_data.keys())[i]])
        print('Val: done loading Ischemia_CTA cases:', len(test_index_CTA)) 

        return train_list_NI, val_list_NI, train_list_SPECT, val_list_SPECT, train_list_CTA, val_list_CTA

    def __len__(self):
        return self.nSamples//self.bs


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


    def plot_spatial(self, data, outpath):
        data = data - np.min(data)
        data = data/np.max(data)
        data = (colormap(data) * 255).astype(np.uint8)[:,:,:3]
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        data = cv2.resize(data, (128, 128))
        cv2.imwrite(outpath, data)
        return       
        

    def preprocessing(self, mcg_data):
        qrs_len, t_len = 100, 200
        waveforms = mcg_data['waveforms']
        # nomalize        
        tmp = list(np.multiply(waveforms, waveforms).sum(axis=1))
        mglb = np.array([np.sqrt(t) for t in tmp])
        waveforms = (waveforms-mglb.min())/mglb.max()
    
        q_loc = mcg_data['loc_q_peak'] - random.randint(10, 25)
        r_loc = mcg_data['loc_r_peak']
        s_loc = mcg_data['loc_s_peak'] + random.randint(10, 25)
        t_onset_loc = mcg_data['loc_t_onset'] - random.randint(-20, 20)
        t_offset_loc = mcg_data['loc_t_end'] + random.randint(-20, 20)
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
        
        # # aug-t
        # # self.plot_spatial(t_scale[100,:,:], '/media/cygzz/data/rtao/projects/MCG-NC/debug/t100.png')
        # data = np.zeros((1, 1, 200, 6, 6), dtype=np.float32)
        # seg = np.zeros((1, 1, 200, 6, 6), dtype=np.float32)
        # data[0][0] = t_scale
        # patient_dict = {'data':data, 'seg':seg}
        # patient_dict = self.transform_spatial_t(**patient_dict)
        # t_scale_rot  = patient_dict['data'][0][0]
        # # self.plot_spatial(t_scale_rot[100,:,:], '/media/cygzz/data/rtao/projects/MCG-NC/debug/t100_aug.png')
        
        # # aug-qrs
        # # self.plot_spatial(qrs_scale[50,:,:], '/media/cygzz/data/rtao/projects/MCG-NC/debug/qrs50.png')
        # data = np.zeros((1, 1, 100, 6, 6), dtype=np.float32)
        # seg = np.zeros((1, 1, 100, 6, 6), dtype=np.float32)
        # data[0][0] = qrs_scale
        # patient_dict = {'data':data, 'seg':seg}
        # patient_dict = self.transform_spatial_qrs(**patient_dict)
        # qrs_scale_rot  = patient_dict['data'][0][0]
        # # self.plot_spatial(qrs_scale_rot[50,:,:], '/media/cygzz/data/rtao/projects/MCG-NC/debug/qrs50_aug.png')
        return qrs_scale, t_scale

    def shuffle_list(self, input, indx):
        random.shuffle(indx)
        input_new = dict()
        for i, k in enumerate(input.keys()):
            old_list = input[k]
            new_list = [old_list[item] for item in indx]
            if k != 'id':
                input_new[k] = torch.tensor(np.stack(new_list.copy()))
        return input_new


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        data_SPECT = random.sample(self.train_list_SPECT, self.bs)    #self.train_list_SPECT[index]
        data_CTA = random.sample(self.train_list_CTA, self.bs)
        data_NI = random.sample(self.train_list_NI, self.bs*2)
        
        cnt = 0
        indx, qrs_input, t_input, is_ischemia = [], [], [], []
        ischemia_intensity, ischemia_area = [],[]
        xin_jian, qian_bi, jiange_bi, xia_bi, ce_bi, id = [], [], [], [], [], []
        LAD, LCX, RCA = [], [], []
        SPECT_idx, CTA_idx = [], []
        # SPECT
        for i in range(self.bs):       
            indx.append(cnt) 
            cnt += 1
            
            QRS, T = self.preprocessing(data_SPECT[i])
            qrs_input.append(QRS)
            t_input.append(T)
            is_ischemia.append(1)
            ischemia_intensity.append(data_SPECT[i]['ischemia_intensity']/3.0)
            ischemia_area.append(data_SPECT[i]['ischemia_area']/100.0)
            xin_jian.append(data_SPECT[i]['xin_jian'])
            qian_bi.append(data_SPECT[i]['qian_bi'])
            jiange_bi.append(data_SPECT[i]['jiange_bi'])
            xia_bi.append(data_SPECT[i]['xia_bi'])
            ce_bi.append(data_SPECT[i]['ce_bi'])
            id.append(data_SPECT[i]['id'])
            
            LAD.append(-np.inf)
            LCX.append(-np.inf)
            RCA.append(-np.inf)
            
            SPECT_idx.append(1)
            CTA_idx.append(0)
            
        # CTA
        for i in range(self.bs):       
            indx.append(cnt) 
            cnt += 1
            
            QRS, T = self.preprocessing(data_CTA[i])
            qrs_input.append(QRS)
            t_input.append(T)
            is_ischemia.append(1)
            ischemia_intensity.append(-np.inf)
            ischemia_area.append(-np.inf)
            xin_jian.append(-np.inf)
            qian_bi.append(-np.inf)
            jiange_bi.append(-np.inf)
            xia_bi.append(-np.inf)
            ce_bi.append(-np.inf)
            id.append(-np.inf)
            
            LAD.append(data_CTA[i]['LAD'])
            LCX.append(data_CTA[i]['LCX'])
            RCA.append(data_CTA[i]['RCA']) 
            
            SPECT_idx.append(0)
            CTA_idx.append(1)          
        
        # NI
        for i in range(self.bs*2):       
            indx.append(cnt) 
            cnt += 1
            
            QRS, T = self.preprocessing(data_NI[i])
            qrs_input.append(QRS)
            t_input.append(T)
            is_ischemia.append(0)
            ischemia_intensity.append(-np.inf)
            ischemia_area.append(-np.inf)
            xin_jian.append(-np.inf)
            qian_bi.append(-np.inf)
            jiange_bi.append(-np.inf)
            xia_bi.append(-np.inf)
            ce_bi.append(-np.inf)
            id.append(-np.inf)
            
            LAD.append(-np.inf)
            LCX.append(-np.inf)
            RCA.append(-np.inf) 
            
            SPECT_idx.append(0)
            CTA_idx.append(0)               

        input =  {'QRS':qrs_input, 'T':t_input, 'is_ischemia':is_ischemia, 
                  'ischemia_intensity':ischemia_intensity, 'ischemia_area':ischemia_area,
                  'xin_jian':xin_jian, 'qian_bi':qian_bi,'jiange_bi':jiange_bi,
                  'xia_bi':xia_bi, 'ce_bi':ce_bi, 'LAD':LAD, 'LCX':LCX,
                  'RCA':RCA, 'id':id, 'indx':indx, 'SPECT_idx':SPECT_idx, 'CTA_idx':CTA_idx}   
            
        input_shuffle = self.shuffle_list(input, indx.copy())  
        input_shuffle['QRS'] = input_shuffle['QRS'].unsqueeze(1).permute(0,1,3,4,2)
        input_shuffle['T'] = input_shuffle['T'].unsqueeze(1).permute(0,1,3,4,2)
        
        return input_shuffle



def test():
    loader = load_train()
    loader.__getitem__(1)

test()