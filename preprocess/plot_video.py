# sort data
import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import torch
import cv2
from matplotlib import cm
viridis = cm.get_cmap('jet', 256)
colormap = plt.get_cmap('jet')

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



def read_txt(data_path):
    input = pd.read_csv(data_path, sep='\t', header=None,)
    waveforms = input.values[:, 1:]
    waveforms_2d = waveforms.reshape(waveforms.shape[0],6,6)
    return waveforms, waveforms_2d


class VideoWriter:
    def __init__(self, name, width, height, fps=25):
        # type: (str, int, int, int) -> None
        if not name.endswith('.mp4'):  # 保证文件名的后缀是.mp4
            name += '.mp4'
        self.__name = name          # 文件名
        self.__height = height      # 高
        self.__width = width        # 宽
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 如果是mp4视频，编码需要为mp4v
        self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

    def write(self, frame):
        if frame.dtype != np.uint8:  # 检查frame的类型
            raise ValueError('frame.dtype should be np.uint8')
        # 检查frame的大小
        row, col, _ = frame.shape
        if row != self.__height or col != self.__width:
            return
        self.__writer.write(frame)

    def close(self):
        self.__writer.release()
        
def plot_waveform(waveforms3d, time_stamp, outpath):
    q_loc = time_stamp[0] - 10
    r_loc = time_stamp[1]
    s_loc = time_stamp[2] + 10
    t_onset_loc = time_stamp[4] 
    t_offset_loc = time_stamp[5] 
    t_loc = time_stamp[3]     
    qrs = waveforms3d[q_loc:s_loc, :, :]
    t = waveforms3d[t_onset_loc:t_offset_loc, :, :]
    
    r_peak = waveforms3d[r_loc-1:r_loc+1, :, :]
    t_peak = waveforms3d[t_loc-1:t_loc+1, :, :]
    r_peak = resize3D(r_peak, size=(r_peak.shape[0], 128, 128))[1,:,:]
    t_peak = resize3D(t_peak, size=(t_peak.shape[0], 128, 128))[1,:,:]
    r_peak = (colormap(r_peak) * 255).astype(np.uint8)[:,:,:3]
    t_peak = (colormap(t_peak) * 255).astype(np.uint8)[:,:,:3]
    cv2.imwrite(outpath + '_qrs.png', r_peak)
    cv2.imwrite(outpath + '_t.png', t_peak)
    
        
    qrs = resize3D(qrs, size=(qrs.shape[0], 128, 128))
    t =  resize3D(t, size=(t.shape[0], 128, 128))
    
    vw = VideoWriter(outpath + '_qrs.mp4', 128, 128)
    for j in range(qrs.shape[0]):
        data = (colormap(qrs[j,:,:]) * 255).astype(np.uint8)[:,:,:3]
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        # 写入图像
        vw.write(data)
    # 关闭
    vw.close()
    
    vw = VideoWriter(outpath + '_t.mp4', 128, 128)
    for j in range(t.shape[0]):
        data = (colormap(t[j,:,:]) * 255).astype(np.uint8)[:,:,:3]
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        # 写入图像
        vw.write(data)
    # 关闭
    vw.close()
    
    return

    


# import numpy as np
# import cv2
# size = 720*16//9, 720
# duration = 2
# fps = 25
# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
# for _ in range(fps * duration):
#     data = np.random.randint(0, 256, size, dtype='uint8')
#     out.write(data)
# out.release()

    
#     return




labelpath = '/media/cygzz/data/rtao/projects/MCG-NC/data/data001/Labels.xlsx'
datapath = '/media/cygzz/data/rtao/projects/MCG-NC/data/data001/2023-7-21MCG DATA'
outpath = '/media/cygzz/data/rtao/projects/MCG-NC/data/data001/vis3D'
out_pickle = '/media/cygzz/data/rtao/projects/MCG-NC/data/data001/data_pickle'

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
    
   
    MCG_data[one_id] = one_data
    
    time_stamp = [int(loc_q_peak.values[i]),int(loc_r_peak.values[i]), 
                  int(loc_s_peak.values[i]),int(loc_t_peak.values[i]),
                  int(loc_t_onset.values[i]), int(loc_t_end.values[i])]
    
    out_path_one = os.path.join(outpath, str(is_ischemia.values[i]) +'_'+ one_id)
    plot_waveform(waveforms_2d, time_stamp, out_path_one)
    
    



