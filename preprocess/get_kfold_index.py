import sklearn.model_selection as ms
import numpy as np
import pickle


# control: 914
# spect is: 127
# cta_is: 448

kf = ms.KFold(n_splits=5, shuffle=True)

# control
tag = 'Non_Ischemia'
X = np.linspace(0,913,914)
fold = kf.split(X)
Non_Ischemia_fold = dict()
for i, (train_index, test_index) in enumerate(kf.split(X)):
    Non_Ischemia_fold[i] = {'train_index': list(train_index), 'test_index':test_index}
    print(f"Fold {i}------Normal:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    
# CTA
tag = 'Ischemia_CTA'
X = np.linspace(0,448-1,448)
fold = kf.split(X)
Ischemia_CTA_fold = dict()
for i, (train_index, test_index) in enumerate(kf.split(X)):
    Ischemia_CTA_fold[i] = {'train_index': list(train_index), 'test_index':test_index}    
    print(f"Fold {i}--------CTA:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
        
# SPECT
tag = 'Ischemia_SPECT'
X = np.linspace(0,127-1,127)
fold = kf.split(X)
Ischemia_SPECT_fold = dict()
for i, (train_index, test_index) in enumerate(kf.split(X)):
    Ischemia_SPECT_fold[i] = {'train_index': list(train_index), 'test_index':test_index}    
    print(f"Fold {i}---------SPECT:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")   
    
fold_all = {'Non_Ischemia_fold':Non_Ischemia_fold, 'Ischemia_CTA_fold':Ischemia_CTA_fold,
            'Ischemia_SPECT_fold':Ischemia_SPECT_fold} 

# with open('/media/cygzz/data/rtao/projects/MCG-NC/data/data001/data_pickle/5fold_CV_index.pickle', 'wb') as f:
#     pickle.dump(fold_all, f)
# f.close()
