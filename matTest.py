import scipy.io as sio
import numpy as np
import ABIDE.ABIDEParser_1_HOFC_selected as Reader
vae_mat = r'F:\demo\GCN\GCN_AD\data\ADNI_fMRI_data\Result_AAL116signals\ROISignals_002_S_0295_S110474.mat'
vae_dict = sio.loadmat(vae_mat)
vae_data = vae_dict['ROISignals']
print(vae_data.shape)#(137, 116)
print(type(vae_data))#<class 'numpy.ndarray'>




