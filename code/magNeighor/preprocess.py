import kplr
import random
import numpy as np
import matplotlib.pyplot as plt
import magNeighorFit as mnf
import math

def filter(target_tpf, neighor_tpfs, distance, rms=False, num=0):
    dtype = [('kic', int), ('rms', float), ('tpf', type(target_tpf)), ('lightcurve', np.ndarray)]
    
    time, target_flux, target_pixel_mask, target_kplr_mask, epoch_mask, flux_err, target_column, target_row= mnf.load_data(target_tpf)

    neighor_list = []
    tpfs = {}

    for key, tpf in neighor_tpfs.items():
        time, flux, pixel_mask, kplr_mask, epoch_mask, flux_err, column, row = mnf.load_data(tpf)
        dis_square = (target_row-row)**2+(target_column-column)**2
        if dis_square < distance**2:
            continue
        elif not rms:
            tpfs[key] = tpf
            continue
        
        time = time[epoch_mask>0]
        flux = flux[epoch_mask>0]
        lightcurve = np.sum(flux, axis=1)
        mean = np.mean(lightcurve)
        std = np.std(lightcurve)

        #construct the running mean and std
        mean_list = np.zeros_like(epoch_mask, dtype=float)
        std_list = np.zeros_like(epoch_mask, dtype=float)
        half_group_length = 12
        for i in range(0, epoch_mask.shape[0]):
            group_mask = np.zeros_like(epoch_mask)
            if i <= half_group_length:
                group_mask[0:i+half_group_length+1] = 1
            elif i >= epoch_mask.shape[0]-half_group_length-1:
                group_mask[i-half_group_length:] = 1
            else:
                group_mask[i-half_group_length:i+half_group_length+1] = 1
            co_mask = group_mask[epoch_mask>0]
            mean_list[i] = np.mean(lightcurve[co_mask>0])
            std_list[i] = np.std(lightcurve[co_mask>0])
        mean_list = mean_list[epoch_mask>0]
        std_list = std_list[epoch_mask>0]

        mean_std = np.mean(std_list)

        neighor_list.append((key, mean_std, tpf, lightcurve))

    if rms:
        neighor_list = np.array(neighor_list, dtype=dtype)
        neighor_list = np.sort(neighor_list, order='rms')

        for i in range(0, num):
            tmp_kic, tmp_rms, tmp_tpf, tmp_lightcurve = neighor_list[i]
            tpfs[tmp_kic] = tmp_tpf
            '''
            plt.plot(time, tmp_lightcurve, '.b', markersize=1)
            plt.xlabel('time[BKJD]')
            plt.ylabel('lightcurve')
            plt.title('kic:%d quarter:%d rms:%f'%(tmp_kic, quarter, tmp_rms))
            plt.savefig('./kic%d/preprocess/num%d.png'%(kid, i))
            plt.clf()
            '''

    return tpfs


