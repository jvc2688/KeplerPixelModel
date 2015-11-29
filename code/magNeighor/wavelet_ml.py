import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
import pywt
import h5py
import kplr
import math
import mlpy.wavelet as wave
import sys
from scipy.signal import medfilt


client = kplr.API()

def remove(item_list, remove_list):
    dtype = [('kic', int), ('mag', float), ('rms_3', float), ('rms_6', float), ('rms_12', float)]
    item_list = list(item_list)
    for item in item_list:
        if item[0] in remove_list:
            item_list.remove(item)
    return np.array(item_list, dtype=dtype)

#help function to find the pixel mask
def get_pixel_mask(flux, kplr_mask):
    pixel_mask = np.zeros(flux.shape)
    pixel_mask[np.isfinite(flux)] = 1 # okay if finite
    pixel_mask[:, (kplr_mask < 1)] = 0 # unless masked by kplr
    return pixel_mask

#help function to find the epoch mask
def get_epoch_mask(pixel_mask):
    foo = np.sum(np.sum((pixel_mask > 0), axis=2), axis=1)
    epoch_mask = np.zeros_like(foo)
    epoch_mask[(foo > 0)] = 1
    return epoch_mask

def load_data(tpf):
    kplr_mask, time, flux, flux_err = [], [], [], []
    with tpf.open() as file:
        hdu_data = file[1].data
        kplr_mask = file[2].data
        meta = file[1].header
        column = meta['1CRV4P']
        row = meta['2CRV4P']
        time = hdu_data["time"]
        flux = hdu_data["flux"]
        flux_err = hdu_data["flux_err"]
    pixel_mask = get_pixel_mask(flux, kplr_mask)
    epoch_mask = get_epoch_mask(pixel_mask)
    flux = flux[:, kplr_mask>0]
    flux_err = flux_err[:, kplr_mask>0]
    shape = flux.shape

    flux = flux.reshape((flux.shape[0], -1))
    flux_err = flux_err.reshape((flux.shape[0], -1))

    #interpolate the bad points
    for i in range(flux.shape[1]):
        interMask = np.isfinite(flux[:,i])
        flux[~interMask,i] = np.interp(time[~interMask], time[interMask], flux[interMask,i])
        flux_err[~interMask,i] = np.inf
    
    return time, flux, pixel_mask, kplr_mask, epoch_mask, flux_err, column, row


def get_lightcurve_info(prefix, koi_num=None, pixel=False):
    
    f = h5py.File('%s.hdf5'%prefix, 'r')
    cpm_info = f['/cpm_info']
    data_group = f['/data']
    
    kid = f.attrs['kid'][()]
    quarter = f.attrs['quarter'][()]
    ccd = data_group['ccd'][()]
    target_flux = data_group['target_flux'][:]
    target_kplr_mask = data_group['target_kplr_mask'][:,:]
    epoch_mask = data_group['epoch_mask'][:]
    data_mask = data_group['data_mask'][:]
    time = data_group['time'][:]
    
    target_tpf = client.target_pixel_files(ktc_kepler_id=kid, sci_data_quarter=quarter, ktc_target_type="LC")[0]
    target_kepmag = target_tpf.kic_kepmag

    origin_time, origin_target_flux, origin_target_pixel_mask, origin_target_kplr_mask, origin_epoch_mask, origin_flux_err, column, row= load_data(target_tpf)
    
    origin_target_flux = origin_target_flux[epoch_mask>0]
    origin_time = origin_time[epoch_mask>0]

    target_kplr_mask = target_kplr_mask.flatten()
    target_kplr_mask = target_kplr_mask[target_kplr_mask>0]
    target_flux = target_flux[:, target_kplr_mask==3]
    origin_target_flux = origin_target_flux[:, target_kplr_mask==3]
    
    #construct the lightcurve
    if pixel:
        mean = np.mean(target_flux, axis=0)
        pixel = np.argmax(mean)
        #print pixel
        target_lightcurve = target_flux[:, pixel]
        origin_target_lightcurve = origin_target_flux[:, pixel]
        fit_lightcurve = np.load('./%s.npy'%prefix)
        ratio = np.divide(target_lightcurve, fit_lightcurve[:, 0])
    else:
        target_lightcurve = np.sum(target_flux, axis=1)
        origin_target_lightcurve = np.sum(origin_target_flux, axis=1)
        
        fit_pixel = data_group['fit_flux'][:]
        fit_lightcurve = np.sum(fit_pixel, axis=1)
        ratio = np.divide(target_lightcurve, fit_lightcurve)
    
    f.close()

    return time, target_lightcurve, fit_lightcurve, ratio, ccd, target_kepmag

def signal_extension(signal):
    total_length = 2**13
    length = signal.shape[0]
    if length%2 != 0:
        signal = np.concatenate((signal, np.array([signal[0]])), axis=0)
    length = signal.shape[0]
    delta = (total_length-length)/2
    prior = signal[-delta:]
    poster =  signal[0:delta]
    signal = np.concatenate((prior, signal, poster), axis=0)
    return signal

def variance(x, K):
    sigma = []
    for i in range(1, len(x)+1):
        x2_i = x[i-1]**2
        length = x2_i.shape[0]
        sigma_i = np.zeros(length)
        
        num = (2**i)*K+1
        loop_num = num//length
        sum = loop_num*np.sum(x2_i)
        remain = num%length
        sigma_tmp = sum
        for k in range((-2**(i-1))*K, -(2**(i-1))*K+remain):
            k_i = k%length
            sigma_tmp += x2_i[k_i]
        sigma_i[0] = 1./((2**i)*K+1)*sigma_tmp
        for j in range(1,length):
            begin = (j-(2**(i-1))*K-1)%length
            end = (j+(2**(i-1))*K)%length
            sigma_tmp = sigma_tmp - x2_i[begin] + x2_i[end]
            '''
            sigma_tmp = sum
            for k in range(j-(2**(i-1))*K, j-(2**(i-1))*K+remain):
                k_i = k%length
                sigma_tmp += x2_i[k_i]
            '''
            '''
            for k in range(j-(2**(i-1))*K, j+(2**(i-1))*K+1):
                k_i = k%length
                sigma_tmp += x2_i[k_i]
            '''
            sigma_i[j] = 1./((2**i)*K+1)*sigma_tmp
        sigma.append(sigma_i)
    return sigma

def get_cdpp(total_ratio):
    length = total_ratio.shape[0]
    level=12
    cdpp_list = []
    for trail_length in [6, 12, 24]:
        signal_origin = np.zeros(length)
        for i in range(1500, 1500+trail_length):
            signal_origin[i] = 1.

        ratio = signal_extension(total_ratio)
        signal = signal_extension(signal_origin)

        wavelet = pywt.Wavelet('db6')

        swc_ratio = pywt.swt(ratio, wavelet, level)
        swc_signal = pywt.swt(signal, wavelet, level)
        x = []
        s = []
        for i in range(0, level):
            x.append(swc_ratio[level-1-i][1])
            s.append(swc_signal[level-1-i][1])
        x.append(swc_ratio[0][0])
        s.append(swc_signal[0][0])

        K = 50*trail_length
        sigma = variance(x, K)
        
        M = len(s)
        D = np.zeros(8192)
        for i in range(1, M+1):
            power = np.min([i, M-1])
            D += (2**(-power))*np.convolve(sigma[i-1]**(-1), s[i-1]**2, 'same')
        cdpp = (1e6)*np.sqrt(D)**(-1)
        rms_cdpp = math.sqrt(np.mean(cdpp**2))
        cdpp_list.append(rms_cdpp)
    return cdpp_list

def get_cdpp_ml(total_ratio):
    length = total_ratio.shape[0]
    level=12
    cdpp_list = []
    for trail_length in [6, 12, 24]:
        signal_origin = np.zeros(length)
        for i in range(1500, 1500+trail_length):
            signal_origin[i] = 1.

        ratio = signal_extension(total_ratio)
        signal = signal_extension(signal_origin)

        wavelet = pywt.Wavelet('db6')

        swc_ratio = wave.uwt(ratio, 'd', 12, level)
        swc_signal = wave.uwt(signal, 'd', 12, level)
        x = []
        s = []
        for i in range(0, level):
            x.append(swc_ratio[i])
            s.append(swc_signal[i])
        x.append(swc_ratio[-1])
        s.append(swc_signal[-1])

        K = 50*trail_length
        sigma = variance(x, K)
        
        M = len(s)
        D = np.zeros(8192)
        for i in range(1, M+1):
            power = np.min([i, M-1])
            D += (2**(-power))*np.convolve(sigma[i-1]**(-1), s[i-1]**2, 'same')
        cdpp = (1e6)*np.sqrt(D)**(-1)
        rms_cdpp = math.sqrt(np.mean(cdpp**2))
        cdpp_list.append(rms_cdpp)
    return cdpp_list

def load_cdpp(tpf):
    with tpf.open() as file:
        meta = file[1].header
        cdpp_3 = meta['CDPP3_0']
        if type(cdpp_3) is not float:
            cdpp_3 = np.nan
        cdpp_6 = meta['CDPP6_0']
        if type(cdpp_6) is not float:
            cdpp_6 = np.nan
        cdpp_12 = meta['CDPP12_0']
        if type(cdpp_12) is not float:
            cdpp_12 = np.nan
    return cdpp_3, cdpp_6, cdpp_12

def get_fake_data(target_flux, length, strength):
    #the sine distortion
    '''
        factor = np.arange(target_flux.shape[0])
        factor = (1+0.004*np.sin(12*np.pi*factor/factor[-1]))
        for i in range(0, target_flux.shape[0]):
        target_flux[i] = target_flux[i] * factor[i]
        '''
    position = target_flux.shape[0]/2-100
    #the fake transit
    target_flux[position:position+length, :] = target_flux[position:position+length, :]*(1-strength)
    return target_flux, position


if __name__ == "__main__":

    if False:
        argv = sys.argv
        kid = int(argv[1])
        quarter = 5
        offset = 0
        total_num = 160
        filter_num = 108
        l2 = 1e5
        auto_l2 = 1e5
        ccd = True
        auto = True
        poly = 0
        auto_offset = 18
        auto_window = 3
        margin = 18

        koi_num = None

        all_time, all_target_lightcurve, all_fit_lightcurve, all_ratio, all_transit_mask, all_mean_list, all_std_list, all_sn = [],[],[],[],[],[],[],[]

        for part in range(1,4):
            prefix = 'kic%d/lightcurve_%d_q%d_num%d-%d_reg%.0e_poly%r_auto%r-%d-%d-%.0e_margin%d_part%d_default_pixel'%(kid, kid, quarter, offset+1, total_num, l2, poly, auto, auto_offset, auto_window, auto_l2, margin, part)
            tmp_time, tmp_target_lightcurve, tmp_fit_lightcurve, tmp_ratio, ccd, target_kepmag = get_lightcurve_info(prefix, koi_num, False)
            all_time.append(tmp_time)
            all_target_lightcurve.append(tmp_target_lightcurve)
            all_fit_lightcurve.append(tmp_fit_lightcurve)
            all_ratio.append((tmp_ratio-1.))
        total_time = np.concatenate(all_time, axis=0)
        total_target_lightcurve = np.concatenate(all_target_lightcurve, axis=0)
        total_fit_lightcurve = np.concatenate(all_fit_lightcurve, axis=0)
        total_ratio = np.concatenate(all_ratio, axis=0)

        print target_kepmag, get_cdpp(total_ratio)
        print target_kepmag, get_cdpp_ml(total_ratio)

        target_tpf = client.target_pixel_files(ktc_kepler_id=kid, sci_data_quarter=quarter, ktc_target_type="LC")[0]
        cdpp_3, cdpp_6, cdpp_12 =  load_cdpp(target_tpf)
        print target_kepmag, cdpp_3, cdpp_6, cdpp_12

    if False:
        argv = sys.argv
        stat = int(argv[1])
        
        quarter = 8

        kic_list = []
        g_stars_list = []
        stars_list = []
        
        f  = h5py.File('star_list_new.hdf5', 'r')
        g_stars_list.extend(f['select_g_stars'])
        stars_list.extend(f['select_stars'])
        kic_list.extend(f['select_g_stars'])
        kic_list.extend(f['select_stars'])
        f.close()
        
        g_stars_list = set(g_stars_list)
        stars_list = set(stars_list)
        kic_list = set(kic_list)
        data_list = []
        for kic in kic_list:
            data_list.append((int(kic), None))

        k = 1
        cdpp_list = []
        cdpp_g_list = []
        cdpp_rand_list = []
        dtype = [('kic', int), ('mag', float), ('rms_3', float), ('rms_6', float), ('rms_12', float)]
        
        load_list = []
        if stat == 1:
            cdpp_load_1 = list(np.load('cdpp/ml12_cdpp_list_q%d_m.npy'%quarter))
            cdpp_load_2 = list(np.load('cdpp/ml12_cdpp_g_list_q%d_m.npy'%quarter))
            cdpp_load_3 = list(np.load('cdpp/ml12_cdpp_rand_list_q%d_m.npy'%quarter))

            for i in range(0, len(cdpp_load_1)):
                cdpp_tuple = tuple(cdpp_load_1[i])
                cdpp_list.append(cdpp_tuple)
                load_list.append(int(cdpp_tuple[0]))
            for i in range(0, len(cdpp_load_2)):
                cdpp_tuple = tuple(cdpp_load_2[i])
                cdpp_g_list.append(cdpp_tuple)
            for i in range(0, len(cdpp_load_3)):
                cdpp_tuple = tuple(cdpp_load_3[i])
                cdpp_rand_list.append(cdpp_tuple)
        load_list = set(load_list)
        
        for kid, koi_num in data_list:
            print (kid, k)
            k+=1
            if kid in load_list:
                continue
            #quarter = 5
            offset = 0
            total_num = 160
            filter_num = 108
            l2 = 1e5
            auto_l2 = 1e5
            ccd = True
            auto = True
            poly = 0
            auto_offset = 18
            auto_window = 3
            margin = 18

            all_time, all_target_lightcurve, all_fit_lightcurve, all_ratio = [],[],[],[]

            for part in range(1,3):
                prefix = 'kic%d/lightcurve_%d_q%d_num%d-%d_reg%.0e_poly%r_auto%r-%d-%d-%.0e_margin%d_part%d_default_pixel'%(kid, kid, quarter, offset+1, total_num, l2, poly, auto, auto_offset, auto_window, auto_l2, margin, part)
                tmp_time, tmp_target_lightcurve, tmp_fit_lightcurve, tmp_ratio, ccd, target_kepmag = get_lightcurve_info(prefix, koi_num, False)
                all_time.append(tmp_time)
                all_target_lightcurve.append(tmp_target_lightcurve)
                all_fit_lightcurve.append(tmp_fit_lightcurve)
                all_ratio.append((tmp_ratio-1.))
            total_time = np.concatenate(all_time, axis=0)
            total_target_lightcurve = np.concatenate(all_target_lightcurve, axis=0)
            total_fit_lightcurve = np.concatenate(all_fit_lightcurve, axis=0)
            total_ratio = np.concatenate(all_ratio, axis=0)

            rms_cdpp_3, rms_cdpp_6, rms_cdpp_12 = get_cdpp_ml(total_ratio)

            print target_kepmag, rms_cdpp_3, rms_cdpp_6, rms_cdpp_12
            cdpp_list.append((kid, target_kepmag, rms_cdpp_3, rms_cdpp_6, rms_cdpp_12))
            if kid in g_stars_list:
                cdpp_g_list.append((kid, target_kepmag, rms_cdpp_3, rms_cdpp_6, rms_cdpp_12))
            elif kid in stars_list:
                cdpp_rand_list.append((kid, target_kepmag, rms_cdpp_3, rms_cdpp_6, rms_cdpp_12))
            else:
                print 'error'
            np.save('cdpp/ml12_cdpp_list_q%d.npy'%quarter, cdpp_list)
            np.save('cdpp/ml12_cdpp_g_list_q%d.npy'%quarter, cdpp_g_list)
            np.save('cdpp/ml12_cdpp_rand_list_q%d.npy'%quarter, cdpp_rand_list)


        cdpp_list = np.array(cdpp_list, dtype=dtype)
        cdpp_g_list = np.array(cdpp_g_list, dtype=dtype)
        cdpp_rand_list = np.array(cdpp_rand_list, dtype=dtype)

        np.save('cdpp/ml12_cdpp_list_q%d.npy'%quarter, cdpp_list)
        np.save('cdpp/ml12_cdpp_g_list_q%d.npy'%quarter, cdpp_g_list)
        np.save('cdpp/ml12_cdpp_rand_list_q%d.npy'%quarter, cdpp_rand_list)

        for i in range(cdpp_list.shape[0]):
            cdpp_item = cdpp_list[i]
            if cdpp_item['rms_3']>500:
                print cdpp_item

        plt.plot(cdpp_list['mag'], cdpp_list['rms_3'], '.k')
        plt.show()
        plt.clf()


    if False:
        argv = sys.argv
        stat = int(argv[1])
        
        kic_list = []
        g_stars_list = []
        stars_list = []
        
        quarter = 8

        f  = h5py.File('star_list_new.hdf5', 'r')
        g_stars_list.extend(f['select_g_stars'])
        stars_list.extend(f['select_stars'])
        kic_list.extend(f['select_g_stars'])
        kic_list.extend(f['select_stars'])
        f.close()
        
        g_stars_list = set(g_stars_list)
        stars_list = set(stars_list)
        kic_list = set(kic_list)
        data_list = []
        for kic in kic_list:
            data_list.append((int(kic), None))

        k = 1
        cdpp_list = []
        cdpp_g_list = []
        cdpp_rand_list = []
        dtype = [('kic', int), ('mag', float), ('rms_3', float), ('rms_6', float), ('rms_12', float)]
        
        load_list = []
        if stat == 1:
            cdpp_load_1 = list(np.load('cdpp/pdc12_cdpp_list_q%d.npy'%quarter))
            cdpp_load_2 = list(np.load('cdpp/pdc12_cdpp_g_list_q%d.npy'%quarter))
            cdpp_load_3 = list(np.load('cdpp/pdc12_cdpp_rand_list_q%d.npy'%quarter))

            for i in range(0, len(cdpp_load_1)):
                cdpp_tuple = tuple(cdpp_load_1[i])
                cdpp_list.append(cdpp_tuple)
                load_list.append(int(cdpp_tuple[0]))
            for i in range(0, len(cdpp_load_2)):
                cdpp_tuple = tuple(cdpp_load_2[i])
                cdpp_g_list.append(cdpp_tuple)
            for i in range(0, len(cdpp_load_3)):
                cdpp_tuple = tuple(cdpp_load_3[i])
                cdpp_rand_list.append(cdpp_tuple)
        load_list = set(load_list)
        
        for kid, koi_num in data_list:
            print (kid, k)
            k+=1
            if kid in load_list:
                continue
            offset = 0
            total_num = 160
            filter_num = 108
            l2 = 1e5
            auto_l2 = 1e5
            ccd = True
            auto = True
            poly = 0
            auto_offset = 18
            auto_window = 3
            margin = 18

            all_time, all_target_lightcurve, all_fit_lightcurve, all_ratio = [],[],[],[]

            lc = client.light_curves(ktc_kepler_id=kid, sci_data_quarter=quarter, ktc_target_type="LC")[0]
            data = lc.read()
            flux = data["PDCSAP_FLUX"]
            inds = np.isfinite(flux)
            flux = flux[inds]
            pdc_time = data["TIME"]
            pdc_time = pdc_time[inds]
            flux = flux/np.median(flux)-1.
            target_kepmag = lc.kic_kepmag

            rms_cdpp_3, rms_cdpp_6, rms_cdpp_12 = get_cdpp_ml(flux)

            print target_kepmag, rms_cdpp_3, rms_cdpp_6, rms_cdpp_12
            cdpp_list.append((kid, target_kepmag, rms_cdpp_3, rms_cdpp_6, rms_cdpp_12))
            if kid in g_stars_list:
                cdpp_g_list.append((kid, target_kepmag, rms_cdpp_3, rms_cdpp_6, rms_cdpp_12))
            elif kid in stars_list:
                cdpp_rand_list.append((kid, target_kepmag, rms_cdpp_3, rms_cdpp_6, rms_cdpp_12))
            else:
                print 'error'
            np.save('cdpp/pdc12_cdpp_list.npy_q%d'%quarter, cdpp_list)
            np.save('cdpp/pdc12_cdpp_g_list.npy_q%d'%quarter, cdpp_g_list)
            np.save('cdpp/pdc12_cdpp_rand_list_q%d.npy'%quarter, cdpp_rand_list)


        cdpp_list = np.array(cdpp_list, dtype=dtype)
        cdpp_g_list = np.array(cdpp_g_list, dtype=dtype)
        cdpp_rand_list = np.array(cdpp_rand_list, dtype=dtype)

        np.save('cdpp/pdc12_cdpp_list_q%d.npy'%quarter, cdpp_list)
        np.save('cdpp/pdc12_cdpp_g_list_q%d.npy'%quarter, cdpp_g_list)
        np.save('cdpp/pdc12_cdpp_rand_list_q%d.npy'%quarter, cdpp_rand_list)
        
        for i in range(cdpp_list.shape[0]):
            cdpp_item = cdpp_list[i]
            if cdpp_item['rms_3']>500:
                print cdpp_item

        plt.plot(cdpp_list['mag'], cdpp_list['rms_3'], '.k')
        plt.show()
        plt.clf()

#PDC median filter
    if False:
        argv = sys.argv
        stat = int(argv[1])

        win_size = 193
        
        kic_list = []
        g_stars_list = []
        stars_list = []
        
        quarter = 8

        f  = h5py.File('star_list_new.hdf5', 'r')
        g_stars_list.extend(f['select_g_stars'])
        stars_list.extend(f['select_stars'])
        kic_list.extend(f['select_g_stars'])
        kic_list.extend(f['select_stars'])
        f.close()
        
        g_stars_list = set(g_stars_list)
        stars_list = set(stars_list)
        kic_list = set(kic_list)
        data_list = []
        for kic in kic_list:
            data_list.append((int(kic), None))

        k = 1
        cdpp_list = []
        cdpp_g_list = []
        cdpp_rand_list = []
        dtype = [('kic', int), ('mag', float), ('rms_3', float), ('rms_6', float), ('rms_12', float)]
        
        load_list = []
        if stat == 1:
            cdpp_load_1 = list(np.load('cdpp/pdc12_cdpp_list_q%d_m%d.npy'%(quarter, win_size)))
            cdpp_load_2 = list(np.load('cdpp/pdc12_cdpp_g_list_q%d_m%d.npy'%(quarter, win_size)))
            cdpp_load_3 = list(np.load('cdpp/pdc12_cdpp_rand_list_q%d_m%d.npy'%(quarter, win_size)))

            for i in range(0, len(cdpp_load_1)):
                cdpp_tuple = tuple(cdpp_load_1[i])
                cdpp_list.append(cdpp_tuple)
                load_list.append(int(cdpp_tuple[0]))
            for i in range(0, len(cdpp_load_2)):
                cdpp_tuple = tuple(cdpp_load_2[i])
                cdpp_g_list.append(cdpp_tuple)
            for i in range(0, len(cdpp_load_3)):
                cdpp_tuple = tuple(cdpp_load_3[i])
                cdpp_rand_list.append(cdpp_tuple)
        load_list = set(load_list)
        
        for kid, koi_num in data_list:
            print (kid, k)
            k+=1
            if kid in load_list:
                continue
            offset = 0
            total_num = 160
            filter_num = 108
            l2 = 1e5
            auto_l2 = 1e5
            ccd = True
            auto = True
            poly = 0
            auto_offset = 18
            auto_window = 3
            margin = 18

            all_time, all_target_lightcurve, all_fit_lightcurve, all_ratio = [],[],[],[]

            lc = client.light_curves(ktc_kepler_id=kid, sci_data_quarter=quarter, ktc_target_type="LC")[0]
            data = lc.read()
            flux = data["PDCSAP_FLUX"]
            inds = np.isfinite(flux)
            flux = flux[inds]
            pdc_time = data["TIME"]
            pdc_time = pdc_time[inds]
            flux_m = medfilt(flux, kernel_size=win_size)
            flux_new = flux/flux_m
            flux_new = flux_new/np.median(flux_new)-1.
            flux = flux/np.median(flux)-1.
            target_kepmag = lc.kic_kepmag

            #rms_cdpp_3, rms_cdpp_6, rms_cdpp_12 = get_cdpp_ml(flux)

            #print target_kepmag, rms_cdpp_3, rms_cdpp_6, rms_cdpp_12

            rms_cdpp_3, rms_cdpp_6, rms_cdpp_12 = get_cdpp_ml(flux_new)

            print target_kepmag, rms_cdpp_3, rms_cdpp_6, rms_cdpp_12
            cdpp_list.append((kid, target_kepmag, rms_cdpp_3, rms_cdpp_6, rms_cdpp_12))
            if kid in g_stars_list:
                cdpp_g_list.append((kid, target_kepmag, rms_cdpp_3, rms_cdpp_6, rms_cdpp_12))
            elif kid in stars_list:
                cdpp_rand_list.append((kid, target_kepmag, rms_cdpp_3, rms_cdpp_6, rms_cdpp_12))
            else:
                print 'error'
            np.save('cdpp/pdc12_cdpp_list_q%d_m%d.npy'%(quarter, win_size), cdpp_list)
            np.save('cdpp/pdc12_cdpp_g_list_q%d_m%d.npy'%(quarter, win_size), cdpp_g_list)
            np.save('cdpp/pdc12_cdpp_rand_list_q%d_m%d.npy'%(quarter, win_size), cdpp_rand_list)


        cdpp_list = np.array(cdpp_list, dtype=dtype)
        cdpp_g_list = np.array(cdpp_g_list, dtype=dtype)
        cdpp_rand_list = np.array(cdpp_rand_list, dtype=dtype)

        np.save('cdpp/pdc12_cdpp_list_q%d_m%d.npy'%(quarter, win_size), cdpp_list)
        np.save('cdpp/pdc12_cdpp_g_list_q%d_m%d.npy'%(quarter, win_size), cdpp_g_list)
        np.save('cdpp/pdc12_cdpp_rand_list_q%d_m%d.npy'%(quarter, win_size), cdpp_rand_list)
        
        for i in range(cdpp_list.shape[0]):
            cdpp_item = cdpp_list[i]
            if cdpp_item['rms_3']>500:
                print cdpp_item

        plt.plot(cdpp_list['mag'], cdpp_list['rms_3'], '.k')
        plt.show()
        plt.clf()


    if False:
        quarter = 5
        cdpp_list = np.load('cdpp/ml12_cdpp_list_q%d.npy'%quarter)

        remove_list = np.load('cdpp/remove_list_r.npy')
        print remove_list.shape
        remove_list = set(remove_list)

        cdpp_list = remove(cdpp_list, remove_list)
        print cdpp_list.shape

        cdpp_list = cdpp_list[np.logical_and(cdpp_list['mag']>13,cdpp_list['mag']<13.5)]
        print cdpp_list.shape

        choice = np.random.choice(cdpp_list, 100, replace=False)
        print choice.shape
        choice_kid = set(choice['kic'])
        print len(choice_kid)

        for i in range(0, 10):
            stars = choice['kic'][i*10:(i+1)*10]
            f1 = open('star_list_inject/kic%d-%d'%(i*10+1, (i+1)*10),'w')
            for kid in stars:
                f1.write('%d\n'%kid)
            f1.close()







