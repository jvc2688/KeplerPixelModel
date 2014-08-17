import kplr
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import leastSquareSolver as lss
from matplotlib.patches import Rectangle
import time as tm
import threading
import os
import math
import preprocess

client = kplr.API()

def find_mag_neighor(kic, quarter, num, offset=0, ccd=True):
    """
    ## inputs:
    - `kic` - target KIC number
    - `quarter` - target quarter
    - `num` - number of tpfs needed
    - `offset` - number of tpfs that are excluded
    - `ccd` - if the tpfs need to be on the same CCD
    
    ## outputs:
    - `target_tpf` - tpf of the target star
    - `tpfs` - tpfs of stars that are closet to the target star in magnitude
    """
    target_tpf = client.target_pixel_files(ktc_kepler_id=kic, sci_data_quarter=quarter, ktc_target_type="LC")[0]
    
    if ccd:
        stars_over = client.target_pixel_files(ktc_kepler_id="!=%d"%target_tpf.ktc_kepler_id, kic_kepmag=">=%f"%target_tpf.kic_kepmag, sci_data_quarter=target_tpf.sci_data_quarter, sci_channel=target_tpf.sci_channel, sort=("kic_kepmag", 1),ktc_target_type="LC", max_records=num+offset)
        stars_under = client.target_pixel_files(ktc_kepler_id="!=%d"%target_tpf.ktc_kepler_id, kic_kepmag="<=%f"%target_tpf.kic_kepmag, sci_data_quarter=target_tpf.sci_data_quarter, sci_channel=target_tpf.sci_channel, sort=("kic_kepmag", -1),ktc_target_type="LC", max_records=num+offset)
    else:
        stars_over = client.target_pixel_files(ktc_kepler_id="!=%d"%target_tpf.ktc_kepler_id, kic_kepmag=">=%f"%target_tpf.kic_kepmag, sci_data_quarter=target_tpf.sci_data_quarter, sci_channel="!=%d"%target_tpf.sci_channel, sort=("kic_kepmag", 1),ktc_target_type="LC", max_records=num+offset)
        stars_under = client.target_pixel_files(ktc_kepler_id="!=%d"%target_tpf.ktc_kepler_id, kic_kepmag="<=%f"%target_tpf.kic_kepmag, sci_data_quarter=target_tpf.sci_data_quarter, sci_channel="!=%d"%target_tpf.sci_channel, sort=("kic_kepmag", -1),ktc_target_type="LC", max_records=num+offset)

    print(len(stars_over), len(stars_under))
    tpfs = {}
    target_kepmag = target_tpf.kic_kepmag

    dtype = [('kic', int), ('bias', float), ('tpf', type(target_tpf))]
    neighor_list = []
    tpf_list = stars_over+stars_under
    for tpf in tpf_list:
        neighor_list.append((tpf.ktc_kepler_id, math.fabs(tpf.kic_kepmag-target_kepmag), tpf))

    neighor_list = np.array(neighor_list, dtype=dtype)
    neighor_list = np.sort(neighor_list, order='bias')

    for i in range(offset, offset+num):
        tmp_kic, tmp_bias, tmp_tpf = neighor_list[i]
        tpfs[tmp_kic] = tmp_tpf
    
    return target_tpf, tpfs

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

#help function to load data from tpf
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

def get_kfold_train_mask(length, k, rand=False):
    train_mask = np.ones(length, dtype=int)
    if random:
        for i in range(0, length):
            group = random.randint(0, k-1)
            train_mask[i] = group
    else:
        step = length//k
        for i in range(0, k-1):
            train_mask[i*step:(i+1)*step] = i
        train_mask[(k-1)*step:] = k-1
    return train_mask

def get_fit_matrix(target_tpf, neighor_tpfs, l2,  poly=0, auto=False, offset=0, window=0, auto_l2=0, part=None):
    """
    ## inputs:
    - `target_tpf` - target tpf
    - `neighor_tpfs` - neighor tpfs in magnitude
    - `auto` - if autorgression
    - `poly` - number of orders of polynomials of time need to be added
    
    ## outputs:
    - `neighor_flux_matrix` - fitting matrix of neighor flux
    - `target_flux` - target flux
    - `covar_list` - covariance matrix for every pixel
    - `time` - one dimension array of BKJD time
    - `neighor_kid` - KIC number of the neighor stars in the fitting matrix
    - `neighor_kplr_maskes` - kepler maskes of the neighor stars in the fitting matrix
    - `target_kplr_mask` - kepler mask of the target star
    - `epoch_mask` - epoch mask
    - `l2_vector` - array of L2 regularization strength
    """

    time, target_flux, target_pixel_mask, target_kplr_mask, epoch_mask, flux_err, column, row= load_data(target_tpf)

    neighor_kid, neighor_fluxes, neighor_pixel_maskes, neighor_kplr_maskes = [], [], [], []

    for key, tpf in neighor_tpfs.items():
        neighor_kid.append(key)
        tmpResult = load_data(tpf)
        neighor_fluxes.append(tmpResult[1])
        neighor_pixel_maskes.append(tmpResult[2])
        neighor_kplr_maskes.append(tmpResult[3])
        epoch_mask *= tmpResult[4]
    
    #construt the neighor flux matrix
    neighor_flux_matrix = np.float64(np.concatenate(neighor_fluxes, axis=1))
    target_flux = np.float64(target_flux)

    epoch_len = epoch_mask.shape[0]
    data_mask = np.zeros(epoch_len, dtype=int)

    print neighor_flux_matrix.shape

    #spilt lightcurve
    if part is not None:
        epoch_len = epoch_mask.shape[0]
        split_point = []
        split_point.append(0)
        i = 0
        while i <epoch_len:
            if epoch_mask[i] == 0:
                i += 1
            else:
                j = i+1
                while j<epoch_len and epoch_mask[j] == 0:
                    j += 1
                if j-i > 30:
                    split_point.append(j)
                i = j
        split_point.append(epoch_len)
        print split_point
        data_mask[split_point[part-1]:split_point[part]] = 1
        '''
        fit_epoch_mask =  np.split(fit_epoch_mask, split_point)[part-1]
        fit_time = np.split(time, split_point)[part-1]
        fit_target_flux = np.split(target_flux, split_point, axis=0)[part-1]
        flux_err = np.split(flux_err, split_point, axis=0)[part-1]
        neighor_flux_matrix = np.split(neighor_flux_matrix, split_point, axis=0)[part-1]

        print (fit_epoch_mask.shape, fit_time.shape, fit_target_flux.shape, flux_err.shape, neighor_flux_matrix.shape)
        '''
    else:
        data_mask = 1

    #add auto-regression terms
    if auto:
        tmp_target_kplr_mask = target_kplr_mask.flatten()
        tmp_target_kplr_mask = tmp_target_kplr_mask[tmp_target_kplr_mask>0]
        auto_flux = target_flux[:, tmp_target_kplr_mask==3]
        for i in range(offset+1, offset+window+1):
            neighor_flux_matrix = np.concatenate((neighor_flux_matrix, np.roll(auto_flux, i, axis=0)), axis=1)
            neighor_flux_matrix = np.concatenate((neighor_flux_matrix, np.roll(auto_flux, -i, axis=0)), axis=1)
        data_mask[0:offset+window] = 0
        data_mask[-offset+window:] = 0

    #remove bad time point based on simulteanous epoch mask
    co_mask = data_mask*epoch_mask
    time = time[co_mask>0]
    target_flux = target_flux[co_mask>0]
    flux_err = flux_err[co_mask>0]
    neighor_flux_matrix = neighor_flux_matrix[co_mask>0, :]

    print neighor_flux_matrix.shape
    '''
    #add autoregression terms
    if auto:
        epoch_len = epoch_mask.shape[0]
        tmp_target_kplr_mask = target_kplr_mask.flatten()
        tmp_target_kplr_mask = tmp_target_kplr_mask[tmp_target_kplr_mask>0]
        target_len = np.sum(tmp_target_kplr_mask==3)
        print target_len
        auto_flux = np.zeros((epoch_len, target_len))
        #auto_flux[epoch_mask>0] = inter_target_flux[:, tmp_target_kplr_mask==3]
        auto_flux = inter_target_flux[:, tmp_target_kplr_mask==3]
        auto_pixel = np.zeros((epoch_len, 2*window*target_len))
        auto_len = window*target_len
        for i in range(offset+window, epoch_len-window-offset):
            auto_pixel[i, 0:auto_len] = auto_flux[i-offset-window:i-offset, :].flatten()
            auto_pixel[i, auto_len:2*auto_len] = auto_flux[i+offset+1:i+offset+window+1].flatten()
        for i in range(0, offset+window):
            auto_pixel[i, auto_len:2*auto_len] = auto_flux[i+offset+1:i+offset+window+1].flatten()
        for i in range(epoch_len-window-offset, epoch_len):
            auto_pixel[i, 0:auto_len] = auto_flux[i-offset-window:i-offset].flatten()
        auto_pixel = auto_pixel[epoch_mask>0, :]
        neighor_flux_matrix = np.concatenate((neighor_flux_matrix, auto_pixel), axis=1)
    '''

    #construct l2 vectors
    pixel_num = neighor_flux_matrix.shape[1]
    l2_vector = np.ones(pixel_num, dtype=float)*l2

    #add polynomial terms
    if poly is not None:
        time_mean = np.mean(time)
        time_std = np.std(time)
        nor_time = (time-time_mean)/time_std
        p = np.polynomial.polynomial.polyvander(nor_time, poly)
        neighor_flux_matrix = np.concatenate((neighor_flux_matrix, p), axis=1)
        l2_vector = np.concatenate((l2_vector, np.zeros(poly+1)), axis=0)

    print neighor_flux_matrix.shape

    return neighor_flux_matrix, target_flux, flux_err, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, data_mask, l2_vector, pixel_num

def get_transit_mask(koi, time, epoch_mask, measure_half_len):
    #get the transit information of the corressponding koi
    koi = client.koi(koi_num)
    transit_time = koi.koi_time0bk
    period = koi.koi_period
    transit_duration = koi.koi_duration
    
    #find the transit locations of the koi in lightcurve
    time_len = time.shape[0]
    transit_list = []
    while transit_time < time[-1]:
        if transit_time > time[0]:
            transit_list.append(transit_time)
        transit_time += period
    print transit_list
    whole_time = np.zeros_like(epoch_mask, dtype=float)
    whole_time[epoch_mask>0] = np.float64(time)
    
    half_len = round(transit_duration/2/0.5)

    transit_mask = np.zeros_like(epoch_mask)
    for i in range(0,len(transit_list)):
        loc = (np.abs(whole_time-transit_list[i])).argmin()
        transit_mask[loc-measure_half_len:loc+measure_half_len+1] = 1
        transit_mask[loc-half_len:loc+half_len+1] = 2
        transit_mask[loc] = 3
    return transit_mask

def fit_target(target_flux, target_kplr_mask, neighor_flux_matrix, time, epoch_mask, covar_list, margin, l2_vector=None, thread_num=1, prefix="lightcurve", transit_mask=None):
    """
    ## inputs:
    - `target_flux` - target flux
    - `target_kplr_mask` - kepler mask of the target star
    - `neighor_flux_matrix` - fitting matrix of neighor flux
    - `time` - array of time 
    - `epoch_mask` - epoch mask
    - `covar_list` - covariance list
    - `margin` - size of the test region
    - `poly` - number of orders of polynomials of time(zero order is the constant level)
    - `l2_vector` - array of L2 regularization strength
    - `thread_num` - thread number
    - `prefix` - output file's prefix
    
    ## outputs:
    - prefix.npy file - fitting fluxes of pixels
    """
    target_kplr_mask = target_kplr_mask.flatten()
    target_kplr_mask = target_kplr_mask[target_kplr_mask>0]

    optimal_len = np.sum(target_kplr_mask==3)
    print optimal_len
    target_flux = target_flux[:, target_kplr_mask==3]

    covar_list = covar_list[:, target_kplr_mask==3]
    covar = np.mean(covar_list, axis=1)**2
    fit_flux = []
    fit_coe = []
    length = target_flux.shape[0]
    total_length = epoch_mask.shape[0]
    
    thread_len = total_length//thread_num
    last_len = total_length - (thread_num-1)*thread_len
    
    filename = "./%s"%prefix
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    class fit_epoch(threading.Thread):
        def __init__(self, thread_id, initial, len, time_initial, time_len):
            threading.Thread.__init__(self)
            self.thread_id = thread_id
            self.initial = initial
            self.len = len
            self.time_initial = time_initial
            self.time_len = time_len
        def run(self):
            print('Starting%d'%self.thread_id)
            print (self.thread_id , self.time_initial, self.time_len)
            tmp_fit_flux = np.empty((self.time_len, optimal_len))
            time_stp = 0
            for i in range(self.initial, self.initial+self.len):
                if epoch_mask[i] == 0:
                    continue
                #fit the transit part
                '''
                if transit_mask[i] == 0:
                    time_stp += 1
                    continue
                '''
                train_mask = np.ones(total_length)
                if i<margin:
                    train_mask[0:i+margin+1] = 0
                elif i > total_length-margin-1:
                    train_mask[i-margin:] = 0
                else:
                    train_mask[i-margin:i+margin+1] = 0
                train_mask = train_mask[epoch_mask>0]
                
                tmp_covar = covar[train_mask>0]
                result = lss.linear_least_squares(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], tmp_covar, l2_vector)
                tmp_fit_flux[time_stp, :] = np.dot(neighor_flux_matrix[time_stp+self.time_initial, :], result)
                np.save('./%stmp%d.npy'%(prefix, self.thread_id), tmp_fit_flux)
                time_stp += 1
                print('done%d'%i)
            print('Exiting%d'%self.thread_id)
    
    thread_list = []
    time_initial = 0
    for i in range(0, thread_num-1):
        initial = i*thread_len
        thread_epoch = epoch_mask[initial:initial+thread_len]
        time_len = np.sum(thread_epoch)
        thread = fit_epoch(i, initial, thread_len, time_initial, time_len)
        thread.start()
        thread_list.append(thread)
        time_initial += time_len
    
    initial = (thread_num-1)*thread_len
    thread_epoch = epoch_mask[initial:initial+last_len]
    time_len = np.sum(thread_epoch)
    thread = fit_epoch(thread_num-1, initial, last_len, time_initial, time_len)
    thread.start()
    thread_list.append(thread)
    
    for t in thread_list:
        t.join()
    print 'all done'
    
    offset = 0
    window = 0
    
    for i in range(0, thread_num):
        tmp_fit_flux = np.load('./%stmp%d.npy'%(prefix, i))
        if i==0:
            fit_flux = tmp_fit_flux
        else:
            fit_flux = np.concatenate((fit_flux, tmp_fit_flux), axis=0)
    np.save('./%s.npy'%prefix, fit_flux)
    
    for i in range(0, thread_num):
        os.remove('./%stmp%d.npy'%(prefix, i))

def plot_fit(kid, quarter, l2, offset, num, pixel_num, poly, ccd, target_flux, target_kplr_mask, epoch_mask, data_mask, time, margin, prefix, koi_num):
    
    target_tpf = client.target_pixel_files(ktc_kepler_id=kid, sci_data_quarter=quarter, ktc_target_type="LC")[0]
    origin_time, origin_target_flux, origin_target_pixel_mask, origin_target_kplr_mask, origin_epoch_mask, origin_flux_err, column, row= load_data(target_tpf)
    
    origin_target_flux = origin_target_flux[epoch_mask>0]
    origin_time = origin_time[epoch_mask>0]

    target_kplr_mask = target_kplr_mask.flatten()
    target_kplr_mask = target_kplr_mask[target_kplr_mask>0]
    target_flux = target_flux[:, target_kplr_mask==3]
    origin_target_flux = origin_target_flux[:, target_kplr_mask==3]
    
    #construct the lightcurve 
    target_lightcurve = np.sum(target_flux, axis=1)
    origin_target_lightcurve = np.sum(origin_target_flux, axis=1)
    
    fit_pixel = np.load('./%s.npy'%prefix)
    fit_lightcurve = np.sum(fit_pixel, axis=1)
    
    ratio = np.divide(target_lightcurve, fit_lightcurve)
    
    origin_epoch_mask = epoch_mask.copy()
    epoch_mask = epoch_mask[data_mask>0]
    print epoch_mask.shape
    
    #load and normalize the PDC lightcurve
    
    star = client.star(kid)
    lc = star.get_light_curves(short_cadence=False)[quarter]
    data = lc.read()
    flux = data["PDCSAP_FLUX"]
    flux = flux[data_mask>0]
    inds = np.isfinite(flux)
    flux = flux[inds]
    pdc_time = data["TIME"]
    pdc_time = pdc_time[data_mask>0]
    pdc_time = pdc_time[inds]
    
    pdc_mean = np.mean(flux)
    flux = flux/pdc_mean
    
    #construct the running mean and std
    mean_list = np.zeros_like(epoch_mask, dtype=float)
    std_list = np.zeros_like(epoch_mask, dtype=float)
    pdc_mean_list = np.zeros_like(epoch_mask, dtype=float)
    pdc_std_list = np.zeros_like(epoch_mask, dtype=float)
    half_group_length = 6*24
    for i in range(0, epoch_mask.shape[0]):
        group_mask = np.zeros_like(epoch_mask)
        if i <= half_group_length:
            group_mask[0:i+half_group_length+1] = 1
        elif i >= epoch_mask.shape[0]-half_group_length-1:
            group_mask[i-half_group_length:] = 1
        else:
            group_mask[i-half_group_length:i+half_group_length+1] = 1
        co_mask = group_mask[epoch_mask>0]
        mean_list[i] = np.mean(ratio[co_mask>0])
        std_list[i] = np.std(ratio[co_mask>0])
        co_mask = group_mask[inds]
        pdc_mean_list[i] = np.mean(flux[co_mask>0])
        pdc_std_list[i] = np.std(flux[co_mask>0])
    mean_list = mean_list[epoch_mask>0]
    std_list = std_list[epoch_mask>0]
    pdc_mean_list = pdc_mean_list[inds]
    pdc_std_list = pdc_std_list[inds]
    
    median_std = np.median(std_list)
    pdc_median_std = np.median(pdc_std_list)
    print median_std
    print pdc_median_std

    transit_mask = get_transit_mask(koi_num, time, epoch_mask, 3*24*2)
    pdc_transit_mask = transit_mask[inds]
    transit_mask = transit_mask[epoch_mask>0]

    origin_transit_mask = get_transit_mask(koi_num, origin_time, origin_epoch_mask, 3*24*2)
    origin_transit_mask = origin_transit_mask[origin_epoch_mask>0]
    
    #calculate the signal to noise ratio
    signal = np.mean(ratio[transit_mask==1])-np.mean(ratio[transit_mask>=2])
    pdc_signal = np.mean(flux[pdc_transit_mask==1])-np.mean(flux[pdc_transit_mask>=2])
    depth = np.mean(ratio[transit_mask==1])- np.mean(ratio[transit_mask==3])
    pdc_depth = np.mean(flux[pdc_transit_mask==1])- np.mean(flux[pdc_transit_mask==3])
    print (signal, depth)
    print (pdc_signal, pdc_depth)

    sn = signal/median_std
    pdc_sn =pdc_signal/pdc_median_std
    
    #plot the lightcurve
    koi = client.koi(koi_num)
    transit_duration = koi.koi_duration
    half_len = round(transit_duration/2/0.5)
    transit_period = half_len*2*0.5
    
    f, axes = plt.subplots(4, 1)
    axes[0].plot(origin_time[origin_transit_mask<2], origin_target_lightcurve[origin_transit_mask<2], '.b', markersize=1)
    axes[0].plot(origin_time[origin_transit_mask>=2], origin_target_lightcurve[origin_transit_mask>=2], '.k', markersize=1, label="Transit singal \n within %.1f hrs window"%transit_period)
    plt.setp( axes[0].get_xticklabels(), visible=False)
    plt.setp( axes[0].get_yticklabels(), visible=False)
    axes[0].set_ylabel("Data")
    ylim = axes[0].get_ylim()
    xlim = axes[0].get_xlim()
    axes[0].legend(loc=1, ncol=3, prop={'size':8})
    
    axes[1].plot(time, fit_lightcurve, '.b', markersize=1)
    plt.setp( axes[1].get_xticklabels(), visible=False)
    plt.setp( axes[1].get_yticklabels(), visible=False)
    axes[1].set_ylabel("Fit")
    axes[1].set_ylim(ylim)
    axes[1].set_xlim(xlim)
    
    axes[2].plot(time[transit_mask<2], ratio[transit_mask<2], '.b', markersize=2)
    axes[2].plot(time[transit_mask>=2], ratio[transit_mask>=2], '.k', markersize=2, label="Transit singal \n within %.1f hrs window"%transit_period)
    axes[2].plot(time, mean_list, 'r-')
    axes[2].plot(time, mean_list-std_list, 'r-')
    plt.setp( axes[2].get_xticklabels(), visible=False)
    axes[2].set_ylim(0.999,1.001)
    axes[2].set_ylabel("Ratio")

    axes[2].text(time[500], 1.0006, 'S/N = %.3f'%sn)
    axes[2].legend(loc=1, ncol=3, prop={'size':8})
    axes[2].set_xlim(xlim)

    
    #plot the PDC curve
    
    axes[3].plot(pdc_time[pdc_transit_mask<2], flux[pdc_transit_mask<2], '.b', markersize=2)
    axes[3].plot(pdc_time[pdc_transit_mask>=2], flux[pdc_transit_mask>=2], '.k', markersize=2, label="Transit singal \n within %.1f hrs window"%transit_period)
    axes[3].plot(pdc_time, pdc_mean_list, 'r-')
    axes[3].plot(pdc_time, pdc_mean_list-pdc_std_list, 'r-')
    axes[3].set_ylim(0.999,1.001)
    axes[3].yaxis.tick_right()
    axes[3].set_ylabel("pdc flux")
    axes[3].set_xlabel("time [BKJD]")
    axes[3].text(time[500], 1.0006, 'S/N = %.3f'%pdc_sn)
    axes[3].legend(loc=1, ncol=3, prop={'size':8})
    axes[3].set_xlim(xlim)


    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0, hspace=0)
    plt.suptitle('Kepler %d Quarter %d L2-Reg %.0e poly:%r\n Fit Source[Initial:%d Number Stars:%d Pixels:%d CCD:%r] Test Region:%d-%d'%(kid, quarter, l2, poly, offset+1, num, pixel_num, ccd, -margin, margin))
    plt.savefig('./%s.png'%prefix, dpi=190)

def plot_transit_part(kid, quarter, l2, offset, num, pixel_num, poly, ccd, target_flux, target_kplr_mask, epoch_mask, time, margin, prefix, auto, auto_offset, auto_window, transit_mask):
    target_kplr_mask = target_kplr_mask.flatten()
    target_kplr_mask = target_kplr_mask[target_kplr_mask>0]
    target_flux = target_flux[:, target_kplr_mask==3]
    
    #construct the lightcurve
    target_lightcurve = np.sum(target_flux, axis=1)
    
    fit_pixel = np.load('./%s.npy'%prefix)
    fit_lightcurve = np.sum(fit_pixel, axis=1)
    
    ratio = np.divide(target_lightcurve, fit_lightcurve)
    
    #load and normalize the PDC lightcurve
    star = client.star(kid)
    lc = star.get_light_curves(short_cadence=False)[quarter]
    data = lc.read()
    flux = data["PDCSAP_FLUX"]
    inds = np.isfinite(flux)
    flux = flux[inds]
    pdc_time = data["TIME"][inds]
    pdc_mean = np.mean(flux)
    flux = flux/pdc_mean

    #plot the lightcurve
    half_len = 6
    transit_period = half_len*2*0.5
    
    pdc_transit_mask = transit_mask[inds]
    transit_mask = transit_mask[epoch_mask>0]
    
    f, axes = plt.subplots(4, 1)
    axes[0].plot(time[transit_mask<2], target_lightcurve[transit_mask<2], '.b', markersize=1)
    axes[0].plot(time[transit_mask>=2], target_lightcurve[transit_mask>=2], '.k', markersize=1, label="Transit singal \n within %.1f hrs window"%transit_period)
    plt.setp( axes[0].get_xticklabels(), visible=False)
    plt.setp( axes[0].get_yticklabels(), visible=False)
    axes[0].set_ylabel("Data")
    ylim = axes[0].get_ylim()
    axes[0].legend(loc=1, ncol=3, prop={'size':8})
    
    axes[1].plot(time[transit_mask>0], fit_lightcurve[transit_mask>0], '.b', markersize=1)
    plt.setp( axes[1].get_xticklabels(), visible=False)
    plt.setp( axes[1].get_yticklabels(), visible=False)
    axes[1].set_ylabel("Fit")
    axes[1].set_ylim(ylim)
    axes[1].set_xlim(440,540)

    
    axes[2].plot(time[np.logical_and(transit_mask>0, transit_mask<2)], ratio[np.logical_and(transit_mask>0, transit_mask<2)], '.b', markersize=2)
    axes[2].plot(time[transit_mask>=2], ratio[transit_mask>=2], '.k', markersize=2, label="Transit singal \n within %.1f hrs window"%transit_period)
    #axes[2].plot(time, mean_list, 'r-')
    #axes[2].plot(time, mean_list-std_list, 'r-')
    plt.setp( axes[2].get_xticklabels(), visible=False)
    axes[2].set_ylim(0.999,1.001)
    axes[2].set_ylabel("Ratio")
    axes[2].set_xlim(440,540)

    
    #axes[2].text(time[2000], 1.0006, 'S/N = %.3f'%sn)
    axes[2].legend(loc=1, ncol=3, prop={'size':8})
    
    #plot the PDC curve
    
    axes[3].plot(pdc_time[np.logical_and(pdc_transit_mask>0, pdc_transit_mask<2)], flux[np.logical_and(pdc_transit_mask>0, pdc_transit_mask<2)], '.b', markersize=2)
    axes[3].plot(pdc_time[pdc_transit_mask>=2], flux[pdc_transit_mask>=2], '.k', markersize=2, label="Transit singal \n within %.1f hrs window"%transit_period)
    #axes[3].plot(pdc_time, pdc_mean_list, 'r-')
    #axes[3].plot(pdc_time, pdc_mean_list-pdc_std_list, 'r-')
    axes[3].set_ylim(0.999,1.001)
    axes[3].yaxis.tick_right()
    axes[3].set_ylabel("pdc flux")
    axes[3].set_xlabel("time [BKJD]")
    #axes[3].text(time[2000], 1.0006, 'S/N = %.3f'%pdc_sn)
    axes[3].legend(loc=1, ncol=3, prop={'size':8})
    axes[3].set_xlim(440,540)

    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0, hspace=0)
    plt.suptitle('Kepler %d Quarter %d L2-Reg %.0e poly:%r Auto:%r Window:+/-%d-%d\n Fit Source[Initial:%d Number Stars:%d Pixels:%d CCD:%r] Test Region:%d-%d'%(kid, quarter, l2, poly, auto, auto_offset, auto_offset+auto_window, offset+1, num, pixel_num, ccd, -margin, margin))
    plt.savefig('./%s.png'%prefix, dpi=190)


if __name__ == "__main__":

#generate lightcurve train-and-test, multithreads
    if True:
        kid = 5088536
        quarter = 5
        offset = 0
        total_num = 300
        filter_num = 108
        l2 = 1e5
        ccd = True
        auto = False
        poly = 0
        auto_offset = 0
        auto_window = 0
        margin = 12
        thread_num = 4
        prefix = 'kic%d/lightcurve_%d_q%d_num%d-%d_reg%.0e_poly%r_auto%r-%d-%d_margin%d_part1'%(kid, kid, quarter, offset+1, total_num, l2, poly, auto, auto_offset, auto_window, margin)
        rms = False
        
        koi_num = 282.01
        
        target_tpf, neighor_tpfs = find_mag_neighor(kid, quarter, total_num, offset=0, ccd=True)
        
        #neighor_tpfs = preprocess.filter(target_tpf, neighor_tpfs, 400, rms, filter_num)
        
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, data_mask, l2_vector, pixel_num = get_fit_matrix(target_tpf, neighor_tpfs, l2, poly, auto, auto_offset, auto_window, 1, 1)
        
        #transit_mask = get_transit_mask(koi_num, time, epoch_mask, 3*24*2)
        transit_mask = None
        
        fit_target(target_flux, target_kplr_mask, neighor_flux_matrix, time, epoch_mask[data_mask>0], covar_list, margin, l2_vector, thread_num, prefix, transit_mask)
        
        #plot_transit_part(kid, quarter, l2, offset, total_num, pixel_num, poly, ccd, target_flux, target_kplr_mask, epoch_mask, time, margin, prefix, auto, auto_offset, auto_window, transit_mask)
    
        plot_fit(kid, quarter, l2, offset, total_num, pixel_num, poly, ccd, target_flux, target_kplr_mask, epoch_mask, data_mask, time, margin, prefix, koi_num)

    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        total_num = 1
        filter_num = 108
        l2 = 1e5
        ccd = True
        auto = False
        poly = 0
        auto_offset = 0
        auto_window = 0
        rms = False
        
        koi_num = 282.01

        k = 10
        
        target_tpf, neighor_tpfs = find_mag_neighor(kid, quarter, total_num, offset=0, ccd=True)
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, data_mask, l2_vector, pixel_num = get_fit_matrix(target_tpf, neighor_tpfs, l2, poly, auto, auto_offset, auto_window, 1, 1)
        
        kfold_mask = get_kfold_train_mask(neighor_flux_matrix.shape[0], k, True)
        
        target_kplr_mask = target_kplr_mask.flatten()
        target_kplr_mask = target_kplr_mask[target_kplr_mask>0]
        
        optimal_len = np.sum(target_kplr_mask==3)
        print optimal_len
        target_flux = target_flux[:, target_kplr_mask==3]
        
        covar_list = covar_list[:, target_kplr_mask==3]
        covar = np.mean(covar_list, axis=1)**2
        
        l2_list = np.arange(3)+4
        num_list = (np.arange(7)+1)*100
        window_list = (np.arange(1))*2

        f = open('./k_fold_optimization/lightcurve_optimize_30_part1_large.log', 'w')
        for total_num in num_list:
            tmp_target_tpf, neighor_tpfs = find_mag_neighor(kid, quarter, total_num, offset=0, ccd=True)
            for auto_window in window_list:
                neighor_flux_matrix, tmp_target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, tmp_target_kplr_mask, epoch_mask, data_mask, l2_vector, pixel_num = get_fit_matrix(target_tpf, neighor_tpfs, 1., poly, auto, auto_offset, auto_window, 1, 1)
                f.write('%d\t'%total_num)
                #neighor_flux_matrix = neighor_flux_matrix[reduced_transit_mask<2]
                for l2 in l2_list:
                    l2 = math.pow(10, l2)
                    tmp_l2_vector = l2_vector*l2
                    mean_rms = 0
                    for i in range(0, k):
                        tmp_covar = covar[kfold_mask!=i]
                        result = lss.linear_least_squares(neighor_flux_matrix[kfold_mask!=i, :], target_flux[kfold_mask!=i,:], tmp_covar, tmp_l2_vector)
                        fit_flux = np.dot(neighor_flux_matrix[kfold_mask==i, :], result)
                        fit_lightcurve = np.sum(fit_flux, axis=1)
                        lightcurve = np.sum(target_flux[kfold_mask==i, :], axis=1)
                        ratio = np.divide(lightcurve, fit_lightcurve)
                        dev = ratio - 1.0
                        rms = np.sqrt(np.mean(dev**2, axis=0))
                        mean_rms += rms
                    mean_rms /= k
                    f.write('%.8e\t'%mean_rms)
                    print (total_num, auto_window, l2, mean_rms)
                f.write('\n')
        f.close()













