import kplr
import random
import numpy as np
import matplotlib.pyplot as plt
import leastSquareSolver as lss
import threading
import os
import math
import h5py
import sys

client = kplr.API()

sap_style = dict(color='w', linestyle='', marker='.', markersize=2, markerfacecolor='k', markeredgecolor='k', markevery=None)
cpm_prediction_style = dict(color='r', ls='-', lw=1, alpha=0.8)
cpm_style = dict(color='w', linestyle='', marker='.', markersize=2, markerfacecolor='r', markeredgecolor='r', markevery=None)
fit_prediction_style = dict(color='g', linestyle='-', lw=1, markersize=2, markerfacecolor='g', markeredgecolor='g', markevery=None)
fit_style = dict(color='w', linestyle='', marker='+', markersize=2, markerfacecolor='g', markeredgecolor='g', markevery=None)
best_prediction_style = dict(color='g', linestyle='-', marker='+', lw=1, markersize=1.5, markerfacecolor='g', markeredgecolor='g', markevery=None)
best_style = dict(color='w', linestyle='', marker='.', markersize=2, markerfacecolor='g', markeredgecolor='g', markevery=None)


def find_mag_neighbor(kic, quarter, num, offset=0, ccd=True):
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
    target_row, target_column = load_header(target_tpf)
    print target_row, target_column

    tpfs = {}

    if num != 0:
        load_num = int(math.ceil(1.2*(num+offset)))
        print load_num
        if ccd:
            stars_over = client.target_pixel_files(ktc_kepler_id="!=%d"%target_tpf.ktc_kepler_id, kic_kepmag=">=%f"%target_tpf.kic_kepmag, sci_data_quarter=target_tpf.sci_data_quarter, sci_channel=target_tpf.sci_channel, sort=("kic_kepmag", 1),ktc_target_type="LC", max_records=load_num)
            stars_under = client.target_pixel_files(ktc_kepler_id="!=%d"%target_tpf.ktc_kepler_id, kic_kepmag="<=%f"%target_tpf.kic_kepmag, sci_data_quarter=target_tpf.sci_data_quarter, sci_channel=target_tpf.sci_channel, sort=("kic_kepmag", -1),ktc_target_type="LC", max_records=load_num)
        else:
            stars_over = client.target_pixel_files(ktc_kepler_id="!=%d"%target_tpf.ktc_kepler_id, kic_kepmag=">=%f"%target_tpf.kic_kepmag, sci_data_quarter=target_tpf.sci_data_quarter, sci_channel="!=%d"%target_tpf.sci_channel, sort=("kic_kepmag", 1),ktc_target_type="LC", max_records=load_num)
            stars_under = client.target_pixel_files(ktc_kepler_id="!=%d"%target_tpf.ktc_kepler_id, kic_kepmag="<=%f"%target_tpf.kic_kepmag, sci_data_quarter=target_tpf.sci_data_quarter, sci_channel="!=%d"%target_tpf.sci_channel, sort=("kic_kepmag", -1),ktc_target_type="LC", max_records=load_num)

        #print(len(stars_over), len(stars_under))
        target_kepmag = target_tpf.kic_kepmag

        dtype = [('kic', int), ('bias', float), ('tpf', type(target_tpf))]
        neighbor_list = []
        tpf_list = stars_over+stars_under
        for tpf in tpf_list:
            tmp_row, tmp_column = load_header(tpf)
            if (tmp_row-target_row)**2+(tmp_column-target_column)**2 > 400:
                neighbor_list.append((tpf.ktc_kepler_id, math.fabs(tpf.kic_kepmag-target_kepmag), tpf))
        #print len(neighbor_list)

        neighbor_list = np.array(neighbor_list, dtype=dtype)
        neighbor_list = np.sort(neighbor_list, order='bias')

        for i in range(offset, offset+num):
            tmp_kic, tmp_bias, tmp_tpf = neighbor_list[i]
            tpfs[tmp_kic] = tmp_tpf
        print ('kic%d load tpf successfully'%kic)
    
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

#help function to load header from tpf
def load_header(tpf):
    with tpf.open() as file:
        meta = file[1].header
        column = meta['1CRV4P']
        row = meta['2CRV4P']
    return row,column

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

def load_data_r(tpf):
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
    flux = flux[epoch_mask>0,:]
    
    return flux, kplr_mask, column, row


def predictor_filter(predictor, num):
    total = predictor.shape[1]
    if total >= num:
        select_mask = np.zeros(total)
        select = random.sample(xrange(0, total), num)
        for i in select:
            select_mask[i] = 1
        predictor = predictor[:, select_mask>0]
    return predictor

def predictor_filter_bin(predictor, num, bin_num=5):
    total = predictor.shape[1]
    if total > num:
        count = []
        predictor_mean = np.mean(predictor, axis=0)
        count, bins = np.histogram(predictor_mean, bin_num)
        indices = np.digitize(predictor_mean, bins)
        max_indice = np.argmax(predictor_mean)
        if indices[max_indice] > bin_num:
            indices[max_indice] = bin_num
        select_mask = np.arange(total)
        select = np.zeros(total, dtype=int)
        bin_list = np.arange(bin_num)+1
        bin_list = bin_list[count>0]
        count = count[count>0]
        bin_num = bin_list.shape[0]
        remain = num
        while remain>0:
            sample_num = remain//bin_num
            min = np.min(count)
            pend_select = np.array([], dtype=int)
            if sample_num == 0:
                pend_select = np.concatenate((pend_select, random.sample(select_mask, remain)))
                remain -= remain
            else:
                if min<sample_num:
                    sample_num = min
                for i in bin_list:
                    pend_select = np.concatenate((pend_select, random.sample(select_mask[indices==i], sample_num)))
                remain -= sample_num*bin_num
                count -= sample_num
                bin_list =  bin_list[count>0]
                count = count[count>0]
                bin_num = bin_list.shape[0]
            print pend_select
            select_old = np.copy(select)
            for i in pend_select:
                select[i] = 1
            tmp_select = select[select_old<1]
            select_mask = select_mask[tmp_select<1]
            indices = indices[tmp_select<1]
        predictor = predictor[:, select>0]
        print remain
    print predictor.shape
    return predictor


def get_kfold_train_mask(length, k, rand=False):
    train_mask = np.ones(length, dtype=int)
    num_sequence = length//25
    if rand:
        '''
        for i in range(0, length):
            group = random.randint(0, k-1)
            train_mask[i] = group
        '''
        for i in range(0, num_sequence):
            group = random.randint(0, k-1)
            train_mask[i*25:(i+1)*25] = group
            train_mask[i*25+12] = group+k
        group = random.randint(0, k-1)
        train_mask[num_sequence*25:] = group
        if length - num_sequence*25>12:
            train_mask[num_sequence*25+12] = group+k
        else:
            train_mask[-1] = group+k
    else:
        step = length//k
        for i in range(0, k-1):
            train_mask[i*step:(i+1)*step] = i
        #train_mask[i*step+12] = i+k
        train_mask[(k-1)*step:] = k-1
        '''
        if length - (k-1)*step>12:
            train_mask[(k-1)*step+12] = 2*k-1
        else:
            train_mask[-1] = 2*k-1
        '''
    return train_mask

def get_fit_matrix(target_tpf, neighbor_tpfs, l2,  poly=0, auto=False, offset=0, window=0, auto_l2=0, part=None, filter=False, prefix='lightcurve'):
    """
    ## inputs:
    - `target_tpf` - target tpf
    - `neighbor_tpfs` - neighbor tpfs in magnitude
    - `auto` - if autorgression
    - `poly` - number of orders of polynomials of time need to be added
    
    ## outputs:
    - `neighbor_flux_matrix` - fitting matrix of neighbor flux
    - `target_flux` - target flux
    - `covar_list` - covariance matrix for every pixel
    - `time` - one dimension array of BKJD time
    - `neighbor_kid` - KIC number of the neighbor stars in the fitting matrix
    - `neighbor_kplr_maskes` - kepler maskes of the neighbor stars in the fitting matrix
    - `target_kplr_mask` - kepler mask of the target star
    - `epoch_mask` - epoch mask
    - `l2_vector` - array of L2 regularization strength
    """
    
    filename = "./%s"%prefix
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    f = h5py.File('%s.hdf5'%prefix, 'w')
    
    time, target_flux, target_pixel_mask, target_kplr_mask, epoch_mask, flux_err, column, row= load_data(target_tpf)

    neighbor_kid, neighbor_fluxes, neighbor_pixel_maskes, neighbor_kplr_maskes = [], [], [], []

    #construt the neighbor flux matrix
    pixel_num = 0
    if neighbor_tpfs:
        for key, tpf in neighbor_tpfs.items():
            tmpResult = load_data(tpf)
            pixel_num += tmpResult[1].shape[1]
            if pixel_num >= 4000:
                break
            if filter:
                neighbor_fluxes.append(predictor_filter(tmpResult[1], 30))
            else:
                neighbor_fluxes.append(tmpResult[1])
            neighbor_kid.append(key)
            neighbor_pixel_maskes.append(tmpResult[2])
            neighbor_kplr_maskes.append(tmpResult[3])
            epoch_mask *= tmpResult[4]
        neighbor_flux_matrix = np.float64(np.concatenate(neighbor_fluxes, axis=1))
        #print neighbor_flux_matrix.shape
    else:
        neighbor_flux_matrix = np.array([])
        #print neighbor_flux_matrix.size

    target_flux = np.float64(target_flux)

    epoch_len = epoch_mask.shape[0]
    data_mask = np.zeros(epoch_len, dtype=int)

    #construct l2 vectors
    pixel_num = neighbor_flux_matrix.shape[1]
    auto_pixel_num = 0
    l2_vector = np.ones(pixel_num, dtype=float)*l2
    '''
    l2_vector = np.array([])
    if neighbor_flux_matrix.size != 0:
        pixel_num = neighbor_flux_matrix.shape[1]
        l2_vector = np.ones(pixel_num, dtype=float)*l2
    else:
        pixel_num = 0
    '''

    #spilt lightcurve
    if part is not None:
        epoch_len = epoch_mask.shape[0]
        split_point = []
        split_point_forward = []
        split_point.append(0)
        i = 0
        while i <epoch_len:
            if epoch_mask[i] == 0:
                i += 1
            else:
                j = i+1
                while j<epoch_len and epoch_mask[j] == 0:
                    j += 1
                if j-i > 30 and j-split_point[-1] > 1200:
                    split_point.append(i)
                    split_point.append(j)
                i = j
        split_point.append(epoch_len)
        #print split_point
        #data_mask[split_point[part-1]:split_point[part]] = 1
        start = split_point[2*(part-1)]+offset+window
        end = split_point[2*(part-1)+1]-offset-window+1
        data_mask[start:end] = 1
        '''
        fit_epoch_mask =  np.split(fit_epoch_mask, split_point)[part-1]
        fit_time = np.split(time, split_point)[part-1]
        fit_target_flux = np.split(target_flux, split_point, axis=0)[part-1]
        flux_err = np.split(flux_err, split_point, axis=0)[part-1]
        neighbor_flux_matrix = np.split(neighbor_flux_matrix, split_point, axis=0)[part-1]

        print (fit_epoch_mask.shape, fit_time.shape, fit_target_flux.shape, flux_err.shape, neighbor_flux_matrix.shape)
        '''
    else:
        data_mask[:] = 1

    #add auto-regression terms
    if auto and (window != 0):
        #print 'auto'
        tmp_target_kplr_mask = target_kplr_mask.flatten()
        tmp_target_kplr_mask = tmp_target_kplr_mask[tmp_target_kplr_mask>0]
        auto_flux = target_flux[:, tmp_target_kplr_mask==3]
        for i in range(offset+1, offset+window+1):
            if neighbor_flux_matrix.size != 0:
                neighbor_flux_matrix = np.concatenate((neighbor_flux_matrix, np.roll(auto_flux, i, axis=0)), axis=1)
                neighbor_flux_matrix = np.concatenate((neighbor_flux_matrix, np.roll(auto_flux, -i, axis=0)), axis=1)
            else:
                neighbor_flux_matrix = np.roll(auto_flux, i, axis=0)
                neighbor_flux_matrix = np.concatenate((neighbor_flux_matrix, np.roll(auto_flux, -i, axis=0)), axis=1)
        data_mask[0:offset+window] = 0
        data_mask[-offset-window:] = 0
        auto_pixel_num = neighbor_flux_matrix.shape[1] - pixel_num
        l2_vector = np.concatenate((l2_vector, np.ones(auto_pixel_num, dtype=float)*auto_l2), axis=0)
        '''
        other_pixel_num = pixel_num
        pixel_num = neighbor_flux_matrix.shape[1]
        if l2_vector.size != 0:
            l2_vector = np.concatenate((l2_vector, np.ones(pixel_num-other_pixel_num, dtype=float)*auto_l2), axis=0)
        else:
            l2_vector = np.ones(pixel_num-other_pixel_num, dtype=float)*auto_l2
        '''

    #remove bad time point based on simulteanous epoch mask
    co_mask = data_mask*epoch_mask
    time = time[co_mask>0]
    target_flux = target_flux[co_mask>0]
    flux_err = flux_err[co_mask>0]
    neighbor_flux_matrix = neighbor_flux_matrix[co_mask>0, :]

    #print neighbor_flux_matrix.shape

    #add polynomial terms
    if poly is not None:
        time_mean = np.mean(time)
        time_std = np.std(time)
        nor_time = (time-time_mean)/time_std
        p = np.polynomial.polynomial.polyvander(nor_time, poly)
        neighbor_flux_matrix = np.concatenate((neighbor_flux_matrix, p), axis=1)
        l2_vector = np.concatenate((l2_vector, np.zeros(poly+1)), axis=0)
        '''
        median = np.median(target_flux)
        fourier_flux = []
        period = 30.44
        print('month:%f, time difference:%f'%(time[-1]-time[0], time[1]-time[0]))
        for i in range(1,33):
            fourier_flux.append(median*np.sin(2*np.pi*i*(time-time[0])/2./30.44).reshape(time.shape[0],1))
            fourier_flux.append(median*np.cos(2*np.pi*i*(time-time[0])/2./30.44).reshape(time.shape[0],1))
        print fourier_flux[0].shape
        fourier_components = np.concatenate(fourier_flux, axis=1)
        print fourier_components.shape
        neighbor_flux_matrix = np.concatenate((neighbor_flux_matrix, fourier_components), axis=1)
        l2_vector = np.concatenate((l2_vector, np.ones(fourier_components.shape[1])), axis=0)
        '''
    #print neighbor_flux_matrix.shape

    f.attrs['kid'] = target_tpf.ktc_kepler_id
    f.attrs['quarter'] = target_tpf.sci_data_quarter
    f.attrs['part'] = part

    data_group = f.create_group('data')
    cpm_info = f.create_group('cpm_info')
    
    data_group['ccd'] = target_tpf.sci_channel
    data_group['target_flux'] = target_flux
    data_group['time'] = time
    data_group['target_kplr_mask'] = target_kplr_mask
    data_group['epoch_mask'] = epoch_mask
    data_group['data_mask'] = data_mask

    cpm_info['pixel_num'] = pixel_num
    cpm_info['auto_pixel_num'] = auto_pixel_num
    cpm_info['neighbor_kid'] = neighbor_kid
    cpm_info['l2'] = l2
    if auto:
        cpm_info['auto'] = 1
    else:
        cpm_info['auto'] = 0
    cpm_info['auto_l2'] = auto_l2
    cpm_info['auto_window'] = window
    cpm_info['auto_offset'] = offset
    cpm_info['poly'] = poly

    print('kic%d load matrix successfully'%target_tpf.ktc_kepler_id)

    f.close()

    return neighbor_flux_matrix, target_flux, flux_err, time, neighbor_kid, neighbor_kplr_maskes, target_kplr_mask, epoch_mask, data_mask, l2_vector, pixel_num, auto_pixel_num

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
    #print transit_list
    whole_time = np.zeros_like(epoch_mask, dtype=float)
    whole_time[epoch_mask>0] = np.float64(time)
    
    half_len = round(transit_duration/2/0.49044)

    transit_mask = np.zeros_like(epoch_mask)
    for i in range(0,len(transit_list)):
        loc = (np.abs(whole_time-transit_list[i])).argmin()
        transit_mask[loc-measure_half_len:loc+measure_half_len+1] = 1
        transit_mask[loc-half_len:loc+half_len+1] = 2
        transit_mask[loc] = 3
    return transit_mask

def get_transit_boundary(transit_mask):
    transit_boundary = []
    length = transit_mask.shape[0]
    for i in range(1, length-1):
        if transit_mask[i]==1 and (transit_mask[i+1]==2 or transit_mask[i-1]==2):
            transit_boundary.append(i)
    return transit_boundary

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
    target_flux[position:position+length, :] = target_flux[position:position+length, :]*strength
    return target_flux, position

def fit_target(target_flux, target_kplr_mask, neighbor_flux_matrix, time, epoch_mask, covar_list, margin, l2_vector=None, thread_num=1, prefix="lightcurve", transit_mask=None):
    """
    ## inputs:
    - `target_flux` - target flux
    - `target_kplr_mask` - kepler mask of the target star
    - `neighbor_flux_matrix` - fitting matrix of neighbor flux
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
    filename = "./%s"%prefix
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    f = h5py.File('%s.hdf5'%prefix, 'a')
    cpm_info = f['/cpm_info']
    data_group = f['/data']
    cpm_info['margin'] = margin
    
    target_kplr_mask = target_kplr_mask.flatten()
    target_kplr_mask = target_kplr_mask[target_kplr_mask>0]

    optimal_len = np.sum(target_kplr_mask==3)
    #print optimal_len
    target_flux = target_flux[:, target_kplr_mask==3]

    covar_list = covar_list[:, target_kplr_mask==3]
    covar = np.mean(covar_list, axis=1)**2
    fit_flux = []
    fit_coe = []
    length = target_flux.shape[0]
    total_length = epoch_mask.shape[0]
    
    thread_len = total_length//thread_num
    last_len = total_length - (thread_num-1)*thread_len
    
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
                train_mask = np.ones(total_length)
                if i<margin:
                    train_mask[0:i+margin+1] = 0
                elif i > total_length-margin-1:
                    train_mask[i-margin:] = 0
                else:
                    train_mask[i-margin:i+margin+1] = 0
                train_mask = train_mask[epoch_mask>0]
                
                tmp_covar = covar[train_mask>0]
                result = lss.linear_least_squares(neighbor_flux_matrix[train_mask>0], target_flux[train_mask>0], tmp_covar, l2_vector)
                tmp_fit_flux[time_stp, :] = np.dot(neighbor_flux_matrix[time_stp+self.time_initial, :], result)
                #np.save('./%stmp%d.npy'%(prefix, self.thread_id), tmp_fit_flux)
                time_stp += 1
                #print('done%d'%i)
            print('Exiting%d'%self.thread_id)
            np.save('./%stmp%d.npy'%(prefix, self.thread_id), tmp_fit_flux)
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

    data_group['fit_flux'] = fit_flux
    f.close()

    for i in range(0, thread_num):
        os.remove('./%stmp%d.npy'%(prefix, i))

def fit_target_pixel(target_flux, target_kplr_mask, neighbor_flux_matrix, time, epoch_mask, covar_list, margin, l2_vector=None, thread_num=1, prefix="lightcurve", transit_mask=None):
    """
        ## inputs:
        - `target_flux` - target flux
        - `target_kplr_mask` - kepler mask of the target star
        - `neighbor_flux_matrix` - fitting matrix of neighbor flux
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
    target_flux = target_flux[:, target_kplr_mask==3]
    
    mean = np.mean(target_flux, axis=0)
    pixel = np.argmax(mean)
    print pixel
    target_flux = target_flux[:, pixel]
    
    covar_list = covar_list[:, target_kplr_mask==3]
    covar_list = covar_list[:, pixel]
    covar = covar_list**2
    
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
            tmp_fit_flux = np.empty((self.time_len, 1))
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
                result = lss.linear_least_squares(neighbor_flux_matrix[train_mask>0], target_flux[train_mask>0], tmp_covar, l2_vector)
                tmp_fit_flux[time_stp] = np.dot(neighbor_flux_matrix[time_stp+self.time_initial, :], result)
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

def plot_fit(kid, quarter, l2, offset, num, pixel_num, auto_pixel_num, auto_l2, auto_window, auto_offset, poly, ccd, target_flux, target_kplr_mask, epoch_mask, data_mask, time, margin, prefix, koi_num, pixel=False):
    
    target_tpf = client.target_pixel_files(ktc_kepler_id=kid, sci_data_quarter=quarter, ktc_target_type="LC")[0]
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
        print pixel
        target_lightcurve = target_flux[:, pixel]
        origin_target_lightcurve = origin_target_flux[:, pixel]
        fit_lightcurve = np.load('./%s.npy'%prefix)
        ratio = np.divide(target_lightcurve, fit_lightcurve[:, 0])
    else:
        target_lightcurve = np.sum(target_flux, axis=1)
        origin_target_lightcurve = np.sum(origin_target_flux, axis=1)
        
        fit_pixel = np.load('./%s.npy'%prefix)
        fit_lightcurve = np.sum(fit_pixel, axis=1)
        ratio = np.divide(target_lightcurve, fit_lightcurve)
    
    print origin_target_lightcurve.shape
    print target_lightcurve.shape
    print fit_lightcurve.shape
    print ratio.shape
    print ratio
    
    origin_epoch_mask = epoch_mask.copy()
    epoch_mask = epoch_mask[data_mask>0]
    print epoch_mask.shape
    
    #load and normalize the PDC lightcurve
    print data_mask.shape
    star = client.star(kid)
    lc = star.get_light_curves(short_cadence=False)[quarter-2]
    lc = client.light_curves(ktc_kepler_id=kid, sci_data_quarter=quarter, ktc_target_type="LC")[0]
    data = lc.read()
    flux = data["PDCSAP_FLUX"]
    print flux.shape
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

    if koi_num is not None:
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
        half_len = round(transit_duration/2/0.49044)
        transit_period = half_len*2*0.5
    
    if pixel:
        f, axes = plt.subplots(3, 1)
    else:
        f, axes = plt.subplots(4, 1)
    if koi_num is not None:
        axes[0].plot(origin_time[origin_transit_mask<2], origin_target_lightcurve[origin_transit_mask<2], '.b', markersize=1)
        #axes[0].plot(origin_time[origin_transit_mask>=2], origin_target_lightcurve[origin_transit_mask>=2], '.k', markersize=1, label="Transit singal \n within %.1f hrs window"%transit_period)
        axes[0].plot(origin_time[origin_transit_mask>=2], origin_target_lightcurve[origin_transit_mask>=2], '.k', markersize=1, label="Transit singal")
    else:
        axes[0].plot(origin_time, origin_target_lightcurve, '.b', markersize=1)
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

    if koi_num is not None:
        axes[2].plot(time[transit_mask<2], ratio[transit_mask<2], '.b', markersize=2)
        axes[2].plot(time[transit_mask>=2], ratio[transit_mask>=2], '.k', markersize=2, label="Transit singal \n within %.1f hrs window"%transit_period)
        axes[2].text(time[500], 1.0006, 'S/N = %.3f'%sn)
    else:
        axes[2].plot(time, ratio, '.b', markersize=2)
    axes[2].plot(time, mean_list, 'r-')
    axes[2].plot(time, mean_list-std_list, 'r-')
    axes[2].set_ylim(0.999,1.001)
    axes[2].set_ylabel("Ratio")
    #axes[2].legend(loc=1, ncol=3, prop={'size':8})
    axes[2].set_xlim(xlim)

    if pixel:
        axes[2].set_xlabel("time [BKJD]")
    else:
        plt.setp( axes[2].get_xticklabels(), visible=False)

    if not pixel:
        #plot the PDC curve
        if koi_num is not None:
            axes[3].plot(pdc_time[pdc_transit_mask<2], flux[pdc_transit_mask<2], '.b', markersize=2)
            axes[3].plot(pdc_time[pdc_transit_mask>=2], flux[pdc_transit_mask>=2], '.k', markersize=2, label="Transit singal \n within %.1f hrs window"%transit_period)
            axes[3].text(time[500], 1.0006, 'S/N = %.3f'%pdc_sn)
        else:
            axes[3].plot(pdc_time, flux, '.b', markersize=2)
        axes[3].plot(pdc_time, pdc_mean_list, 'r-')
        axes[3].plot(pdc_time, pdc_mean_list-pdc_std_list, 'r-')
        axes[3].set_ylim(0.999,1.001)
        axes[3].yaxis.tick_right()
        axes[3].set_ylabel("pdc flux")
        axes[3].set_xlabel("time [BKJD]")
        #axes[3].legend(loc=1, ncol=3, prop={'size':8})
        axes[3].set_xlim(xlim)
        plt.suptitle('KIC%d Q%d Aperture flux Mag:%f poly:%r Test Region:%d-%d\n Star[Number:%d Pixels:%d L2:%.0e] Auto[Window:%d Pixels:%d L2:%.0e]'%(kid, quarter, target_tpf.kic_kepmag, poly, -margin, margin, num, pixel_num, l2, auto_window, auto_pixel_num, auto_l2))
    else:
        plt.suptitle('Kepler %d Quarter %d Pixel flux L2-Reg %.0e poly:%r Mag:%f\n Fit Source[Initial:%d Number Stars:%d Pixels:%d CCD:%r] Test Region:%d-%d'%(kid, quarter, l2, poly, target_tpf.kic_kepmag, offset+1, num, pixel_num, ccd, -margin, margin))


    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0, hspace=0)
    
    plt.savefig('./%s.png'%prefix, dpi=190)

def get_plot_info(prefix, koi_num=None, pixel=False):
    
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
    '''
    print origin_target_lightcurve.shape
    print target_lightcurve.shape
    print fit_lightcurve.shape
    print ratio.shape
    print ratio
    '''
    origin_epoch_mask = epoch_mask.copy()
    epoch_mask = epoch_mask[data_mask>0]
    #print epoch_mask.shape
    
    #load and normalize the PDC lightcurve
    '''
    star = client.star(kid)
    lc = client.light_curves(ktc_kepler_id=kid, sci_data_quarter=quarter, ktc_target_type="LC")[0]
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
    '''
    
    #construct the running mean and std
    mean_list = np.zeros_like(epoch_mask, dtype=float)
    std_list = np.zeros_like(epoch_mask, dtype=float)
    #pdc_mean_list = np.zeros_like(epoch_mask, dtype=float)
    #pdc_std_list = np.zeros_like(epoch_mask, dtype=float)
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
        #co_mask = group_mask[inds]
        #pdc_mean_list[i] = np.mean(flux[co_mask>0])
        #pdc_std_list[i] = np.std(flux[co_mask>0])
    mean_list = mean_list[epoch_mask>0]
    std_list = std_list[epoch_mask>0]
    #pdc_mean_list = pdc_mean_list[inds]
    #pdc_std_list = pdc_std_list[inds]
    
    median_std = np.median(std_list)
    #pdc_median_std = np.median(pdc_std_list)
    #print median_std
    #print pdc_median_std

    if koi_num is not None:
        transit_mask = get_transit_mask(koi_num, time, epoch_mask, 3*24*2)
        #pdc_transit_mask = transit_mask[inds]
        transit_mask = transit_mask[epoch_mask>0]

        origin_transit_mask = get_transit_mask(koi_num, origin_time, origin_epoch_mask, 3*24*2)
        origin_transit_mask = origin_transit_mask[origin_epoch_mask>0]
        
        #calculate the signal to noise ratio
        signal = np.mean(ratio[transit_mask==1])-np.mean(ratio[transit_mask>=2])
        #pdc_signal = np.mean(flux[pdc_transit_mask==1])-np.mean(flux[pdc_transit_mask>=2])
        depth = np.mean(ratio[transit_mask==1])- np.mean(ratio[transit_mask==3])
        #pdc_depth = np.mean(flux[pdc_transit_mask==1])- np.mean(flux[pdc_transit_mask==3])
        #print (signal, depth)
        #print (pdc_signal, pdc_depth)

        sn = signal/median_std
        #pdc_sn =pdc_signal/pdc_median_std
    
        #plot the lightcurve
        koi = client.koi(koi_num)
        transit_duration = koi.koi_duration
        half_len = round(transit_duration/2/0.5)
        transit_period = half_len*2*0.5
    else:
        transit_mask = None
        sn = None

    return time, target_lightcurve, fit_lightcurve, ratio, transit_mask, mean_list, std_list, sn, ccd

def plot_lightcurve(axe, num, time, lightcurve, **kwargs):
    for i in range(0, num):
        axe.plot(time[i], lightcurve[i],  **kwargs)

def plot_ratio(axe, num, time, ratio, mean_list, std_list, transit_mask, **kwargs):
    for i in range(0, num):
        axe.plot(time[i], ratio[i], **kwargs)
        axe.plot(time[i], mean_list[i], '-b', zorder=4)
        axe.fill_between(time[i], mean_list[i]-std_list[i], mean_list[i]+std_list[i], facecolor='green', alpha=0.2, interpolate=True, zorder=10)
    axe.set_ylim(-1., 0.99)

    total_time = np.concatenate(time, axis=0)
    total_ratio = np.concatenate(ratio, axis=0)
    total_transit_mask = np.concatenate(transit_mask, axis=0)
    total_std_list = np.concatenate(std_list, axis=0)

    transit_boundary = get_transit_boundary(total_transit_mask)
    mean_std = np.mean(total_std_list)
    ratio_depth = np.mean(total_ratio[total_transit_mask==3])
    for j in transit_boundary:
        x = total_time[j]
        axe.axvline(x=x, ymin=(ratio_depth+1.)/1.999,ymax=(mean_std+1.)/1.999, c="b",linewidth=0.5,zorder=4)


if __name__ == "__main__":
#generate lightcurve train-and-test, multithreads
    if True:
        print sys.argv[1]
        kid = int(sys.argv[1])#5000456#8880157#10187017#5695396#5088536#6196457#6442183#5088536#10005473#8866102#8150320
        print kid
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
        thread_num = 1
        part = 1
        
        for part in range(1,4):
            
            prefix = 'kic%d/lightcurve_%d_q%d_num%d-%d_reg%.0e_poly%r_auto%r-%d-%d-%.0e_margin%d_part%d_default_pixel'%(kid, kid, quarter, offset+1, total_num, l2, poly, auto, auto_offset, auto_window, auto_l2, margin, part)
            if  os.path.exists('%s.hdf5'%prefix):
                continue
            koi_num = 117.01
        
            target_tpf, neighbor_tpfs = find_mag_neighbor(kid, quarter, total_num, offset=0, ccd=True)
        
            neighbor_flux_matrix, target_flux, covar_list, time, neighbor_kid, neighbor_kplr_maskes, target_kplr_mask, epoch_mask, data_mask, l2_vector, pixel_num, auto_pixel_num = get_fit_matrix(target_tpf, neighbor_tpfs, l2, poly, auto, auto_offset, auto_window, auto_l2, part, False, prefix)
        
            transit_mask = None
        
            fit_target(target_flux, target_kplr_mask, neighbor_flux_matrix, time, epoch_mask[data_mask>0], covar_list, margin, l2_vector, thread_num, prefix, transit_mask)
            print('kic%d part%d done'%(kid, part))
        
            #plot_fit(kid, quarter, l2, offset, total_num, pixel_num, auto_pixel_num, auto_l2, auto_window, auto_offset, poly, ccd, target_flux, target_kplr_mask, epoch_mask, data_mask, time, margin, prefix, koi_num)

#Single pixel distortion
    if False:
        kid = 9822284#5905728#1575873#8396660#7625138#5088536
        quarter = 5
        offset = 0
        total_num = 160
        filter_num = 108
        l2 = 1e5
        auto_l2 = 0
        ccd = True
        auto = False
        poly = 0
        auto_offset = 0
        auto_window = 0
        margin = 24
        thread_num = 3
        part = 2
        distort_len = 20
        distort_str = 1.0005
        prefix = 'kic%d/lightcurve_%d_q%d_num%d-%d_reg%.0e_poly%r_auto%r-%d-%d-%.0e_margin%d_part%r_distort_%d'%(kid, kid, quarter, offset+1, total_num, l2, poly, auto, auto_offset, auto_window, auto_l2, margin, part, distort_len)
        rms = False
        
        target_tpf, neighbor_tpfs = find_mag_neighbor(kid, quarter, total_num, offset=0, ccd=True)
        
        ccd = target_tpf.sci_channel
        
        neighbor_flux_matrix, target_flux, covar_list, time, neighbor_kid, neighbor_kplr_maskes, target_kplr_mask, epoch_mask, data_mask, l2_vector, pixel_num, auto_pixel_num = get_fit_matrix(target_tpf, neighbor_tpfs, l2, poly, auto, auto_offset, auto_window, auto_l2, part)
        
        target_flux, position = get_fake_data(target_flux, distort_len, distort_str)
        
        #fit_target_pixel(target_flux, target_kplr_mask, neighbor_flux_matrix, time, epoch_mask[data_mask>0], covar_list, margin, l2_vector, thread_num, prefix, transit_mask)
        
        prefix_fit = 'kic%d/lightcurve_%d_q%d_num%d-%d_reg%.0e_poly%r_auto%r-%d-%d-%.0e_margin%d_part%r_distort_%d_fit'%(kid, kid, quarter, offset+1, total_num, l2, poly, auto, auto_offset, auto_window, auto_l2, margin, part, distort_len)
        
        target_kplr_mask_ori = np.copy(target_kplr_mask)
        target_kplr_mask = target_kplr_mask.flatten()
        target_kplr_mask = target_kplr_mask[target_kplr_mask>0]
    
        optimal_len = np.sum(target_kplr_mask==3)
        target_flux = target_flux[:, target_kplr_mask==3]
    
        mean = np.mean(target_flux, axis=0)
        pixel = np.argmax(mean)
        target_flux_pixel = target_flux[:, pixel]
        pixel_num_mask_r = np.zeros(target_kplr_mask.shape[0])
        pixel_num_mask_r[pixel] = 1
        pixel_num_mask = np.zeros(target_kplr_mask_ori.flatten().shape[0])
        pixel_num_mask[target_kplr_mask>0] = pixel_num_mask_r
        pixel_num = np.argmax(pixel_num_mask)
        pixel_row = pixel_num//target_kplr_mask_ori.shape[1]
        pixel_column = pixel_num-pixel_row*target_kplr_mask_ori.shape[1]
        print (pixel_row, pixel_column)
    
        covar_list = covar_list[:, target_kplr_mask==3]
        covar_list = covar_list[:, pixel]
        covar = covar_list**2
        
        result = lss.linear_least_squares(neighbor_flux_matrix, target_flux_pixel, covar, l2_vector)
        fit = np.dot(neighbor_flux_matrix, result)
        
        show_mask = np.zeros(fit.shape[0])
        show_mask[position-3*24*2:position+distort_len+3*24*2] = 1

        cpm = np.load("%s.npy"%prefix)[:,0]
        
        ratio_cpm = np.divide(target_flux_pixel, cpm)
        ratio_fit = np.divide(target_flux_pixel, fit)
        
        ratio_cpm = (ratio_cpm-1.0)*1000.
        ratio_fit = (ratio_fit-1.0)*1000.

        target_flux_pixel = (target_flux_pixel/np.median(target_flux_pixel)-1.)*1000.
        fit = (fit/np.median(fit)-1.)*1000.
        cpm = (cpm/np.median(cpm)-1.)*1000.
        
        cpm_signal = np.mean(ratio_cpm[position:position+distort_len])
        fit_signal = np.mean(ratio_fit[position:position+distort_len])
        
        begin = time[position-3*24*2]
        end = time[position+distort_len+3*24*2-1]
        print(begin, end)
        length = end-begin
        begin_day = time[position]
        end_day = time[position+distort_len]
        signal_begin = (time[position]-begin)/length
        signal_end = (time[position+distort_len]-begin)/length
        
        time = time[show_mask>0]
        target_flux_pixel = target_flux_pixel[show_mask>0]
        fit = fit[show_mask>0]
        cpm = cpm[show_mask>0]
        ratio_fit = ratio_fit[show_mask>0]
        ratio_cpm = ratio_cpm[show_mask>0]

        f, axes = plt.subplots(2, 1)
        plot_lightcurve(axes[0], 1, [time], [target_flux_pixel], **sap_style)
        plot_lightcurve(axes[0], 1, [time], [fit], **fit_prediction_style)
        plot_lightcurve(axes[0], 1, [time], [cpm], **cpm_prediction_style)

        axes[0].set_ylabel('SAP Flux [PPT]')
        axes[0].set_xlim(begin, end)
        plt.setp( axes[0].get_xticklabels(), visible=False)
        #plt.setp( axes[0].get_yticklabels(), visible=False)

        axes[0].text(0.65, 0.95, 'KIC %d Pixel(%d,%d)\nCCD Channel %d'%(kid, pixel_row, pixel_column, ccd), transform=axes[0].transAxes, fontsize=12,
        verticalalignment='top')

        axes[1].plot(time, ratio_fit, **fit_style)
        axes[1].plot(time, ratio_cpm, **cpm_style)
        axes[1].set_ylim(-1., 0.99)
        axes[1].set_ylabel('CPM/Fit Flux [PPT]')
        axes[1].set_xlabel('time [BKJD]')
        axes[1].set_xlim(begin, end)

        axes[1].axhline(y=(distort_str-1.)*1000.,xmin=0,xmax=signal_begin,c="k",linewidth=0.5,zorder=10)
        axes[1].axhline(y=(distort_str-1.)*1000.,xmin=signal_end,xmax=3,c="k",linewidth=0.5,zorder=10)
        axes[1].axhline(y=cpm_signal,xmin=signal_begin,xmax=signal_end,c="r",linewidth=1,zorder=10, alpha=1)
        axes[1].axhline(y=fit_signal,xmin=signal_begin,xmax=signal_end,c="g",linewidth=1,zorder=10, alpha=1)


        axes[1].axvline(x=begin_day, ymin=0,ymax=3,c="blue",linewidth=0.5,zorder=10)
        axes[1].axvline(x=end_day, ymin=0,ymax=3,c="blue",linewidth=0.5,zorder=10)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0, hspace=0)
        plt.savefig('distortion_%d_normal_try.png'%kid, dpi=190)
        plt.clf()

#Compare default and best
    if False:
        kid = 10187017#11295426#5000456#8880157#10187017#5695396#5088536#6196457#6442183#5088536#10005473#8866102#8150320
        quarter = 5
        offset = 0
        total_num = 160
        filter_num = 108
        l2 = 1e5
        auto_l2 = 1e5
        ccd = True
        auto = True
        poly = 0
        auto_offset = 12
        auto_window = 3
        margin = 12
        thread_num = 1
        part = 2
        prefix = 'kic%d/lightcurve_%d_q%d_num%d-%d_reg%.0e_poly%r_auto%r-%d-%d-%.0e_margin%d_part%r_default_pixel'%(kid, kid, quarter, offset+1, total_num, l2, poly, auto, auto_offset, auto_window, auto_l2, margin, part)
        rms = False
        
        koi_num = 82.01#82.03#283.02#285.01#None#282.01#42.01#904.02#282.01
        
        target_tpf, neighbor_tpfs = find_mag_neighbor(kid, quarter, total_num, offset=0, ccd=True)
                
        neighbor_flux_matrix, target_flux, covar_list, time, neighbor_kid, neighbor_kplr_maskes, target_kplr_mask, epoch_mask, data_mask, l2_vector, pixel_num, auto_pixel_num = get_fit_matrix(target_tpf, neighbor_tpfs, l2, poly, auto, auto_offset, auto_window, auto_l2, part, False)
        
        fit_target(target_flux, target_kplr_mask, neighbor_flux_matrix, time, epoch_mask[data_mask>0], covar_list, margin, l2_vector, thread_num, prefix, transit_mask)
    
        time_1, target_lightcurve_1, fit_lightcurve_1, ratio_1, transit_mask_1, mean_list_1, std_list_1, sn_1 = get_plot_info(kid, quarter, l2, offset, total_num,  auto_l2, auto_window, auto_offset, poly, ccd, target_flux, target_kplr_mask, epoch_mask, data_mask, time, margin, prefix, koi_num, False)
        
        total_num = 60
        l2 = 1e7
        auto_l2 = 1e5
        auto = True
        auto_offset = 12
        auto_window = 9
        prefix = 'kic%d/lightcurve_%d_q%d_num%d-%d_reg%.0e_poly%r_auto%r-%d-%d-%.0e_margin%d_part%r'%(kid, kid, quarter, offset+1, total_num, l2, poly, auto, auto_offset, auto_window, auto_l2, margin, part)
        
        target_tpf, neighbor_tpfs = find_mag_neighbor(kid, quarter, total_num, offset=0, ccd=True)

        neighbor_flux_matrix, target_flux, covar_list, time, neighbor_kid, neighbor_kplr_maskes, target_kplr_mask, epoch_mask, data_mask, l2_vector, pixel_num, auto_pixel_num = get_fit_matrix(target_tpf, neighbor_tpfs, l2, poly, auto, auto_offset, auto_window, auto_l2, part, False)

        time_2, target_lightcurve_2, fit_lightcurve_2, ratio_2, transit_mask_2, mean_list_2, std_list_2, sn_2 = get_plot_info(kid, quarter, l2, offset, total_num,  auto_l2, auto_window, auto_offset, poly, ccd, target_flux, target_kplr_mask, epoch_mask, data_mask, time, margin, prefix, koi_num, False)

        transit_boundary = get_transit_boundary(transit_mask_2)
        print transit_boundary
        print transit_mask_2.shape
        
        ccd = target_tpf.sci_channel
        
        target_lightcurve_2 = (target_lightcurve_2/np.median(target_lightcurve_2)-1.)*1000.
        fit_lightcurve_1 = (fit_lightcurve_1/np.median(fit_lightcurve_1)-1.)*1000.
        fit_lightcurve_2 = (fit_lightcurve_2/np.median(fit_lightcurve_2)-1.)*1000.
        
        mean_list_1 = (mean_list_1-1.)*1000.
        mean_list_2 = (mean_list_2-1.)*1000.
        std_list_1 = std_list_1*1000.
        std_list_2 = std_list_2*1000.

        f, axes = plt.subplots(3, 1)
        axes[0].plot(time_2, target_lightcurve_2, **sap_style)
        axes[0].plot(time_2, fit_lightcurve_2, **best_prediction_style)
        axes[0].plot(time_1, fit_lightcurve_1, **cpm_prediction_style)
        axes[0].set_ylabel('SAP Flux [PPT]')
        plt.setp( axes[0].get_xticklabels(), visible=False)
        #plt.setp( axes[0].get_yticklabels(), visible=False)

        axes[0].text(0.75, 0.95, 'KIC %d\nCCD Channel %d'%(kid, ccd), transform=axes[0].transAxes, fontsize=12,
        verticalalignment='top')
        
        ratio_1 = (ratio_1-1.)*1000.
        ratio_2 = (ratio_2-1.)*1000.

        plot_ratio(axes[1], 1, [time_2], [ratio_2], [mean_list_2], [std_list_2], [transit_mask_2], **best_style)
        plt.setp( axes[1].get_xticklabels(), visible=False)
        axes[1].set_ylabel('CPM Flux [PPT]')
        axes[1].text(0.75, 0.95, 'Best S/N: %.3f'%sn_2, transform=axes[1].transAxes, fontsize=12,
                     verticalalignment='top')
                     
        plot_ratio(axes[2], 1, [time_1], [ratio_1], [mean_list_1], [std_list_1], [transit_mask_1], **cpm_style)
        axes[2].text(0.75, 0.95, 'Default S/N: %.3f'%sn_1, transform=axes[2].transAxes, fontsize=12,
                     verticalalignment='top')
        axes[2].set_ylabel('CPM Flux [PPT]')
        axes[2].set_xlabel('time [BKJD]')

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0, hspace=0)
        plt.savefig('compare_%d_try.png'%kid, dpi=190)
        plt.clf()

    if False:
        data_list = [(6677841, 1236.02), (9950612, 719.01), (5966154, 655.01), (9451706, 271.02), (5695396, 283.01), (11401755, 277.01), (8292840, 260.01), (8866102, 42.01), (11295426, 246.01), (9892816, 1955.01), (4914423, 108.01), (3544595, 69.01)]
        data_list = [(10875245, 117.01)]
        for kid, koi_num in data_list:
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
            
            all_time, all_target_lightcurve, all_fit_lightcurve, all_ratio, all_transit_mask, all_mean_list, all_std_list, all_sn = [],[],[],[],[],[],[],[]

            for part in range(1,4):
                prefix = 'kic%d/lightcurve_%d_q%d_num%d-%d_reg%.0e_poly%r_auto%r-%d-%d-%.0e_margin%d_part%d_default_pixel'%(kid, kid, quarter, offset+1, total_num, l2, poly, auto, auto_offset, auto_window, auto_l2, margin, part)
                tmp_time, tmp_target_lightcurve, tmp_fit_lightcurve, tmp_ratio, tmp_transit_mask, tmp_mean_list, tmp_std_list, tmp_sn, ccd = get_plot_info(prefix, koi_num, False)
                all_time.append(tmp_time)
                all_target_lightcurve.append(tmp_target_lightcurve)
                all_fit_lightcurve.append(tmp_fit_lightcurve)
                all_ratio.append((tmp_ratio-1.)*1000.)
                all_transit_mask.append(tmp_transit_mask)
                all_mean_list.append((tmp_mean_list-1.)*1000.)
                all_std_list.append(tmp_std_list*1000.)
                all_sn.append(tmp_sn)
            total_target_lightcurve = np.concatenate(all_target_lightcurve, axis=0)
            total_fit_lightcurve = np.concatenate(all_fit_lightcurve, axis=0)
            target_median = np.median(total_target_lightcurve)
            fit_median = np.median(total_fit_lightcurve)
            for i in range(0, 3):
                all_target_lightcurve[i] = (all_target_lightcurve[i]/target_median-1.)*1000.
                all_fit_lightcurve[i] = (all_fit_lightcurve[i]/fit_median-1.)*1000.

            f, axes = plt.subplots(2,1)
            plot_lightcurve(axes[0], 3, all_time, all_target_lightcurve, **sap_style)
            plot_lightcurve(axes[0], 3, all_time, all_fit_lightcurve, **cpm_prediction_style)
            axes[0].set_ylabel('SAP Flux/CPM Prediction [PPT]')
            axes[0].text(0.75, 0.95, 'KIC %d\nCCD Channel %d'%(kid, ccd), transform=axes[0].transAxes, fontsize=12,
            verticalalignment='top')

            plt.setp(axes[0].get_xticklabels(), visible=False)
            #plt.setp( axes[0].get_yticklabels(), visible=False)
            
            plot_ratio(axes[1], 3, all_time, all_ratio, all_mean_list, all_std_list, all_transit_mask, **cpm_style)
            axes[1].set_ylabel('CPM Flux [PPT]')
            axes[1].set_xlabel('time [BKJD]')

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
            
            plt.savefig('kic_%d.png'%kid, dpi=190)
            plt.clf()

    if False:

        kid_list = [5000454, 5000456]
        
        f, axes = plt.subplots(2,1)
        i=0
        for kid in kid_list:
            target_tpf = client.target_pixel_files(ktc_kepler_id=kid, sci_data_quarter=5, ktc_target_type="LC")[0]
            ccd = target_tpf.sci_channel
            time, flux, pixel_mask, kplr_mask, epoch_mask, flux_err, column, row = load_data(target_tpf)
            
            kplr_mask = kplr_mask.flatten()
            kplr_mask = kplr_mask[kplr_mask>0]

            time = time[epoch_mask>0]
            flux = flux[epoch_mask>0]
            print flux.shape
            print kplr_mask.shape
            lightcurve = np.sum(flux[:, kplr_mask==3], axis=1)
            lightcurve = (lightcurve/np.median(lightcurve)-1.)*1000.
            print time.shape
            print lightcurve.shape
            plot_lightcurve(axes[i], 1, [time], [lightcurve], **sap_style)
            axes[i].set_ylabel('SAP Flux [PPT]')
            if i==1:
                axes[i].text(0.75, 0.95, 'B KIC %d\nCCD Channel %d'%(kid, ccd), transform=axes[i].transAxes, fontsize=12,
                             verticalalignment='top')
                ylimt = axes[i].get_ylim()
                print ylimt
                axes[i].set_ylim(ylimt[0], ylimt[1]-1)
            else:
                axes[i].text(0.75, 0.95, 'A KIC %d\nCCD Channel %d'%(kid, ccd), transform=axes[i].transAxes, fontsize=12,
                             verticalalignment='top')
            plt.setp(axes[0].get_xticklabels(), visible=False)
            i+=1
        axes[1].set_xlabel('time [BKJD]')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
        plt.savefig('two_light.png', dpi=190)
        plt.clf()
        
        kid1 = 5000454
        kid2 = 5000456
        target_tpf1 = client.target_pixel_files(ktc_kepler_id=kid1, sci_data_quarter=5, ktc_target_type="LC")[0]
        target_tpf2 = client.target_pixel_files(ktc_kepler_id=kid2, sci_data_quarter=5, ktc_target_type="LC")[0]
        
        flux_1_t, kplr_mask_1, column_1, row_1 = load_data_r(target_tpf1)
        flux_2_t, kplr_mask_2, column_2, row_2 = load_data_r(target_tpf2)
        
        row_len_1 = kplr_mask_1.shape[0]
        column_len_1 = kplr_mask_1.shape[1]
        row_len_2 = kplr_mask_2.shape[0]
        column_len_2 = kplr_mask_2.shape[1]
        row_offset = row_2 - row_1
        column_offset = column_2 - column_1
        if row_len_2+row_offset >= row_len_1:
            total_row_len = row_len_2+row_offset
        else:
            total_row_len = row_len_1
        if column_len_2+column_offset >= column_len_1:
            total_column_len = column_len_2+column_offset
        else:
            total_column_len = column_len_1

        print (total_row_len, total_column_len)

        flux_1 = np.mean(flux_1_t, axis=0)
        flux_2 = np.mean(flux_2_t, axis=0)

        total_kplr_mask = np.zeros((total_row_len, total_column_len))
        total_kplr_mask_1 = np.zeros((total_row_len, total_column_len))
        total_kplr_mask_2 = np.zeros((total_row_len, total_column_len))
        total_flux = np.zeros((total_row_len, total_column_len))

        for i in range(0, total_row_len):
            for j in range(0, total_column_len):
                if i>row_len_1-1 or j>column_len_1-1:
                    total_kplr_mask[i,j] = kplr_mask_2[i-row_offset, j-column_offset]
                    total_kplr_mask_2[i,j] = kplr_mask_2[i-row_offset, j-column_offset]
                    total_flux[i,j] = flux_2[i-row_offset, j-column_offset]
                elif i-row_offset<0 or j-column_offset<0 or i-row_offset>row_len_2-1 or j-column_offset>column_len_2-1:
                    total_kplr_mask[i,j] = kplr_mask_1[i, j]
                    total_kplr_mask_1[i,j] = kplr_mask_1[i, j]
                    total_flux[i,j] = flux_1[i, j]
                else:
                    total_kplr_mask[i,j] = kplr_mask_1[i, j]+kplr_mask_2[i-row_offset, j-column_offset]
                    total_kplr_mask_1[i,j] = kplr_mask_1[i, j]
                    total_kplr_mask_2[i,j] = kplr_mask_2[i-row_offset, j-column_offset]
                    if kplr_mask_1[i, j] == 0:
                        total_flux[i, j] = flux_2[i-row_offset, j-column_offset]
                    elif kplr_mask_2[i-row_offset, j-column_offset] == 0:
                        total_flux[i, j] = flux_1[i, j]
                    else:
                        total_flux[i, j] = (flux_1[i, j]+flux_2[i-row_offset, j-column_offset])/2.

        total_flux = total_flux/np.max(total_flux[total_kplr_mask>0])
        total_flux = total_flux - 0.25

        fig = plt.figure()
        axes = fig.add_subplot(111)
        im = axes.imshow(total_flux, cmap='gray_r', aspect='auto', interpolation="none")
        
        axes.text(4, 0, 'A', fontsize=12, color='r')
        axes.text(4, 1, 'A', fontsize=12, color='r')
        axes.text(4, 2, 'A', fontsize=12, color='r')
        axes.text(4, 4, 'A', fontsize=12, color='r')
        axes.text(4, 5, 'A', fontsize=12, color='r')
        axes.text(4, 6, 'A', fontsize=12, color='r')
        axes.text(3, 2, 'A', fontsize=12, color='r')
        axes.text(3, 3, 'A', fontsize=12, color='r')
        axes.text(3, 4, 'A', fontsize=12, color='r')
        axes.text(2, 2, 'A', fontsize=12, color='r')
        axes.text(2, 4, 'A', fontsize=12, color='r')
        axes.text(5, 4, 'A', fontsize=12, color='r')
        axes.text(5, 5, 'A', fontsize=12, color='r')
        
        axes.text(5, 1, 'B', fontsize=12, color='b')
        axes.text(5, 2, 'B', fontsize=12, color='b')
        axes.text(5, 3, 'B', fontsize=12, color='b')
        axes.text(6, 1, 'B', fontsize=12, color='b')
        axes.text(6, 2, 'B', fontsize=12, color='b')
        axes.text(6, 3, 'B', fontsize=12, color='b')

        plt.savefig('tow_stars.png', dpi=190)
        plt.clf()
