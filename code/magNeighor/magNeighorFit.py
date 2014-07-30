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

client = kplr.API()
Pixel = 17
Percent = 0
Fake_Po = 2000
Fake_Len = 20

column = []
row = []

#find the neighors in magnitude
def find_mag_neighor(origin_tpf, num, offset=0, ccd=True):
    if ccd:
        stars_over = client.target_pixel_files(ktc_kepler_id="!=%d"%origin_tpf.ktc_kepler_id, kic_kepmag=">=%f"%origin_tpf.kic_kepmag, sci_data_quarter=origin_tpf.sci_data_quarter, sci_channel=origin_tpf.sci_channel, sort=("kic_kepmag", 1),ktc_target_type="LC", max_records=num+offset)
        stars_under = client.target_pixel_files(ktc_kepler_id="!=%d"%origin_tpf.ktc_kepler_id, kic_kepmag="<=%f"%origin_tpf.kic_kepmag, sci_data_quarter=origin_tpf.sci_data_quarter, sci_channel=origin_tpf.sci_channel, sort=("kic_kepmag", -1),ktc_target_type="LC", max_records=num+offset)
    else:
        stars_over = client.target_pixel_files(ktc_kepler_id="!=%d"%origin_tpf.ktc_kepler_id, kic_kepmag=">=%f"%origin_tpf.kic_kepmag, sci_data_quarter=origin_tpf.sci_data_quarter, sci_channel="!=%d"%origin_tpf.sci_channel, sort=("kic_kepmag", 1),ktc_target_type="LC", max_records=num+offset)
        stars_under = client.target_pixel_files(ktc_kepler_id="!=%d"%origin_tpf.ktc_kepler_id, kic_kepmag="<=%f"%origin_tpf.kic_kepmag, sci_data_quarter=origin_tpf.sci_data_quarter, sci_channel="!=%d"%origin_tpf.sci_channel, sort=("kic_kepmag", -1),ktc_target_type="LC", max_records=num+offset)
    
    stars = {}
    
    i=0
    j=0
    offset_list =[]
    while len(stars) <num+offset:
        while stars_over[i].ktc_kepler_id in stars:
            i+=1
        tmp_over = stars_over[i]
        while stars_under[j].ktc_kepler_id in stars:
            j+=1
        tmp_under = stars_under[j]
        if tmp_over.kic_kepmag-origin_tpf.kic_kepmag > origin_tpf.kic_kepmag-tmp_under.kic_kepmag:
            stars[tmp_under.ktc_kepler_id] = tmp_under
            j+=1
            if len(stars)>offset:
                pass
            else:
                offset_list.append(tmp_under.ktc_kepler_id)
        elif tmp_over.kic_kepmag-origin_tpf.kic_kepmag < origin_tpf.kic_kepmag-tmp_under.kic_kepmag:
            stars[tmp_over.ktc_kepler_id] = tmp_over
            i+=1
            if len(stars)>offset:
                pass
            else:
                offset_list.append(tmp_over.ktc_kepler_id)
        elif len(stars) < num+offset-1:
            stars[tmp_under.ktc_kepler_id] = tmp_under
            stars[tmp_over.ktc_kepler_id] = tmp_over
            i+=1
            j+=1
            if len(stars)>offset+1:
                pass
            elif len(stars) == offset+1:
                offset_list.append(tmp_under.ktc_kepler_id)
            else:
                offset_list.append(tmp_over.ktc_kepler_id)
                offset_list.append(tmp_under.ktc_kepler_id)
        else:
            stars[tmp_over.ktc_kepler_id] = tmp_over
            i+=1
            if len(stars)>offset:
                pass
            else:
                offset_list.append(tmp_over.ktc_kepler_id)
    
    for key in offset_list:
        stars.pop(key)
    return stars

#find the pixel mask
def get_pixel_mask(flux, kplr_mask):
    pixel_mask = np.zeros(flux.shape)
    pixel_mask[np.isfinite(flux)] = 1 # okay if finite
    pixel_mask[:, (kplr_mask < 1)] = 0 # unless masked by kplr
    return pixel_mask

#find the epoch mask
def get_epoch_mask(pixel_mask):
    foo = np.sum(np.sum((pixel_mask > 0), axis=2), axis=1)
    epoch_mask = np.zeros_like(foo)
    epoch_mask[(foo > 0)] = 1
    return epoch_mask

#load data from tpf
def load_data(tpf):
    kplr_mask, time, flux, flux_err = [], [], [], []
    with tpf.open() as file:
        hdu_data = file[1].data
        kplr_mask = file[2].data
        meta = file[1].header
        column.append(meta['1CRV4P'])
        row.append(meta['2CRV4P'])
        time = hdu_data["time"]
        flux = hdu_data["flux"]
        flux_err = hdu_data["flux_err"]
    #print flux.shape
    pixel_mask = get_pixel_mask(flux, kplr_mask)
    epoch_mask = get_epoch_mask(pixel_mask)
    #flux[pixel_mask==0] = 0
    flux = flux[:, kplr_mask>0]
    flux_err = flux_err[:, kplr_mask>0]
    shape = flux.shape
    '''
    time = time[epoch_mask>0]
    flux = flux[epoch_mask>0,:]
    '''
    flux = flux.reshape((flux.shape[0], -1))
    flux_err = flux_err.reshape((flux.shape[0], -1))

    #mask = np.array(np.sum(np.isfinite(flux), axis=0), dtype=bool)
    #flux = flux[:, mask]

    #interpolate the bad points
    for i in range(flux.shape[1]):
        interMask = np.isfinite(flux[:,i])
        flux[~interMask,i] = np.interp(time[~interMask], time[interMask], flux[interMask,i])
        flux_err[~interMask,i] = np.inf
    
    return time, flux, pixel_mask, kplr_mask, epoch_mask, flux_err

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

def get_train_mask(total_length, percent=0.1, specify=False, initial=0, length=0):
    if specify:
        train_mask = np.ones(total_length)
        train_mask[initial:initial+length] = 0
    else:
        train_mask = np.ones(total_length)
        length = int(total_length * percent)
        initial = int(total_length * (0.5-percent/2.0))
        train_mask[initial:initial+length] = 0
    return train_mask

def get_fake_data(target_flux, position, length, strength):
    #the sine distortion
    '''
    factor = np.arange(target_flux.shape[0])
    factor = (1+0.004*np.sin(12*np.pi*factor/factor[-1]))
    for i in range(0, target_flux.shape[0]):
        target_flux[i] = target_flux[i] * factor[i]
    '''
    #the fake transit
    target_flux[position:position+length, :] = target_flux[position:position+length, :]*strength
    return target_flux

#construct the predictors matrix
def get_fit_matrix(kic, quarter, neighor_num=1, offset=0, ccd=True, normal=False, fake_po=0, fake_len=0, fake_strength=0, constant=True, auto=False, pixel=Pixel, poly=0):
    origin_star = client.star(kic)
    lc = origin_star.get_light_curves()[5]
    origin_tpf = client.target_pixel_files(ktc_kepler_id=origin_star.kepid, sci_data_quarter=quarter, ktc_target_type="LC")[0]
    neighor = find_mag_neighor(origin_tpf, neighor_num, offset, ccd)

    time, target_flux, target_pixel_mask, target_kplr_mask, epoch_mask, flux_err= load_data(origin_tpf)

    neighor_kid, neighor_fluxes, neighor_pixel_maskes, neighor_kplr_maskes = [], [], [], []

    for key, tpf in neighor.items():
        neighor_kid.append(key)
        tmpResult = load_data(tpf)
        neighor_fluxes.append(tmpResult[1])
        neighor_pixel_maskes.append(tmpResult[2])
        neighor_kplr_maskes.append(tmpResult[3])
        epoch_mask *= tmpResult[4]
    
    #remove bad time point based on simulteanous epoch mask
    time = time[epoch_mask>0]
    target_flux = target_flux[epoch_mask>0]
    flux_err = flux_err[epoch_mask>0]

    time_len = time.shape[0]

    #construct covariance matrix
    covar_list = np.zeros((flux_err.shape[1], flux_err.shape[0], flux_err.shape[0]))
    for i in range(0, flux_err.shape[1]):
        for j in range(0, flux_err.shape[0]):
            covar_list[i, j, j] = flux_err[j][i]
    for i in range(0, len(neighor_fluxes)):
        neighor_fluxes[i] = neighor_fluxes[i][epoch_mask>0, :]

    #insert fake signal into the data
    if fake_len != 0:
        target_flux = get_fake_data(target_flux, fake_po, fake_len, fake_strength)

    #construt the neighor flux matrix
    neighor_flux_matrix = np.float64(np.concatenate(neighor_fluxes, axis=1))
    target_flux = np.float64(target_flux)

    #normalize the data
    target_mean, target_std = None, None
    if normal:
        #neighor_flux_matrix = (neighor_flux_matrix - np.mean(neighor_flux_matrix))/np.var(neighor_flux_matrix)
        
        #mean = np.mean(neighor_flux_matrix, axis=0)
        #std = np.std(neighor_flux_matrix, axis=0)
        mean = 0
        std = np.max(neighor_flux_matrix, axis=0)
        neighor_flux_matrix = (neighor_flux_matrix - mean)/std
        print neighor_flux_matrix
        
        #target_mean = np.mean(target_flux, axis=0)
        #target_std = np.std(target_flux, axis=0)
        target_mean = np.zeros(target_flux.shape[1])
        target_std = np.max(target_flux, axis=0)
        target_flux = (target_flux - target_mean)/target_std

    print neighor_flux_matrix.shape
    if auto:
        epoch_len = epoch_mask.shape[0]
        auto_flux = np.zeros(epoch_len)
        auto_flux[epoch_mask>0] = target_flux[:, pixel]
        offset = 0
        window = 144
        scale = 1
        auto_pixel = np.zeros((epoch_len, 2*window))
        for i in range(offset+window, epoch_len-window-offset):
            auto_pixel[i, 0:window] = auto_flux[i-offset-window:i-offset]*scale
            auto_pixel[i, window:2*window] = auto_flux[i+offset+1:i+offset+window+1]*scale
        for i in range(0, offset+window):
            auto_pixel[i, window:2*window] = auto_flux[i+offset+1:i+offset+window+1]*scale
        for i in range(epoch_len-window-offset, epoch_len):
            auto_pixel[i, 0:window] = auto_flux[i-offset-window:i-offset]*scale
        auto_pixel = auto_pixel[epoch_mask>0, :]
        neighor_flux_matrix = np.concatenate((neighor_flux_matrix, auto_pixel), axis=1)

#add the constant level
    if constant:
        time_mean = np.mean(time)
        time_std = np.std(time)
        nor_time = (time-time_mean)/time_std
        print nor_time
        p = np.polynomial.polynomial.polyvander(nor_time, poly)
        neighor_flux_matrix = np.concatenate((neighor_flux_matrix, p), axis=1)

    print neighor_flux_matrix.shape

    pixel_x = (pixel+1)%target_pixel_mask.shape[2]
    pixel_y = (pixel+1)//target_pixel_mask.shape[2]+1

    return neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std

def get_fit_result(neighor_flux_matrix, target_flux, fit_coe, target_mean=None, target_std=None):
    fit_flux = np.dot(neighor_flux_matrix, fit_coe)
    if target_mean is not None:
        fit_flux = fit_flux*target_std + target_mean
        target_flux = target_flux*target_std +target_mean
    ratio = np.divide(target_flux, fit_flux)
    dev = ratio - 1.0
    rms = np.sqrt(np.mean(dev**2, axis=0))

    return fit_flux, ratio, rms

def plot_threepannel(target_flux, neighor_flux_matrix, time, fit_coe, prefix, title, fake=False, fake_po=0, fake_strength=0, target_mean=None, target_std=None, train_mask=None):
    f, axes = plt.subplots(3, 1)
    #region = Rectangle((time[1000], target_flux[1000, Pixel]-1000), time[1049]-time[1000], 2000, facecolor="grey", alpha=0.5)
    if train_mask is not None:
        axes[0].plot(time[train_mask>0], target_flux[train_mask>0, Pixel], '.b', markersize=1, label='train')
        axes[0].plot(time[train_mask==0], target_flux[train_mask==0, Pixel], '.r', markersize=1, label='test')
        axes[0].legend()
    else:
        axes[0].plot(time, target_flux[:, Pixel], '.b', markersize=1)
        #axes[0].add_patch(region)
    plt.setp( axes[0].get_xticklabels(), visible=False)
    plt.setp( axes[0].get_yticklabels(), visible=False)
    axes[0].set_ylabel("flux of tpf")
    
    fit_flux = np.dot(neighor_flux_matrix, fit_coe)
    if target_mean is not None:
        fit_flux = fit_flux*target_std + target_mean
        target_flux = target_flux*target_std + target_mean
    axes[1].plot(time, fit_flux[:, Pixel], '.b', markersize=1)
    plt.setp( axes[1].get_xticklabels(), visible=False)
    plt.setp( axes[1].get_yticklabels(), visible=False)
    axes[1].set_ylabel("flux of fit")
    
    res = fit_flux - target_flux
    ratio = np.divide(target_flux, fit_flux)
    
    axes[2].plot(time, ratio[:, Pixel], 'b.', markersize=2)
    axes[2].set_ylim(0.999,1.001)
    axes[2].set_xlabel("time[BKJD]")
    axes[2].set_ylabel("ratio of data and fit")
    #sin = 1+0.004*np.sin(12*np.pi*(time-time[0])/(time[-1]-time[0]))
    #axes[2].plot(time, sin, color='red')
    if fake:
        axes[2].axhline(y=fake_strength,xmin=0,xmax=3,c="red",linewidth=0.5,zorder=0)
        signal = np.mean(ratio[fake_po:fake_po+Fake_Len, Pixel])
        axes[2].annotate('fitting signal:%.6f'%signal, xy=(time[fake_po], fake_strength+0.0001), xytext=(time[fake_po]+1, 1.0008),
                         arrowprops=dict(arrowstyle="->",
                                         connectionstyle="arc,angleA=0,armA=30,rad=10"),
                     )
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0, hspace=0)
    plt.suptitle('%s'%title)
    plt.savefig('%s.png'%prefix, dpi=150)
    plt.clf()

def print_coe(fit_coe, neighor_kid, neighor_kplr_maskes, prefix, rms, constant=False):
    #print the fitting coefficient
    f = open('%s.dat'%prefix, 'w')
    loc = 0
    for n in range(0, len(neighor_kid)):
        kplr_mask = neighor_kplr_maskes[n].flatten()
        length = np.sum(kplr_mask>0)
        #coe = result[0][:, Pixel]
        coe = np.zeros_like(kplr_mask, dtype=float)
        #coe = np.zeros_like(neighor_pixel_maskes[n], dtype=float)
        #coe = np.ma.masked_equal(coe,0)
        coe[kplr_mask>0] = fit_coe[loc:loc+length, Pixel]
        loc += length
        coe = coe.reshape((neighor_kplr_maskes[n].shape[0], neighor_kplr_maskes[n].shape[1]))

        f.write('fit coefficient of the pixels of kepler %d\n'%neighor_kid[n])
        f.write('================================================\n')
        for i in range(coe.shape[0]):
            for j in range(coe.shape[1]):
                f.write('%8.5f   '%coe[i,j])
            f.write('\n')
        f.write('================================================\n')
    if constant:
        f.write('constant coefficient\n %8.5f\n'%fit_coe[fit_coe.shape[0]-1, Pixel])
        f.write('================================================\n')
    f.write('RMS Deviation:%f'%rms)
    f.close()

if __name__ == "__main__":
#bias vs number of parameters
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        l2 = 0
        ccd = True
        normal = False
        constant = False
        fake_strength = 1.0005
        
        case_num = 20
        
        num_list = np.arange(case_num)
        bias = np.empty_like(num_list, dtype=float)
        
        for num in range(1, case_num+1):
            neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, Fake_Po, Fake_Len, fake_strength, constant)
            num_list[num-1] = neighor_flux_matrix.shape[1]
            covar = covar_list[Pixel]
            result = lss.leastSquareSolve(neighor_flux_matrix, target_flux, covar, l2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[num-1] = (np.mean(ratio[Fake_Po:Fake_Po+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1)

        plt.plot(num_list, bias,'bs')
        #plt.title('test-train')
        plt.xlabel('Number of parameters')
        plt.ylabel('relative bias from the true singal')
        #plt.ylim(ymax=1)
        #plt.ylim(0.99, 1.01)
        plt.savefig('paraNum-bias_pixel(3,4)New.png')

#bias vs number of parameters in trian-and-test framework
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 20
        l2 = 1e-3
        ccd = True
        normal = False
        constant = True
        loc = 2300
        fake_strength = 1.0005
        
        case_num = 80
        initial = 1

        num_list = np.arange(case_num)
        bias = np.empty_like(num_list, dtype=float)
        
        for num in range(initial, initial+case_num):
            neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, loc, Fake_Len, fake_strength, constant)
            train_mask = get_train_mask(neighor_flux_matrix.shape[0], Percent, True, loc-24, Fake_Len+48)
            covar = covar_list[Pixel]
            covar = np.delete(covar, np.s_[loc-24:loc+Fake_Len+24], 0)
            covar = np.delete(covar, np.s_[loc-24:loc+Fake_Len+24], 1)
            num_list[num-initial] = neighor_flux_matrix.shape[1]
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, l2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[num-initial] = (np.mean(ratio[loc:loc+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1)
        
        po = time[loc]
        plt.plot(num_list, bias,'bs')
        plt.title('L2 Regualrization:%.0e Position: %f\n Train-and-Test:%r'%(l2, po, True))
        plt.xlabel('Number of parameters')
        plt.ylabel('relative bias from the true singal')
        plt.savefig('paraNum-bias_pixel(3,4)_l2%.0e_loc%d_Test.png'%(l2, po), dpi=150)

#Single fitting plot
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 220
        l2 = 1e5
        ccd = True
        normal = False
        constant = True
        fake_strength = 0

        Pixel = 17
        
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, fake_strength, constant, Pixel)
        '''
        train_mask = get_train_mask(neighor_flux_matrix.shape[0], Percent, True, 1000, 50)
        print time[1000]
        covar = covar_list[Pixel]
        covar = np.delete(covar, np.s_[1000:1050], 0)
        covar = np.delete(covar, np.s_[1000:1050], 1)
        result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, l2, False)
        '''
        print neighor_flux_matrix.shape
        
        covar = covar_list[Pixel]
        result = lss.leastSquareSolve(neighor_flux_matrix, target_flux, covar, l2, False)
        fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0], target_mean, target_std)
        title = 'Kepler %d Quarter %d Pixel(%d,%d) L2-Reg %.0e \n Fit Source[Initial:%d Number:%d CCD:%r] RMS Deviation:%f'%(kid, quarter, pixel_x, pixel_y, l2, offset+1, num, ccd, rms[Pixel])
        file_prefix = 'fit(%d,%d)_%d_%d_reg%.0e_nor%r_cons%r_norm_tryTrain'%(pixel_x, pixel_y, offset+1, num, l2, normal, constant)
        plot_threepannel(target_flux, neighor_flux_matrix, time, result[0], file_prefix, title, False, 0, target_mean, target_std)
        print_coe(result[0], neighor_kid, neighor_kplr_maskes, file_prefix, rms[Pixel], constant)

#k-fold validation
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        l2 = 1e5
        ccd = True
        normal = False
        constant = True
        k = 10
        case_num = 10
        
        num_list = np.zeros(case_num)
        rms_list = []
        
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, 1, offset, ccd, normal, 0, 0, 0, constant)
        
        kfold_mask = get_kfold_train_mask(neighor_flux_matrix.shape[0], k, True)
        target_kplr_mask = target_kplr_mask.flatten()
        pixel_list = np.arange(target_kplr_mask.shape[0])
        
        skip_num = 0
        
        #for Pixel in pixel_list:
        Pixel = 17
        if target_kplr_mask[Pixel] == 3:
            skip_num += 1
            if skip_num>0:
                print Pixel
                rms_list = []
                for num in range(195, case_num+195):
                    '''
                    if num >= 49:
                        l2= 1
                    '''
                    neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, tmp_target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant, Pixel)
                    num_list[num-195] = neighor_flux_matrix.shape[1]
                    mean_rms = 0
                    for i in range(0, k):
                        covar_mask = np.ones(covar_list[Pixel].shape)
                        covar_mask[kfold_mask==i, :] = 0
                        covar_mask[:, kfold_mask==i] = 0
                        covar = covar_list[Pixel]
                        covar = covar[covar_mask>0]
                        train_length = np.sqrt(covar.shape[0])
                        covar = covar.reshape(train_length, train_length)
                        result = lss.leastSquareSolve(neighor_flux_matrix[kfold_mask!=i, :], target_flux[kfold_mask!=i, Pixel], covar, l2, False)
                        fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix[kfold_mask==i, :], target_flux[kfold_mask==i, Pixel], result[0])
                        mean_rms += rms
                    mean_rms /= k
                    rms_list.append(mean_rms)
                    print 'done %d'%num

                plt.plot(num_list, rms_list, 'bs')
                plt.ylim(ymin=0)
                plt.title('k-fold validation k=%d L2 Reg: %.0e \n KID:%d Pixel(%d,%d)'%(k, l2, kid, pixel_x, pixel_y))
                plt.xlabel('Number of parameters')
                plt.ylabel('RMS Deviation')
                plt.savefig('rms-num_k%d_reg%.0e_kic%d_(%d,%d)_t.png'%(k, l2, kid, pixel_x, pixel_y), dpi=150)
                plt.clf()

#k-fold validation for l2
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 200
        l2 = 0
        ccd = True
        normal = False
        constant = True
        k = 10
        case_num = 5
        
        l2_list = np.zeros(case_num)
        rms_list = []
        
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant)
        kfold_mask = get_kfold_train_mask(neighor_flux_matrix.shape[0], k, True)
        target_kplr_mask = target_kplr_mask.flatten()
        pixel_list = np.arange(target_kplr_mask.shape[0])
        
        #for Pixel in pixel_list:
        Pixel = 17
        if target_kplr_mask[Pixel] == 3:
            print Pixel
            rms_list = []
            neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, tmp_target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant, Pixel)
            for case in range(3, case_num+3):
                l2 = (10**case)
                l2_list[case-3] = case
                mean_rms = 0
                for i in range(0, k):
                    covar_mask = np.ones(covar_list[Pixel].shape)
                    covar_mask[kfold_mask==i, :] = 0
                    covar_mask[:, kfold_mask==i] = 0
                    covar = covar_list[Pixel]
                    covar = covar[covar_mask>0]
                    train_length = np.sqrt(covar.shape[0])
                    print train_length
                    covar = covar.reshape(train_length, train_length)
                    result = lss.leastSquareSolve(neighor_flux_matrix[kfold_mask!=i, :], target_flux[kfold_mask!=i, Pixel], covar, l2, False)
                    fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix[kfold_mask==i, :], target_flux[kfold_mask==i, Pixel], result[0])
                    mean_rms += rms
                mean_rms /= k
                rms_list.append(mean_rms)
                print 'done %d'%case
            
            plt.plot(l2_list, rms_list, 'bs')
            plt.ylim(ymin=0)
            plt.title('k-fold validation k=%d Number of stars: %d \n KID:%d Pixel(%d,%d)'%(k, num, kid, pixel_x, pixel_y))
            plt.xlabel(r'Strength of Regularization($log \lambda$)')
            plt.ylabel('RMS Deviation')
            plt.savefig('rms-l2_k%d_num%d_kic%d_(%d,%d).png'%(k, num, kid, pixel_x, pixel_y), dpi=150)

#k-fold validation for l2 and num
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 30
        l2 = 0
        ccd = True
        normal = False
        constant = True
        k = 10
        l2_case_num = 5
        num_case_num = 40
        
        l2_list = np.zeros(l2_case_num)
        rms_list = np.zeros((l2_case_num, num_case_num))
        
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, 1, offset, ccd, normal, 0, 0, 0, constant)
        kfold_mask = get_kfold_train_mask(neighor_flux_matrix.shape[0], k, True)
        target_kplr_mask = target_kplr_mask.flatten()
        pixel_list = np.arange(target_kplr_mask.shape[0])
        
        #for Pixel in pixel_list:
        Pixel = 16
        if target_kplr_mask[Pixel] == 3:
            f = open('k-fold_l2_num_%d_ext2.dat'%Pixel, 'w')
            print Pixel
            for num in range(61, num_case_num+61):
                if num >= 49:
                    l2= 1
                neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, tmp_target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant, Pixel)
                for case in range(3, l2_case_num+3):
                    l2 = 10**(case)
                    mean_rms = 0
                    for i in range(0, k):
                        covar_mask = np.ones(covar_list[Pixel].shape)
                        covar_mask[kfold_mask==i, :] = 0
                        covar_mask[:, kfold_mask==i] = 0
                        covar = covar_list[Pixel]
                        covar = covar[covar_mask>0]
                        train_length = np.sqrt(covar.shape[0])
                        covar = covar.reshape(train_length, train_length)
                        result = lss.leastSquareSolve(neighor_flux_matrix[kfold_mask!=i, :], target_flux[kfold_mask!=i, Pixel], covar, l2, False)
                        fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix[kfold_mask==i, :], target_flux[kfold_mask==i, Pixel], result[0])
                        mean_rms += rms
                    mean_rms /= k
                    rms_list[case-3, num-61] = mean_rms
                    f.write('%f\t'%mean_rms)
                    print 'l2 done %d'%case
                f.write('\n')
                print 'num done %d'%num

            f.close()
            np.save('k-fold_l2_num_%d_ext2.npy'%Pixel, rms_list)
            plt.imshow(rms_list, cmap='Greys', aspect='auto', extent=[61,num_case_num+60,l2_case_num+1,3])
            bar = plt.colorbar()
            bar.set_label('RMS Deviation')
            plt.title('k-fold validation k=%d \n KID:%d Pixel(%d,%d)'%(k, kid, pixel_x, pixel_y))
            plt.ylabel(r'Strength of Regularization($log \lambda$)')
            plt.xlabel('Number of stars')
            plt.savefig('rms-l2_num_k%d_kic%d_%d(%d,%d).png'%(k, kid, Pixel, pixel_x, pixel_y), dpi=150)
            plt.clf()

#k-fold validation for l2 and num optimize
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 1
        l2 = 10
        ccd = True
        normal = False
        constant = True
        k = 10
        
        num_last = 0
        
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant)
        kfold_mask = get_kfold_train_mask(neighor_flux_matrix.shape[0], k, True)
        target_kplr_mask = target_kplr_mask.flatten()
        pixel_list = np.arange(target_kplr_mask.shape[0])

        mean_rms = 0
        for i in range(0, k):
            covar_mask = np.ones(covar_list[Pixel].shape)
            covar_mask[kfold_mask==i, :] = 0
            covar_mask[:, kfold_mask==i] = 0
            covar = covar_list[Pixel]
            covar = covar[covar_mask>0]
            train_length = np.sqrt(covar.shape[0])
            covar = covar.reshape(train_length, train_length)
            result = lss.leastSquareSolve(neighor_flux_matrix[kfold_mask!=i, :], target_flux[kfold_mask!=i, Pixel], covar, l2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix[kfold_mask==i, :], target_flux[kfold_mask==i, Pixel], result[0])
            mean_rms += rms
        mean_rms /= k
        
        
        #for Pixel in pixel_list:
        Pixel = 17
        if target_kplr_mask[Pixel] == 3:
            f = open('k-fold_l2_num_%d_optimize.dat'%Pixel, 'w')
            print Pixel
            while num<500:
                neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant)
                mean_rms_l2 = 0
                for i in range(0, k):
                    covar_mask = np.ones(covar_list[Pixel].shape)
                    covar_mask[kfold_mask==i, :] = 0
                    covar_mask[:, kfold_mask==i] = 0
                    covar = covar_list[Pixel]
                    covar = covar[covar_mask>0]
                    train_length = np.sqrt(covar.shape[0])
                    covar = covar.reshape(train_length, train_length)
                    result = lss.leastSquareSolve(neighor_flux_matrix[kfold_mask!=i, :], target_flux[kfold_mask!=i, Pixel], covar, l2*10, False)
                    fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix[kfold_mask==i, :], target_flux[kfold_mask==i, Pixel], result[0])
                    mean_rms_l2 += rms
                mean_rms_l2 /= k
                
                neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, tmp_target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num+1, offset, ccd, normal, 0, 0, 0, constant, Pixel)
                mean_rms_num = 0
                for i in range(0, k):
                    covar_mask = np.ones(covar_list[Pixel].shape)
                    covar_mask[kfold_mask==i, :] = 0
                    covar_mask[:, kfold_mask==i] = 0
                    covar = covar_list[Pixel]
                    covar = covar[covar_mask>0]
                    train_length = np.sqrt(covar.shape[0])
                    covar = covar.reshape(train_length, train_length)
                    result = lss.leastSquareSolve(neighor_flux_matrix[kfold_mask!=i, :], target_flux[kfold_mask!=i, Pixel], covar, l2, False)
                    fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix[kfold_mask==i, :], target_flux[kfold_mask==i, Pixel], result[0])
                    mean_rms_num += rms
                mean_rms_num /= k
                
                if (mean_rms<mean_rms_l2) and (mean_rms<mean_rms_num):
                    break
                elif (mean_rms-mean_rms_l2) < (mean_rms-mean_rms_num):
                    num += 1
                    mean_rms = mean_rms_num
                else:
                    l2 *= 10
                    mean_rms = mean_rms_l2
            
                f.write('%d\t%d\t%.8f\n'%(num, l2, mean_rms))
                print '%d\t%d\t%.8f\n'%(num, l2, mean_rms)
            
            f.close()
                
                
#k-fold validation for l2 and num optimize method2
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 1
        l2 = 10
        ccd = True
        normal = False
        constant = True
        auto = False
        k = 10
        
        num_last = 0
        
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant)
        kfold_mask = get_kfold_train_mask(neighor_flux_matrix.shape[0], k, True)
        target_kplr_mask = target_kplr_mask.flatten()
        pixel_list = np.arange(target_kplr_mask.shape[0])
        
        print np.mean(neighor_flux_matrix, axis=0)
        mean_rms = 10
        
        for Pixel in pixel_list:
        #Pixel = 17
            if target_kplr_mask[Pixel] == 3 and Pixel==17:
                f = open('./k_fold_optimization/kic%d/nor/pixel%d_optimize.log'%(kid, Pixel), 'w')
                print Pixel
                num = 70
                l2 = 1e-5
                mean_rms = 10
                while num<700:
                    neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, tmp_target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num+1, offset, ccd, normal, 0, 0, 0, constant, auto, Pixel)
                    mean_rms_num = 0
                    for i in range(0, k):
                        covar_mask = np.ones(covar_list[Pixel].shape)
                        covar_mask[kfold_mask==i, :] = 0
                        covar_mask[:, kfold_mask==i] = 0
                        covar = covar_list[Pixel]
                        covar = covar[covar_mask>0]
                        train_length = np.sqrt(covar.shape[0])
                        covar = covar.reshape(train_length, train_length)
                        result = lss.leastSquareSolve(neighor_flux_matrix[kfold_mask!=i, :], target_flux[kfold_mask!=i, Pixel], covar, l2, False)
                        fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix[kfold_mask==i, :], target_flux[kfold_mask==i, Pixel], result[0], target_mean[Pixel], target_std[Pixel])
                        mean_rms_num += rms
                    mean_rms_num /= k
                    if mean_rms_num<mean_rms:
                        num += 1
                        mean_rms = mean_rms_num
                        f.write('%d\t%e\t%.8f\n'%(num, l2, mean_rms))
                        print '%d\t%e\t%.8f\n'%(num, l2, mean_rms)
                    else:
                        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, tmp_target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant, auto, Pixel)
                        mean_rms_l2 = 0
                        for i in range(0, k):
                            covar_mask = np.ones(covar_list[Pixel].shape)
                            covar_mask[kfold_mask==i, :] = 0
                            covar_mask[:, kfold_mask==i] = 0
                            covar = covar_list[Pixel]
                            covar = covar[covar_mask>0]
                            train_length = np.sqrt(covar.shape[0])
                            covar = covar.reshape(train_length, train_length)
                            result = lss.leastSquareSolve(neighor_flux_matrix[kfold_mask!=i, :], target_flux[kfold_mask!=i, Pixel], covar, l2*10, False)
                            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix[kfold_mask==i, :], target_flux[kfold_mask==i, Pixel], result[0], target_mean[Pixel], target_std[Pixel])
                            mean_rms_l2 += rms
                        mean_rms_l2 /= k

                        if mean_rms_l2<mean_rms:
                            l2 *= 10
                            mean_rms = mean_rms_l2
                            f.write('%d\t%e\t%.8f\n'%(num, l2, mean_rms))
                            print '%d\t%e\t%.8f\n'%(num, l2, mean_rms)
                        else:
                            ind = True
                            for plus in range(2, 8):
                                neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, tmp_target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num+plus, offset, ccd, normal, 0, 0, 0, constant, auto, Pixel)
                                mean_rms_num = 0
                                for i in range(0, k):
                                    covar_mask = np.ones(covar_list[Pixel].shape)
                                    covar_mask[kfold_mask==i, :] = 0
                                    covar_mask[:, kfold_mask==i] = 0
                                    covar = covar_list[Pixel]
                                    covar = covar[covar_mask>0]
                                    train_length = np.sqrt(covar.shape[0])
                                    covar = covar.reshape(train_length, train_length)
                                    result = lss.leastSquareSolve(neighor_flux_matrix[kfold_mask!=i, :], target_flux[kfold_mask!=i, Pixel], covar, l2, False)
                                    fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix[kfold_mask==i, :], target_flux[kfold_mask==i, Pixel], result[0], target_mean[Pixel], target_std[Pixel])
                                    mean_rms_num += rms
                                mean_rms_num /= k
                                if mean_rms_num<mean_rms:
                                    num += plus
                                    mean_rms = mean_rms_num
                                    ind = False
                                    f.write('%d\t%e\t%.8f\n'%(num, l2, mean_rms))
                                    print '%d\t%e\t%.8f\n'%(num, l2, mean_rms)
                                    break
                            if ind:
                                break
            
                f.close()


#signal bias vs l2 regularization
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 30
        l2 = 0
        ccd = True
        normal = False
        constant = True
        loc = 2300
        fake_strength = 1.0005
        
        train = True
        
        strength = np.arange(15)
        bias = np.empty_like(strength, dtype=float)
        
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, loc, Fake_Len, fake_strength, constant)
        train_mask = []
        covar = covar_list[Pixel]
        if train:
            train_mask = get_train_mask(neighor_flux_matrix.shape[0], Percent, True, loc-24, Fake_Len+48)
            covar = np.delete(covar, np.s_[loc-24:loc+Fake_Len+24], 0)
            covar = np.delete(covar, np.s_[loc-24:loc+Fake_Len+24], 1)
        else:
            train_mask = np.ones(neighor_flux_matrix.shape[0])
        max = -100
        for i in strength:
            l2 = (1e-3)*(10**strength[i])
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, l2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0], target_mean, target_std)
            bias[i] = (np.mean(ratio[loc:loc+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1)
            strength[i] += -3
            print (strength[i], bias[i])

        po = time[loc]
        plt.clf()
        plt.plot(strength, bias, 'bs')
        plt.title('Number of stars:%d location:%f \n Sigal Strength:%.4f Train-and-Test:%r'%(num, po, fake_strength, train))
        plt.xlabel(r'Strength of Regularization($log \lambda$)')
        plt.ylabel('relative bias from the true singal')
        plt.savefig('l2-bias_num%d_pixel(3,4)Nor%r_train%r_loc%d_str%.4f.png'%(num, normal, train, po, fake_strength), dpi=150)

#single plot with signal
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 20
        l2 = 0
        ccd = True
        normal = False
        constant = True
        loc = 2300
        fake_strength = 1.0005
        train = True
        window = 24

        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, loc, Fake_Len, fake_strength, constant)
        result = []
        covar = covar_list[Pixel]
        train_mask = None
        if train:
            train_mask = get_train_mask(neighor_flux_matrix.shape[0], Percent, True, loc-window, Fake_Len+2*window)
            covar = np.delete(covar, np.s_[loc-window:loc+Fake_Len+window], 0)
            covar = np.delete(covar, np.s_[loc-window:loc+Fake_Len+window], 1)
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, l2, False)
        else:
            result = lss.leastSquareSolve(neighor_flux_matrix, target_flux, covar, l2, False)
        fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0], target_mean, target_std)
        print np.mean(ratio[loc:loc+Fake_Len, Pixel])
        print((np.mean(ratio[loc:loc+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1))
        po = time[loc]
        title = 'Kepler %d Quarter %d Pixel(%d,%d) L2-Reg %.0e \n Fit Source[Initial:%d Number:%d CCD:%r] RMS Deviation:%f'%(kid, quarter, pixel_x, pixel_y, l2, offset+1, num, ccd, rms[Pixel])
        file_prefix = 'fit(%d,%d)_%d_%d_reg%.0e_nor%r_train%r_loc%d_signal%.4f'%(pixel_x, pixel_y, offset+1, num, l2, normal, train, po, fake_strength)
        plot_threepannel(target_flux, neighor_flux_matrix, time, result[0], file_prefix, title, True, loc, fake_strength, target_mean, target_std, train_mask)
#print_coe(result[0], neighor_kid, neighor_kplr_maskes, file_prefix, result[2][Pixel], constant)

#signal with different location
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 20
        l2 = 0
        ccd = True
        normal = False
        constant = True
        fake_strength = 1.0005
        
        case_num = 22
        
        position_list = np.arange(case_num)
        bias = np.empty_like(position_list, dtype=float)
        
        for i in range(0, case_num):
            position = Fake_Len*2+i*200
            neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, position, Fake_Len, fake_strength, constant)
            position_list[i] = time[position]
            covar = covar_list[Pixel]
            result = lss.leastSquareSolve(neighor_flux_matrix, target_flux, None, l2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[i] = (np.mean(ratio[position:position+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1)
        
        plt.clf()
        plt.plot(position_list, bias, 'bs')
        plt.title('Number of stars:%d'%num)
        plt.xlabel('location of the signal(time[BKJD])')
        plt.ylabel('relative bias from the true singal')
        plt.savefig('loc-bias_num%d_pixel(3,4).png'%num)

#signal with different location and train-and-test framework
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 35
        l2 = 0
        ccd = True
        normal = False
        constant = True
        fake_strength = 1.0005
        case_num = 22
        
        position_list = np.arange(case_num)
        bias = np.empty_like(position_list, dtype=float)
        
        for i in range(0, case_num):
            position = Fake_Len*2+i*200
            neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, position, Fake_Len, fake_strength, constant)
            position_list[i] = time[position]
            train_mask = get_train_mask(neighor_flux_matrix.shape[0], Percent, True, position-24, Fake_Len+48)
            covar = covar_list[Pixel]
            covar = np.delete(covar, np.s_[position-24:position+Fake_Len+24], 0)
            covar = np.delete(covar, np.s_[position-24:position+Fake_Len+24], 1)
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, l2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[i] = (np.mean(ratio[position:position+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1)
        
        plt.clf()
        plt.plot(position_list, bias, 'bs')
        plt.ylim(-1., 1.)
        plt.title('Number of stars:%d L2 Regularization:%.0e\n Signal Strength:%.4f Trian-and-Test:%r'%(num, l2, fake_strength, True))
        plt.xlabel('location of the signal(time[BKJD])')
        plt.ylabel('relative bias from the true singal')
        plt.savefig('loc-bias_num%d_pixel(3,4)Test_cons%r_reg%.0e_str%.4f.png'%(num, constant, l2, fake_strength), dpi=150)

#signal with different location and l2 reg in train-and-test framework
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 20
        l2 = 0
        ccd = True
        normal = False
        constant = True
        fake_strength = 1.0005
        case_num = 22
        
        plot_range = (-1., 1.)
        
        position_list = np.arange(case_num)
        bias = np.empty((case_num, 3), dtype=float)
        
        Pixel = 17
        pixel_x = 0
        pixel_y = 0
        
        num = 86
        for i in range(0, case_num):
            position = Fake_Len*2+i*200
            neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, position, Fake_Len, fake_strength, constant, Pixel)
            position_list[i] = time[position]
            train_mask = get_train_mask(neighor_flux_matrix.shape[0], Percent, True, position-24, Fake_Len+48)
            covar = covar_list[Pixel]
            covar = np.delete(covar, np.s_[position-24:position+Fake_Len+24], 0)
            covar = np.delete(covar, np.s_[position-24:position+Fake_Len+24], 1)
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, 1e2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[i, 0] = (np.mean(ratio[position:position+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1)
            for j in range(1, 3):
                result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, (1e2)*(100**j), False)
                fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
                bias[i, j] = (np.mean(ratio[position:position+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1)
        
        f, axes = plt.subplots(3, 3)
        axes[0, 0].plot(position_list, bias[:, 0], 'bs')
        axes[0, 0].set_title('l2-Reg:1e2')
        axes[0, 0].set_ylabel('Num of Stars: %d'%num)
        axes[0, 0].set_ylim(plot_range)
        plt.setp(axes[0, 0].get_xticklabels(), visible=False)
        axes[0, 1].plot(position_list, bias[:, 1], 'bs')
        axes[0, 1].set_title('l2-Reg:1e4')
        axes[0, 1].set_ylim(plot_range)
        plt.setp(axes[0, 1].get_xticklabels(), visible=False)
        plt.setp(axes[0, 1].get_yticklabels(), visible=False)
        axes[0, 2].plot(position_list, bias[:, 2], 'bs')
        axes[0, 2].set_title('l2-Reg:1e6')
        axes[0, 2].set_ylim(plot_range)
        plt.setp(axes[0, 2].get_xticklabels(), visible=False)
        plt.setp(axes[0, 2].get_yticklabels(), visible=False)
        '''
        axes[0, 3].plot(position_list, bias[:, 3], 'bs')
        axes[0, 3].set_title('l2-Reg:1e7')
        axes[0, 3].set_ylim(plot_range)
        plt.setp(axes[0, 3].get_xticklabels(), visible=False)
        plt.setp(axes[0, 3].get_yticklabels(), visible=False)
        '''
        print('done1')

        num = 56
        for i in range(0, case_num):
            position = Fake_Len*2+i*200
            neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, position, Fake_Len, fake_strength, constant, Pixel)
            position_list[i] = time[position]
            train_mask = get_train_mask(neighor_flux_matrix.shape[0], Percent, True, position-24, Fake_Len+48)
            covar = covar_list[Pixel]
            covar = np.delete(covar, np.s_[position-24:position+Fake_Len+24], 0)
            covar = np.delete(covar, np.s_[position-24:position+Fake_Len+24], 1)
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, 1e2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[i, 0] = (np.mean(ratio[position:position+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1)
            for j in range(1, 3):
                result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, (1e2)*(100**j), False)
                fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
                bias[i, j] = (np.mean(ratio[position:position+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1)

        axes[1, 0].plot(position_list, bias[:, 0], 'bs')
        axes[1, 0].set_ylabel('Num of Stars: %d'%num)
        axes[1, 0].set_ylim(plot_range)
        plt.setp(axes[1, 0].get_xticklabels(), visible=False)
        axes[1, 1].plot(position_list, bias[:, 1], 'bs')
        axes[1, 1].set_ylim(plot_range)
        plt.setp(axes[1, 1].get_xticklabels(), visible=False)
        plt.setp(axes[1, 1].get_yticklabels(), visible=False)
        axes[1, 2].plot(position_list, bias[:, 2], 'bs')
        axes[1, 2].set_ylim(plot_range)
        plt.setp(axes[1, 2].get_xticklabels(), visible=False)
        plt.setp(axes[1, 2].get_yticklabels(), visible=False)
        '''
        axes[1, 3].plot(position_list, bias[:, 3], 'bs')
        axes[1, 3].set_ylim(plot_range)
        plt.setp(axes[1, 3].get_xticklabels(), visible=False)
        plt.setp(axes[1, 3].get_yticklabels(), visible=False)
        '''
        print('done2')

        num = 26
        for i in range(0, case_num):
            position = Fake_Len*2+i*200
            neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, position, Fake_Len, fake_strength, constant, Pixel)
            position_list[i] = time[position]
            train_mask = get_train_mask(neighor_flux_matrix.shape[0], Percent, True, position-24, Fake_Len+48)
            covar = covar_list[Pixel]
            covar = np.delete(covar, np.s_[position-24:position+Fake_Len+24], 0)
            covar = np.delete(covar, np.s_[position-24:position+Fake_Len+24], 1)
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, 1e2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[i, 0] = (np.mean(ratio[position:position+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1)
            for j in range(1, 3):
                result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, (1e2)*(100**j), False)
                fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
                bias[i, j] = (np.mean(ratio[position:position+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1)

        axes[2, 0].plot(position_list, bias[:, 0], 'bs')
        axes[2, 0].set_ylabel('Num of Stars: %d'%num)
        axes[2, 0].set_ylim(plot_range)
        plt.setp(axes[2, 0].get_xticklabels(), visible=False)
        axes[2, 1].plot(position_list, bias[:, 1], 'bs')
        axes[2, 1].set_ylim(plot_range)
        plt.setp(axes[2, 1].get_xticklabels(), visible=False)
        plt.setp(axes[2, 1].get_yticklabels(), visible=False)
        axes[2, 2].plot(position_list, bias[:, 2], 'bs')
        axes[2, 2].set_ylim(plot_range)
        plt.setp(axes[2, 2].get_xticklabels(), visible=False)
        plt.setp(axes[2, 2].get_yticklabels(), visible=False)
        '''
        axes[2, 3].plot(position_list, bias[:, 3], 'bs')
        axes[2, 3].set_ylim(plot_range)
        plt.setp(axes[2, 3].get_xticklabels(), visible=False)
        plt.setp(axes[2, 3].get_yticklabels(), visible=False)
        '''
        print('done3')

        '''
        num = 5
        for i in range(0, case_num):
            position = Fake_Len*2+i*200
            neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, position, Fake_Len, fake_strength, constant)
            position_list[i] = time[position]
            train_mask = get_train_mask(neighor_flux_matrix.shape[0], Percent, True, position-24, Fake_Len+48)
            covar = covar_list[Pixel]
            covar = np.delete(covar, np.s_[position-24:position+Fake_Len+24], 0)
            covar = np.delete(covar, np.s_[position-24:position+Fake_Len+24], 1)
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, 0, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[i, 0] = (np.mean(ratio[position:position+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1)
            for j in range(1, 4):
                result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, (1e1)*(100**j), False)
                fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
                bias[i, j] = (np.mean(ratio[position:position+Fake_Len, Pixel]) - fake_strength)/(fake_strength-1)

        axes[3, 0].plot(position_list, bias[:, 0], 'bs')
        axes[3, 0].set_ylabel('Num of Stars: %d'%num)
        axes[3, 0].set_ylim(plot_range)
        plt.setp(axes[3, 0].get_xticklabels(), visible=False)
        axes[3, 1].plot(position_list, bias[:, 1], 'bs')
        axes[3, 1].set_ylim(plot_range)
        plt.setp(axes[3, 1].get_xticklabels(), visible=False)
        plt.setp(axes[3, 1].get_yticklabels(), visible=False)
        axes[3, 2].plot(position_list, bias[:, 2], 'bs')
        axes[3, 2].set_ylim(plot_range)
        plt.setp(axes[3, 2].get_xticklabels(), visible=False)
        plt.setp(axes[3, 2].get_yticklabels(), visible=False)
        axes[3, 3].plot(position_list, bias[:, 3], 'bs')
        axes[3, 3].set_ylim(plot_range)
        plt.setp(axes[3, 3].get_xticklabels(), visible=False)
        plt.setp(axes[3, 3].get_yticklabels(), visible=False)
        '''

        fig = plt.gcf()
        fig.set_size_inches(18.5,20.5)
        fig.text(0.5, 0.03, 'Location of the signal(time[BKJD])', ha='center', va='center')
        fig.text(0.02, 0.5, 'Relative bias from the true signal\n [(Mearsured-True)/True]', ha='center', va='center', rotation='vertical')

        plt.subplots_adjust(left=0.1, bottom=0.07, right=0.97, top=0.95,
                    wspace=0, hspace=0.12)

        plt.suptitle('Signal Strength:%.4f Train-and-Test: %r'%(fake_strength, True))


        plt.savefig('loc-bias_num_KIC%d_pixel%d(%d,%d)Test_cons%r_diff_str%.4f_range(-1,1)_3.png'%(kid, Pixel, pixel_x, pixel_y, constant, fake_strength))
        #plt.show()

#bias from 1 with different location and train-and-test framework
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 5
        l2 = 0
        ccd = True
        normal = False
        fake_strength = 0
        constant = True
        
        case_num = 22
        
        position_list = np.arange(case_num)
        bias = np.empty_like(position_list, dtype=float)
        
        for i in range(0, case_num):
            position = Fake_Len*2+i*200
            neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, fake_strength, constant)
            position_list[i] = time[position]
            train_mask = get_train_mask(neighor_flux_matrix.shape[0], Percent, True, position-24, Fake_Len+48)
            covar = covar_list[Pixel]
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], None, l2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[i] = np.mean(ratio[position:position+Fake_Len, Pixel]) - 1
        
        plt.clf()
        plt.plot(position_list, bias, 'bs')
        plt.title('Number of stars:%d \n trian-and-test with no signal'%num)
        plt.xlabel('location of the signal(time[BKJD])')
        plt.ylabel('bias from 1')
        plt.savefig('loc-bias_num%d_pixel(3,4)noSignalTest.png'%num)

#Single fitting plot with normalization
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 5
        l2 = 0
        ccd = True
        normal = False
        constant = True
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant)
        covar = covar_list[Pixel]
        covar = covar/(target_std[Pixel]**2)
        result = lss.leastSquareSolve(neighor_flux_matrix, target_flux, covar, l2, False)
        fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0], target_mean, target_std)
        title = 'Kepler %d Quarter %d Pixel(%d,%d) L2-Reg %.0e \n Fit Source[Initial:%d Number:%d CCD:%r] RMS Deviation:%f'%(kid, quarter, pixel_x, pixel_y, l2, offset+1, num, ccd, rms[Pixel])
        file_prefix = 'fit(%d,%d)_%d_%d_reg%.0e_nor%r_cons%r.png'%(pixel_x, pixel_y, offset+1, num, l2, normal, constant)
        plot_threepannel(target_flux, neighor_flux_matrix, time, result[0], file_prefix, title, False, 0, 0, target_mean, target_std)
        print_coe(result[0], neighor_kid, neighor_kplr_maskes, file_prefix, result[2][Pixel], constant)

#Generate light curve
    if False:
        #planet = client.planet("20b")
        #kid = planet.kepid
        kid = 5088536
        quarter = 5
        offset = 0
        num = 20
        l2 = 0
        ccd = True
        normal = False
        constant = True
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant)
        
        target_kplr_mask = target_kplr_mask.flatten()
        covar_list = covar_list[target_kplr_mask == 3]
        target_flux = target_flux[:, target_kplr_mask == 3]
        print target_flux.shape
        result = np.zeros((neighor_flux_matrix.shape[1], target_flux.shape[1]))
        for i in range(0, target_flux.shape[1]):
            covar = covar_list[i]
            result[:, i] = lss.leastSquareSolve(neighor_flux_matrix, target_flux[:, i], covar, l2, False)[0]
        fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result)
        print fit_flux.shape
        light_curve = np.sum(target_flux, axis=1)
        fit_light_curve = np.sum(fit_flux, axis=1)
        light_curve_ratio = np.divide(light_curve, fit_light_curve)
        print light_curve_ratio

        dev = light_curve_ratio - 1.
        rms = np.sqrt(np.mean(dev**2, axis=0))

        f, axes = plt.subplots(4, 1)
        axes[0].plot(time, light_curve, '.b', markersize=1)
        plt.setp( axes[0].get_xticklabels(), visible=False)
        plt.setp( axes[0].get_yticklabels(), visible=False)
        axes[0].set_ylabel("light curve")

        axes[1].plot(time, fit_light_curve, '.b', markersize=1)
        plt.setp( axes[1].get_xticklabels(), visible=False)
        plt.setp( axes[1].get_yticklabels(), visible=False)
        axes[1].set_ylabel("fit")
        
        axes[2].plot(time, light_curve_ratio, '.b', markersize=2)
        plt.setp( axes[2].get_xticklabels(), visible=False)
        #plt.setp( axes[2].get_yticklabels(), visible=False)
        axes[2].set_ylim(0.999,1.001)
        #axes[2].set_xlabel("time")
        axes[2].set_ylabel("ratio")

        #plot the PDC curve
        star = client.star(kid)
        lc = star.get_light_curves(short_cadence=False)[quarter]
        data = lc.read()
        flux = data["PDCSAP_FLUX"]
        inds = np.isfinite(flux)
        flux = flux[inds]
        pdc_time = data["TIME"][inds]

        axes[3].plot(pdc_time, flux, '.b', markersize=2)
        plt.setp( axes[3].get_yticklabels(), visible=False)
        axes[3].set_ylabel("pdc flux")
        axes[3].set_xlabel("time [BKJD]")

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)
        plt.suptitle('Kepler %d Quarter %d L2-Reg %.0e \n Fit Source[Initial:%d Number:%d CCD:%r] RMS Deviation:%f'%(kid, quarter,  l2, offset+1, num, ccd, rms))
        plt.savefig('lightCurve_%d_%d_%d_q%d_reg%.0e_nor%r_pdc_point.png'%(kid, offset+1, num, quarter, l2, normal), dpi=190)
        plt.show()
        plt.clf()

#Generate light curve train-and-test
    if False:
        t0 = tm.time()
        kid = 5088536
        quarter = 5
        offset = 0
        num = 136
        l2 = 1e5
        ccd = True
        normal = False
        constant = True

        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant)
        
        print len(neighor_kid)
        #np.save('./kic%d/coe/kic%d_num%d-%d_neigbourkid.npy'%(kid, kid, offset+1, num), neighor_kid)
        #np.save('./kic%d/coe/kic%d_num%d-%d_neigbourmask.npy'%(kid, kid, offset+1, num), neighor_kplr_maskes)

        target_kplr_mask = target_kplr_mask.flatten()
        covar_list = covar_list[target_kplr_mask == 3]
        target_flux = target_flux[:, target_kplr_mask == 3]

        length = target_flux.shape[0]
        group_num = 1
        margin = 24
        
        group_mask = np.ones(length, dtype=int)
        loc = 0
        group = 0
        while length-loc >= group_num:
            group_mask[loc:loc+group_num] = group
            loc += group_num
            group += 1
        group_mask[loc:] = group

        #fit_flux = np.empty_like(target_flux)
        #fit_coe = []
        fit_flux = np.load('./kic%d/coe/kic%d_num%d-%d_reg%.0e_whole.npy'%(kid, kid, offset+1, num, l2))
        fit_coe = np.load('./kic%d/coe/kic%d_num%d-%d_rge%.0e_whole_coe.npy'%(kid, kid, offset+1, num, l2))
        print fit_coe.shape

        for i in range(1078, group):
            train_mask = np.ones(length)
            if (margin <= i*group_num) and (length-(i+1)*group_num >= margin):
                train_mask[i*group_num-margin:(i+1)*group_num+margin] = 0
            elif margin > i*group_num:
                train_mask[:(i+1)*group_num+margin] = 0
            else:
                train_mask[i*group_num-margin:] = 0

            covar_mask = np.ones((length, length))
            covar_mask[train_mask==0, :] = 0
            covar_mask[:, train_mask==0] = 0
            
            covar = np.mean(covar_list, axis=0)
            covar = covar[covar_mask>0]
            train_length = np.sum(train_mask, axis=0)
            covar = covar.reshape(train_length, train_length)
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, l2, False)[0]
            #fit_coe.append(result)
            fit_flux[i*group_num:(i+1)*group_num] = np.dot(neighor_flux_matrix[group_mask == i], result)
            fit_coe = np.concatenate((fit_coe, np.array([result])), axis=0)
            '''
            for pixel in range(0, target_flux.shape[1]):
                covar = covar_list[pixel]
                covar = covar[covar_mask>0]
                train_length = np.sum(train_mask, axis=0)
                covar = covar.reshape(train_length, train_length)
                result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0, pixel], covar, l2, False)[0]
                fit_flux[i*group_num:(i+1)*group_num, pixel] = np.dot(neighor_flux_matrix[group_mask == i], result)
            '''
            np.save('./kic%d/coe/kic%d_num%d-%d_reg%.0e_whole.npy'%(kid, kid, offset+1, num, l2), fit_flux)
            print('done%d'%i)
        np.save('./kic%d/coe/kic%d_num%d-%d_rge%.0e_whole_coe.npy'%(kid, kid, offset+1, num, l2), fit_coe)


        if length > group*group_num:
            train_mask = np.ones(length)
            train_mask[group*group_num-margin:] = 0
            
            covar_mask = np.ones((length, length))
            covar_mask[train_mask==0, :] = 0
            covar_mask[:, train_mask==0] = 0
            
            
            covar = np.mean(covar_list, axis=0)
            covar = covar[covar_mask>0]
            train_length = np.sum(train_mask, axis=0)
            covar = covar.reshape(train_length, train_length)
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, l2, False)[0]
            #fit_coe.append(result)
            fit_flux[group*group_num:] = np.dot(neighor_flux_matrix[group_mask == group], result)
            fit_coe = np.concatenate((fit_coe, np.array([result])), axis=0)
            np.save('./kic%d/coe/kic%d_num%d-%d_rge%.0e_whole_coe.npy'%(kid, kid, offset+1, num, l2), fit_coe)
            np.save('./kic%d/coe/kic%d_num%d-%d_reg%.0e_whole.npy'%(kid, kid, offset+1, num, l2), fit_flux)

            '''
            for pixel in range(0, target_flux.shape[1]):
                covar = covar_list[pixel]
                covar = covar[covar_mask>0]
                train_length = np.sum(train_mask, axis=0)
                covar = covar.reshape(train_length, train_length)
                result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0, pixel], covar, l2, False)[0]
                fit_flux[group*group_num:, pixel] = np.dot(neighor_flux_matrix[group_mask == group], result)
            '''

        light_curve = np.sum(target_flux, axis=1)
        fit_light_curve = np.sum(fit_flux, axis=1)
        light_curve_ratio = np.divide(light_curve, fit_light_curve)

        plm_mean = np.mean(light_curve_ratio)
        plm_std = np.std(light_curve_ratio)

        dev = light_curve_ratio - 1.
        rms = np.sqrt(np.mean(dev**2, axis=0))

        t = tm.time()
        print(t-t0)

        f, axes = plt.subplots(4, 1)
        axes[0].plot(time, light_curve, '.b', markersize=1)
        plt.setp( axes[0].get_xticklabels(), visible=False)
        plt.setp( axes[0].get_yticklabels(), visible=False)
        axes[0].set_ylabel("light curve")

        ylim = axes[0].get_ylim()

        axes[1].plot(time, fit_light_curve, '.b', markersize=1)
        plt.setp( axes[1].get_xticklabels(), visible=False)
        plt.setp( axes[1].get_yticklabels(), visible=False)
        axes[1].set_ylim(ylim)
        axes[1].set_ylabel("fit")

        axes[2].plot(time, light_curve_ratio, '.b', markersize=2)
        plt.setp( axes[2].get_xticklabels(), visible=False)
        #plt.setp( axes[2].get_yticklabels(), visible=False)
        axes[2].set_ylim(0.999,1.001)
        #axes[2].set_xlabel("time")
        axes[2].set_ylabel("ratio")

        #plot the PDC curve
        star = client.star(kid)
        lc = star.get_light_curves(short_cadence=False)[quarter]
        data = lc.read()
        flux = data["PDCSAP_FLUX"]
        inds = np.isfinite(flux)
        flux = flux[inds]
        pdc_time = data["TIME"][inds]

        mean = np.mean(flux)
        flux = flux/mean

        axes[3].plot(pdc_time, flux, '.b', markersize=2)
        plt.setp( axes[3].get_yticklabels(), visible=False)
        axes[3].set_ylim(0.999,1.001)
        axes[3].set_ylabel("pdc flux")
        axes[3].set_xlabel("time [BKJD]")

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)
        plt.suptitle('Kepler %d Quarter %d L2-Reg %.0e \n Fit Source[Initial:%d Number:%d CCD:%r] Test Region:%d-%d'%(kid, quarter,  l2, offset+1, num, ccd, -margin, margin))
        plt.savefig('./kic%d/coe/lightCurve_%d_%d_%d_q%d_reg%.0e_pdc.png'%(kid, kid, offset+1, num, quarter, l2), dpi=190)

        plt.clf()

#generate lightcurve train-and-test, multithreads
    if True:
        t0 = tm.time()
        kid = 5088536
        quarter = 5
        offset = 0
        num = 90
        l2 = 1e5
        ccd = True
        normal = False
        constant = True
        auto = False
        poly = 0
        
        Pixel = 17
            
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant, auto, Pixel, poly)
        print (pixel_x,pixel_y, Pixel, num, l2)
        coe_len = neighor_flux_matrix.shape[1]
        
        np.save('./kic%d/poly/kic%dnum%d-%dpoly%d_neigbourkid.npy'%(kid, kid, offset+1, num, poly), neighor_kid)
        np.save('./kic%d/poly/kic%dnum%d-%dpoly%d_neigbourmask.npy'%(kid, kid, offset+1, num, poly), neighor_kplr_maskes)
        
        target_kplr_mask = target_kplr_mask.flatten()
        optimal_len = np.sum(target_kplr_mask==3)
        print optimal_len
        target_flux = target_flux[:, target_kplr_mask==3]
        #target_mean = target_mean[Pixel]
        #target_std = target_std[Pixel]
        covar_list = covar_list[target_kplr_mask==3]
        covar = np.mean(covar_list, axis=0)**2
        fit_flux = []
        fit_coe = []
        length = target_flux.shape[0]
        total_length = epoch_mask.shape[0]
        margin = 24
    
        thread_num = 3
        
        thread_len = total_length//thread_num
        last_len = total_length - (thread_num-1)*thread_len
        
        class fit_epoch(threading.Thread):
            def __init__(self, thread_id, intial, len, time_intial, time_len):
                threading.Thread.__init__(self)
                self.thread_id = thread_id
                self.intial = intial
                self.len = len
                self.time_intial = time_intial
                self.time_len = time_len
            def run(self):
                print('Starting%d'%self.thread_id)
                print (self.thread_id , self.time_intial, self.time_len)
                tmp_fit_coe = np.empty((self.time_len, coe_len, optimal_len))
                tmp_fit_flux = np.empty((self.time_len, optimal_len))
                time_stp = 0
                for i in range(self.intial, self.intial+self.len):
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
                    
                    covar_mask = np.ones((length, length))
                    covar_mask[train_mask==0, :] = 0
                    covar_mask[:, train_mask==0] = 0
                    
                    tmp_covar = covar[covar_mask>0]
                    train_length = np.sum(train_mask, axis=0)
                    tmp_covar = tmp_covar.reshape(train_length, train_length)
                    result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], tmp_covar, l2, False, poly)[0]
                    tmp_fit_coe[time_stp, :] = result
                    tmp_fit_flux[time_stp] = np.dot(neighor_flux_matrix[time_stp+time_intial], result)
                    #tmp_fit_flux[time_stp] = np.dot(neighor_flux_matrix[i,:], result)*target_std+target_mean
                    np.save('./kic%d/poly/lightcurve_kic%d%dtmp%d.npy'%(kid, kid, num, self.thread_id), tmp_fit_flux)
                    time_stp += 1
                    print('done%d'%i)
                np.save('./kic%d/poly/lightcurve_kic%d_num%d-%d_reg%.0e_whole_coe_tmp%d.npy'%(kid, kid, offset+1, num, l2, self.thread_id), tmp_fit_coe)
                print('Exiting%d'%self.thread_id)
        
        thread_list = []
        time_intial = 0
        for i in range(0, thread_num-1):
            intial = i*thread_len
            thread_epoch = epoch_mask[intial:intial+thread_len]
            time_len = np.sum(thread_epoch)
            thread = fit_epoch(i, intial, thread_len, time_intial, time_len)
            thread.start()
            thread_list.append(thread)
            time_intial += time_len
        
        intial = (thread_num-1)*thread_len
        thread_epoch = epoch_mask[intial:intial+last_len]
        time_len = np.sum(thread_epoch)
        thread = fit_epoch(thread_num-1, intial, last_len, time_intial, time_len)
        thread.start()
        thread_list.append(thread)
        
        for t in thread_list:
            t.join()
        print 'all done'
    
        offset = 0
        window = 0
        
        for i in range(0, thread_num):
            tmp_fit_flux = np.load('./kic%d/poly/lightcurve_kic%d%dtmp%d.npy'%(kid, kid, num, i))
            tmp_fit_coe = np.load('./kic%d/poly/lightcurve_kic%d_num%d-%d_reg%.0e_whole_coe_tmp%d.npy'%(kid, kid, offset+1, num, l2, i))
            if i==0:
                fit_coe = tmp_fit_coe
                fit_flux = tmp_fit_flux
            else:
                fit_coe = np.concatenate((fit_coe, tmp_fit_coe), axis = 0)
                fit_flux = np.concatenate((fit_flux, tmp_fit_flux), axis=0)
        np.save('./kic%d/poly/lightcurve_kic%dnum%d-%d_reg%.0e_poly%d_whole_coe.npy'%(kid, kid, offset+1, num, l2, poly), fit_coe)
        np.save('./kic%d/poly/lightcurve_kic%dnum%dw%d_%d_poly%d.npy'%(kid, kid, num, offset, window, poly), fit_flux)
        
        for i in range(0, thread_num):
            os.remove('./kic%d/poly/lightcurve_kic%d_num%d-%d_reg%.0e_whole_coe_tmp%d.npy'%(kid, kid, offset+1, num, l2, i))
            os.remove('./kic%d/poly/lightcurve_kic%d%dtmp%d.npy'%(kid, kid, num, i))
        
        target_lightcurve = np.sum(target_flux, axis=1)
        fit_lightcurve = np.sum(fit_flux, axis=1)
        ratio = np.divide(target_lightcurve, fit_lightcurve)

        star = client.star(kid)
        lc = star.get_light_curves(short_cadence=False)[quarter]
        data = lc.read()
        flux = data["PDCSAP_FLUX"]
        inds = np.isfinite(flux)
        flux = flux[inds]
        pdc_time = data["TIME"][inds]
        pdc_mean = np.mean(flux)
        flux = flux/pdc_mean

        t = tm.time()
        print(t-t0)

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
        
        print mean_list
        print pdc_mean_list
        print std_list
        print pdc_std_list
        
        median_std = np.median(std_list)
        pdc_median_std = np.median(pdc_std_list)
        print median_std
        print pdc_median_std
        
        whole_time = np.zeros_like(epoch_mask, dtype=float)
        whole_time[epoch_mask>0] = np.float64(time)
        transit_list = np.array([457.19073, 484.699412, 512.208094])
        transit_duration = 6.1
        
        half_len = round(transit_duration/2/0.5)
        measure_half_len = 3*24*2
        print half_len
        
        transit_mask = np.zeros_like(epoch_mask)
        for i in range(0,3):
            loc = (np.abs(whole_time-transit_list[i])).argmin()
            transit_mask[loc-measure_half_len:loc+measure_half_len+1] = 1
            transit_mask[loc-half_len:loc+half_len+1] = 2
            transit_mask[loc] = 3
        pdc_transit_mask = transit_mask[inds]
        transit_mask = transit_mask[epoch_mask>0]
        
        signal = np.mean(ratio[transit_mask==1])-np.mean(ratio[transit_mask>=2])
        pdc_signal = np.mean(flux[pdc_transit_mask==1])-np.mean(flux[pdc_transit_mask>=2])
        depth = np.mean(ratio[transit_mask==1])- np.mean(ratio[transit_mask==3])
        pdc_depth = np.mean(flux[pdc_transit_mask==1])- np.mean(flux[pdc_transit_mask==3])
        print (signal, depth)
        print (pdc_signal, pdc_depth)

        sn = signal/median_std
        pdc_sn =pdc_signal/pdc_median_std
        
        f, axes = plt.subplots(4, 1)
        axes[0].plot(time[transit_mask<2], target_lightcurve[transit_mask<2], '.b', markersize=1)
        axes[0].plot(time[transit_mask>=2], target_lightcurve[transit_mask>=2], '.k', markersize=1, label="Transit singal \n within 6 hrs window")
        plt.setp( axes[0].get_xticklabels(), visible=False)
        plt.setp( axes[0].get_yticklabels(), visible=False)
        axes[0].set_ylabel("Data")
        ylim = axes[0].get_ylim()
        axes[0].legend(loc=1, ncol=3, prop={'size':8})
        
        axes[1].plot(time, fit_lightcurve, '.b', markersize=1)
        plt.setp( axes[1].get_xticklabels(), visible=False)
        plt.setp( axes[1].get_yticklabels(), visible=False)
        axes[1].set_ylabel("Fit")
        axes[1].set_ylim(ylim)
        
        axes[2].plot(time[transit_mask<2], ratio[transit_mask<2], '.b', markersize=2)
        axes[2].plot(time[transit_mask>=2], ratio[transit_mask>=2], '.k', markersize=2, label="Transit singal \n within 6 hrs window")
        axes[2].plot(time, mean_list, 'r-')
        axes[2].plot(time, mean_list-std_list, 'r-')
        plt.setp( axes[2].get_xticklabels(), visible=False)
        #plt.setp( axes[2].get_yticklabels(), visible=False)
        axes[2].set_ylim(0.999,1.001)
        #axes[2].set_xlabel("time[BKJD]")
        axes[2].set_ylabel("Ratio")
        #axes[2].axhline(y=1-std,xmin=0,xmax=3,c="red",linewidth=0.5,zorder=0)
        #axes[2].annotate(r'$\sigma$', xy=(time[2000], 1-std), xytext=(time[2000]+1, 1-std-0.0003),
        #arrowprops=dict(arrowstyle="->", connectionstyle="arc,angleA=0,armA=30,rad=10"),)
        #axes[2].axhline(y=1,xmin=0,xmax=3,c="red",linewidth=0.5,zorder=0)
        axes[2].text(time[2000], 1.0006, 'S/N = %.3f'%sn)
        axes[2].legend(loc=1, ncol=3, prop={'size':8})
        
        #plot the PDC curve
        
        axes[3].plot(pdc_time[pdc_transit_mask<2], flux[pdc_transit_mask<2], '.b', markersize=2)
        axes[3].plot(pdc_time[pdc_transit_mask>=2], flux[pdc_transit_mask>=2], '.k', markersize=2, label="Transit singal \n within 6 hrs window")
        axes[3].plot(pdc_time, pdc_mean_list, 'r-')
        axes[3].plot(pdc_time, pdc_mean_list-pdc_std_list, 'r-')
        #plt.setp( axes[3].get_yticklabels(), visible=False)
        axes[3].set_ylim(0.999,1.001)
        axes[3].yaxis.tick_right()
        axes[3].set_ylabel("pdc flux")
        axes[3].set_xlabel("time [BKJD]")
        axes[3].text(time[2000], 1.0006, 'S/N = %.3f'%pdc_sn)
        axes[3].legend(loc=1, ncol=3, prop={'size':8})

        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)
        plt.suptitle('Kepler %d Quarter %d L2-Reg %.0e Auto:%r Window:%.1f-%.1f poly:%d\n Fit Source[Initial:%d Number:%d CCD:%r] Test Region:%d-%d'%(kid, quarter, l2, auto, offset/48., (window+offset)/48., poly, 1, num, ccd, -margin, margin))
        plt.savefig('./kic%d/poly/lightcurve_kic%d_%d_%d_q%d_reg%.0e_auto%r_window%d_%d_poly%d_pdc.png'%(kid, kid, 1, num, quarter, l2, auto, offset, window, poly), dpi=190)
                            
        plt.clf()

#fit a single pixel train-and-test
    if False:
        t0 = tm.time()
        kid = 5088536
        quarter = 5
        offset = 0
        num = 70
        l2 = 1e4
        ccd = True
        normal = False
        constant = True
        auto = True
        
        Pixel = 24

        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant, auto, Pixel)
        print (pixel_x,pixel_y, Pixel)
        
        print neighor_flux_matrix.shape

        target_flux = target_flux[:, Pixel]
        length = target_flux.shape[0]
        group_num = 1
        margin = 24
        
        group_mask = np.ones(length, dtype=int)
        loc = 0
        group = 0
        while length-loc >= group_num:
            group_mask[loc:loc+group_num] = group
            loc += group_num
            group += 1
        group_mask[loc:] = group

        fit_flux = np.empty_like(target_flux)

        for i in range(0, group):
            train_mask = np.ones(length)
            if (margin <= i*group_num) and (length-(i+1)*group_num >= margin):
                train_mask[i*group_num-margin:(i+1)*group_num+margin] = 0
            elif margin > i*group_num:
                train_mask[:(i+1)*group_num+margin] = 0
            else:
                train_mask[i*group_num-margin:] = 0
            
            covar_mask = np.ones((length, length))
            covar_mask[train_mask==0, :] = 0
            covar_mask[:, train_mask==0] = 0

            covar = covar_list[Pixel]
            covar = covar[covar_mask>0]
            train_length = np.sum(train_mask, axis=0)
            covar = covar.reshape(train_length, train_length)
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, l2, False)[0]
            fit_flux[i*group_num:(i+1)*group_num] = np.dot(neighor_flux_matrix[group_mask == i], result)
            np.save('./kic%d/auto/kic%d_(%d,%d)%dtest.npy'%(kid, kid, pixel_x, pixel_y, num), fit_flux)
            print('done%d'%i)

        if length > group*group_num:
            train_mask = np.ones(length)
            train_mask[group*group_num-margin:] = 0
            

            covar_mask = np.ones((length, length))
            covar_mask[train_mask==0, :] = 0
            covar_mask[:, train_mask==0] = 0
            
            covar = covar_list[Pixel]
            covar = covar[covar_mask>0]
            train_length = np.sum(train_mask, axis=0)
            covar = covar.reshape(train_length, train_length)
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, l2, False)[0]
            fit_flux[group*group_num:] = np.dot(neighor_flux_matrix[group_mask == group], result)

        t = tm.time()
        np.save('./kic%d/auto/kic%d_(%d,%d)%dtest.npy'%(kid, kid, pixel_x, pixel_y, num), fit_flux)
        print(t-t0)
        
        f, axes = plt.subplots(3, 1)
        axes[0].plot(time, target_flux, '.b', markersize=1)
        plt.setp( axes[0].get_xticklabels(), visible=False)
        plt.setp( axes[0].get_yticklabels(), visible=False)
        axes[0].set_ylabel("Data")

        axes[1].plot(time, fit_flux, '.b', markersize=1)
        plt.setp( axes[1].get_xticklabels(), visible=False)
        plt.setp( axes[1].get_yticklabels(), visible=False)
        axes[1].set_ylabel("Fit")

        ratio = np.divide(target_flux, fit_flux)

        offset = 24
        window = 72

        axes[2].plot(time[0:offset+window], ratio[0:offset+window], '.k', markersize=2)
        axes[2].plot(time[offset+window:length-offset-window], ratio[offset+window:length-offset-window], '.b', markersize=2)
        axes[2].plot(time[length-offset-window:length], ratio[length-offset-window:length], '.r', markersize=2)
        #plt.setp( axes[2].get_xticklabels(), visible=False)
        #plt.setp( axes[2].get_yticklabels(), visible=False)
        axes[2].set_ylim(0.999,1.001)
        axes[2].set_xlabel("time[BKJD]")
        axes[2].set_ylabel("Ratio")


        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)
        plt.suptitle('Kepler %d Quarter %d Pixel(%d,%d) L2-Reg %.0e \n Fit Source[Initial:%d Number:%d CCD:%r] Test Region:%d-%d'%(kid, quarter, pixel_x, pixel_y, l2, 1, num, ccd, -margin, margin))
        plt.savefig('./kic%d/auto/kic%d_(%d,%d)_%d_%d_q%d_reg%.0e_auto%r_test.png'%(kid, kid, pixel_x, pixel_y, 1, num, quarter, l2, auto), dpi=190)

        plt.clf()


#fit a single pixel train-and-test, multithreads
    if False:
        t0 = tm.time()
        kid = 5088536
        quarter = 5
        offset = 0
        num = 90
        l2 = 1e5
        ccd = True
        normal = False
        constant = True
        auto = False
        poly = 0
        
        pixel_list = [(17, 90, 1e5)]
        for Pixel, num, l2 in pixel_list:
            
            neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant, auto, Pixel, poly)
            print (pixel_x,pixel_y, Pixel, num, l2)
            coe_len = neighor_flux_matrix.shape[1]
            
            np.save('./kic%d/new/kic%d_(%d,%d)num%d-%d_neigbourkid.npy'%(kid, kid, pixel_x, pixel_y, offset+1, num), neighor_kid)
            np.save('./kic%d/new/kic%d_(%d,%d)num%d-%d_neigbourmask.npy'%(kid, kid, pixel_x, pixel_y, offset+1, num), neighor_kplr_maskes)
            
            target_flux = target_flux[:, Pixel]
            #target_mean = target_mean[Pixel]
            #target_std = target_std[Pixel]
            covar = covar_list[Pixel]**2
            fit_flux = []
            fit_coe = []
            length = target_flux.shape[0]
            total_length = epoch_mask.shape[0]
            margin = 24
            
            thread_num = 3
            
            thread_len = total_length//thread_num
            last_len = total_length - (thread_num-1)*thread_len
            
            '''
            thread_len = length//thread_num
            last_len = length - (thread_num-1)*thread_len
            '''
            class fit_epoch(threading.Thread):
                def __init__(self, thread_id, intial, len, time_intial, time_len):
                    threading.Thread.__init__(self)
                    self.thread_id = thread_id
                    self.intial = intial
                    self.len = len
                    self.time_intial = time_intial
                    self.time_len = time_len
                def run(self):
                    print('Starting%d'%self.thread_id)
                    print (self.thread_id , self.time_intial, self.time_len)
                    tmp_fit_coe = np.empty((self.time_len, coe_len))
                    tmp_fit_flux = np.empty(self.time_len)
                    time_stp = 0
                    for i in range(self.intial, self.intial+self.len):
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
                        
                        covar_mask = np.ones((length, length))
                        covar_mask[train_mask==0, :] = 0
                        covar_mask[:, train_mask==0] = 0
                        
                        tmp_covar = covar[covar_mask>0]
                        train_length = np.sum(train_mask, axis=0)
                        tmp_covar = tmp_covar.reshape(train_length, train_length)
                        result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], tmp_covar, l2, False, poly)[0]
                        tmp_fit_coe[time_stp, :] = result
                        tmp_fit_flux[time_stp] = np.dot(neighor_flux_matrix[time_stp+time_intial,:], result)
                        #tmp_fit_flux[time_stp] = np.dot(neighor_flux_matrix[i,:], result)*target_std+target_mean
                        np.save('./kic%d/new/kic%d_(%d,%d)%dtmp%d.npy'%(kid, kid, pixel_x, pixel_y, num, self.thread_id), tmp_fit_flux)
                        time_stp += 1
                        print('done%d'%i)
                    np.save('./kic%d/new/kic%d_num%d-%d_rge%.0e_whole_coe_tmp%d.npy'%(kid, kid, offset+1, num, l2, self.thread_id), tmp_fit_coe)
                    print('Exiting%d'%self.thread_id)
            
            thread_list = []
            time_intial = 0
            for i in range(0, thread_num-1):
                intial = i*thread_len
                thread_epoch = epoch_mask[intial:intial+thread_len]
                time_len = np.sum(thread_epoch)
                thread = fit_epoch(i, intial, thread_len, time_intial, time_len)
                thread.start()
                thread_list.append(thread)
                time_intial += time_len
            
            intial = (thread_num-1)*thread_len
            thread_epoch = epoch_mask[intial:intial+last_len]
            time_len = np.sum(thread_epoch)
            thread = fit_epoch(thread_num-1, intial, last_len, time_intial, time_len)
            thread.start()
            thread_list.append(thread)

            for t in thread_list:
                t.join()
            print 'all done'
            
            offset = 0
            window = 0

            for i in range(0, thread_num):
                tmp_fit_flux = np.load('./kic%d/new/kic%d_(%d,%d)%dtmp%d.npy'%(kid, kid, pixel_x, pixel_y, num, i))
                tmp_fit_coe = np.load('./kic%d/new/kic%d_num%d-%d_rge%.0e_whole_coe_tmp%d.npy'%(kid, kid, offset+1, num, l2, i))
                if i==0:
                    fit_coe = tmp_fit_coe
                    fit_flux = tmp_fit_flux
                else:
                    fit_coe = np.concatenate((fit_coe, tmp_fit_coe), axis = 0)
                    fit_flux = np.concatenate((fit_flux, tmp_fit_flux), axis=0)
            np.save('./kic%d/new/kic%d_(%d,%d)num%d-%d_rge%.0e_whole_coe.npy'%(kid, kid, pixel_x, pixel_y, offset+1, num, l2), fit_coe)
            np.save('./kic%d/new/kic%d_(%d,%d)%dw%d_%d.npy'%(kid, kid, pixel_x, pixel_y, num, offset, window), fit_flux)
            
            for i in range(0, thread_num):
                os.remove('./kic%d/new/kic%d_num%d-%d_rge%.0e_whole_coe_tmp%d.npy'%(kid, kid, offset+1, num, l2, i))
                os.remove('./kic%d/new/kic%d_(%d,%d)%dtmp%d.npy'%(kid, kid, pixel_x, pixel_y, num, i))
            
            t = tm.time()
            print(t-t0)
            #target_flux = target_flux*target_std + target_mean
            f, axes = plt.subplots(3, 1)
            axes[0].plot(time, target_flux, '.b', markersize=1)
            plt.setp( axes[0].get_xticklabels(), visible=False)
            plt.setp( axes[0].get_yticklabels(), visible=False)
            axes[0].set_ylabel("Data")
            
            axes[1].plot(time, fit_flux, '.b', markersize=1)
            plt.setp( axes[1].get_xticklabels(), visible=False)
            plt.setp( axes[1].get_yticklabels(), visible=False)
            axes[1].set_ylabel("Fit")
            
            ratio = np.divide(target_flux, fit_flux)
            
            axes[2].plot(time[0:offset+window], ratio[0:offset+window], '.k', markersize=2)
            axes[2].plot(time[offset+window:length-offset-window], ratio[offset+window:length-offset-window], '.b', markersize=2)
            axes[2].plot(time[length-offset-window:length], ratio[length-offset-window:length], '.r', markersize=2)
            #plt.setp( axes[2].get_xticklabels(), visible=False)
            #plt.setp( axes[2].get_yticklabels(), visible=False)
            axes[2].set_ylim(0.999,1.001)
            axes[2].set_xlabel("time[BKJD]")
            axes[2].set_ylabel("Ratio")
            
            
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0, hspace=0)
            plt.suptitle('Kepler %d Quarter %d Pixel(%d,%d) L2-Reg %.0e Auto:%r Window:%.1f-%.1f poly:%d\n Fit Source[Initial:%d Number:%d CCD:%r] Test Region:%d-%d'%(kid, quarter, pixel_x, pixel_y, l2, auto, offset/48., (window+offset)/48., poly, 1, num, ccd, -margin, margin))
            plt.savefig('./kic%d/new/kic%d_(%d,%d)_%d_%d_q%d_reg%.0e_auto%r_window%d_%d.png'%(kid, kid, pixel_x, pixel_y, 1, num, quarter, l2, auto, offset, window), dpi=190)
                                
            plt.clf()

#fit a single pixel train-and-test, multithreads, resume
    if False:
        t0 = tm.time()
        kid = 5088536
        quarter = 5
        offset = 0
        num = 86
        l2 = 1e4
        ccd = True
        normal = False
        constant = True
        auto = False
        
        pixel_list = [(17, 86, 1e4)]
        for Pixel, num, l2 in pixel_list:
            
            neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant, auto, Pixel)
            print (pixel_x,pixel_y, Pixel, num, l2)
            coe_len = neighor_flux_matrix.shape[1]
            
            target_flux = target_flux[:, Pixel]
            fit_flux = []
            fit_coe = []
            length = target_flux.shape[0]
            margin = 24
            
            thread_num = 4
            
            thread_len = length//thread_num
            last_len = length - (thread_num-1)*thread_len
            
            class fit_epoch(threading.Thread):
                def __init__(self, thread_id, intial, begin, len):
                    threading.Thread.__init__(self)
                    self.thread_id = thread_id
                    self.intial = intial
                    self.begin = begin
                    self.len = len
                def run(self):
                    print('Starting%d'%self.thread_id)
                    tmp_fit_coe = np.load('./kic%d/coe/kic%d_num%d-%d_rge%.0e_whole_coe_tmp%d.npy'%(kid, kid, offset+1, num, l2, self.thread_id))
                    tmp_fit_flux = np.load('./kic%d/coe/kic%d_(%d,%d)%dtmp%d.npy'%(kid, kid, pixel_x, pixel_y, num, self.thread_id))
                    for i in range(self.begin, self.intial+self.len):
                        train_mask = np.ones(length)
                        if i<margin:
                            train_mask[0:i+margin+1] = 0
                        elif i > length-margin-1:
                            train_mask[i-margin:] = 0
                        else:
                            train_mask[i-margin:i+margin+1] = 0
                        covar_mask = np.ones((length, length))
                        covar_mask[train_mask==0, :] = 0
                        covar_mask[:, train_mask==0] = 0
                        
                        covar = covar_list[Pixel]
                        covar = covar[covar_mask>0]
                        train_length = np.sum(train_mask, axis=0)
                        covar = covar.reshape(train_length, train_length)
                        result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, l2, False)[0]
                        tmp_fit_coe[i-self.intial, :] = result
                        np.save('./kic%d/coe/kic%d_num%d-%d_rge%.0e_whole_coe_tmp%d.npy'%(kid, kid, offset+1, num, l2, self.thread_id), tmp_fit_coe)
                        tmp_fit_flux[i-self.intial] = np.dot(neighor_flux_matrix[i,:], result)
                        np.save('./kic%d/coe/kic%d_(%d,%d)%dtmp%d.npy'%(kid, kid, pixel_x, pixel_y, num, self.thread_id), tmp_fit_flux)
                        print('done%d'%i)
                    print('Exiting%d'%self.thread_id)
            
            begin_list = [1002, 2122, 3249]
            thread_list = []
            i = 0
            for begin in begin_list:
                thread = fit_epoch(i, i*thread_len, begin, thread_len)
                thread.start()
                thread_list.append(thread)
                i += 1
            
            thread = fit_epoch(thread_num-1, (thread_num-1)*thread_len, 4371, last_len)
            thread.start()
            thread_list.append(thread)
            
            for t in thread_list:
                t.join()
            print 'all done'
            
            offset = 0
            window = 0
            
            for i in range(0, thread_num):
                tmp_fit_flux = np.load('./kic%d/coe/kic%d_(%d,%d)%dtmp%d.npy'%(kid, kid, pixel_x, pixel_y, num, i))
                tmp_fit_coe = np.load('./kic%d/coe/kic%d_num%d-%d_rge%.0e_whole_coe_tmp%d.npy'%(kid, kid, offset+1, num, l2, i))
                if i==0:
                    fit_coe = tmp_fit_coe
                else:
                    fit_coe = np.concatenate((fit_coe, tmp_fit_coe), axis = 0)
                fit_flux = np.concatenate((fit_flux, tmp_fit_flux), axis=0)
            np.save('./kic%d/coe/kic%d_(%d,%d)num%d-%d_rge%.0e_whole_coe.npy'%(kid, kid, pixel_x, pixel_y, offset+1, num, l2), fit_coe)
            np.save('./kic%d/coe/kic%d_(%d,%d)%dw%d_%d.npy'%(kid, kid, pixel_x, pixel_y, num, offset, window), fit_flux)
            
            for i in range(0, thread_num):
                os.remove('./kic%d/coe/kic%d_num%d-%d_rge%.0e_whole_coe_tmp%d.npy'%(kid, kid, offset+1, num, l2, i))
                os.remove('./kic%d/coe/kic%d_(%d,%d)%dtmp%d.npy'%(kid, kid, pixel_x, pixel_y, num, i))
            
            t = tm.time()
            print(t-t0)
            
            f, axes = plt.subplots(3, 1)
            axes[0].plot(time, target_flux, '.b', markersize=1)
            plt.setp( axes[0].get_xticklabels(), visible=False)
            plt.setp( axes[0].get_yticklabels(), visible=False)
            axes[0].set_ylabel("Data")
            
            axes[1].plot(time, fit_flux, '.b', markersize=1)
            plt.setp( axes[1].get_xticklabels(), visible=False)
            plt.setp( axes[1].get_yticklabels(), visible=False)
            axes[1].set_ylabel("Fit")
            
            ratio = np.divide(target_flux, fit_flux)
            
            axes[2].plot(time[0:offset+window], ratio[0:offset+window], '.k', markersize=2)
            axes[2].plot(time[offset+window:length-offset-window], ratio[offset+window:length-offset-window], '.b', markersize=2)
            axes[2].plot(time[length-offset-window:length], ratio[length-offset-window:length], '.r', markersize=2)
            #plt.setp( axes[2].get_xticklabels(), visible=False)
            #plt.setp( axes[2].get_yticklabels(), visible=False)
            axes[2].set_ylim(0.999,1.001)
            axes[2].set_xlabel("time[BKJD]")
            axes[2].set_ylabel("Ratio")
            
            
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0, hspace=0)
            plt.suptitle('Kepler %d Quarter %d Pixel(%d,%d) L2-Reg %.0e Auto:%r Window:%.1f-%.1f \n Fit Source[Initial:%d Number:%d CCD:%r] Test Region:%d-%d'%(kid, quarter, pixel_x, pixel_y, l2, auto, offset/48., (window+offset)/48., 1, num, ccd, -margin, margin))
            plt.savefig('./kic%d/coe/kic%d_(%d,%d)_%d_%d_q%d_reg%.0e_auto%r_window%d_%d.png'%(kid, kid, pixel_x, pixel_y, 1, num, quarter, l2, auto, offset, window), dpi=190)
                                
            plt.clf()

#plot distribution of predictors in CCD
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 100
        l2 = 0
        ccd = True
        normal = False
        constant = True
        auto = True
        
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant, auto, Pixel)

        plt.plot(column[0], row[0], 'rs', label='target')
        plt.plot(column[1:], row[1:], 'bs', label='predictor')
        plt.xlabel("Column on CCD [Pixel]")
        plt.ylabel("Row on CCD [Pixel]")
        plt.legend(loc=1, ncol=3, prop={'size':8})
        plt.title('Star distribution on CCD')
        plt.xlim(1, 1132)
        plt.ylim(1, 1070)
        plt.savefig('./kic%d/distributionCCD'%kid, dpi=190)

#plot running median and rms
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 90
        l2 = 1e5
        ccd = True
        normal = False
        constant = True
        auto = False
        margin=24
        Pixel = 24

        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = mf.get_fit_matrix(kid, quarter, 1, 0, ccd, normal, 0, 0, 0, constant, auto, Pixel)

        target_kplr_mask = target_kplr_mask.flatten()
        pixel_list = np.arange(target_kplr_mask.shape[0])

        target_lightcurve = np.zeros(target_flux.shape[0])
        fit_lightcurve = np.zeros(target_flux.shape[0])

        for pixel in pixel_list:
            if target_kplr_mask[pixel] == 3:
                #neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, tmp_target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = mf.get_fit_matrix(kid, quarter, 1, offset, ccd, normal, 0, 0, 0, constant, pixel)
                target_lightcurve += target_flux[:, pixel]
        #fit_lightcurve += np.load('./kic%d_(%d,%d).npy'%(kid, pixel_x, pixel_y))
        #fit_lightcurve += np.load('./kic%d_(%d,%d).npy'%(kid, pixel_x, pixel_y))
        #fit_lightcurve += fit_flux
        fit_pixel = np.load('./kic5088536/kic5088536_num90_reg1e+05_whole.npy')
        fit_lightcurve = np.sum(fit_pixel, axis=1)
        ratio = np.divide(target_lightcurve, fit_lightcurve)
        dtype = [('index', int), ('erro', float)]
        erro = np.abs(ratio-1.)
        value = []
        erro_len = erro.shape[0]
        print erro_len
        for i in range(0, erro_len):
            value.append((i, erro[i]))
        erro_list = np.array(value, dtype=dtype)
        erro_list = np.sort(erro_list, kind='mergesort', order=['erro'])

        outlier_mask = np.zeros_like(time)

        for i in range(1, erro_len/10):
            index, erro_value = erro_list[erro_len-i]
            outlier_mask[index] = 1


        star = client.star(kid)
        lc = star.get_light_curves(short_cadence=False)[quarter]
        data = lc.read()
        flux = data["PDCSAP_FLUX"]
        inds = np.isfinite(flux)
        flux = flux[inds]
        pdc_time = data["TIME"][inds]
        pdc_mean = np.mean(flux)
        flux = flux/pdc_mean

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

        print mean_list
        print pdc_mean_list
        print std_list
        print pdc_std_list

        median_std = np.median(std_list)
        pdc_median_std = np.median(pdc_std_list)
        print median_std
        print pdc_median_std


        whole_time = np.zeros_like(epoch_mask, dtype=float)
        whole_time[epoch_mask>0] = np.float64(time)
        transit_list = np.array([457.19073, 484.699412, 512.208094])
        transit_duration = 6.1

        half_len = round(transit_duration/2/0.5)
        measure_half_len = 3*24*2
        print half_len

        transit_mask = np.zeros_like(epoch_mask)
        for i in range(0,3):
            loc = (np.abs(whole_time-transit_list[i])).argmin()
            transit_mask[loc-measure_half_len:loc+measure_half_len+1] = 1
            transit_mask[loc-half_len:loc+half_len+1] = 2
            transit_mask[loc] = 3
        pdc_transit_mask = transit_mask[inds]
        transit_mask = transit_mask[epoch_mask>0]

        signal = np.mean(ratio[transit_mask==1])-np.mean(ratio[transit_mask>=2])
        pdc_signal = np.mean(flux[pdc_transit_mask==1])-np.mean(flux[pdc_transit_mask>=2])
        depth = np.mean(ratio[transit_mask==1])- np.mean(ratio[transit_mask==3])
        pdc_depth = np.mean(flux[pdc_transit_mask==1])- np.mean(flux[pdc_transit_mask==3])
        print (signal, depth)
        print (pdc_signal, pdc_depth)


        sn = signal/median_std
        pdc_sn =pdc_signal/pdc_median_std

        f, axes = plt.subplots(4, 1)
        axes[0].plot(time[transit_mask<2], target_lightcurve[transit_mask<2], '.b', markersize=1)
        #axes[0].plot(time[transit_mask>=2], target_lightcurve[transit_mask>=2], '.k', markersize=1, label="Transit singal \n within 6 hrs window")
        axes[0].plot(time[outlier_mask>0], target_lightcurve[outlier_mask>0], '.r', markersize=1, label="Worst in fitting")
        plt.setp( axes[0].get_xticklabels(), visible=False)
        plt.setp( axes[0].get_yticklabels(), visible=False)
        axes[0].set_ylabel("Data")
        ylim = axes[0].get_ylim()
        axes[0].legend(loc=1, ncol=3, prop={'size':8})

        axes[1].plot(time, fit_lightcurve, '.b', markersize=1)
        plt.setp( axes[1].get_xticklabels(), visible=False)
        plt.setp( axes[1].get_yticklabels(), visible=False)
        axes[1].set_ylabel("Fit")
        axes[1].set_ylim(ylim)

        axes[2].plot(time[transit_mask<2], ratio[transit_mask<2], '.b', markersize=2)
        axes[2].plot(time[transit_mask>=2], ratio[transit_mask>=2], '.k', markersize=2, label="Transit singal \n within 6 hrs window")
        axes[2].plot(time, mean_list, 'r-')
        axes[2].plot(time, mean_list-std_list, 'r-')
        plt.setp( axes[2].get_xticklabels(), visible=False)
        #plt.setp( axes[2].get_yticklabels(), visible=False)
        axes[2].set_ylim(0.999,1.001)
        #axes[2].set_xlabel("time[BKJD]")
        axes[2].set_ylabel("Ratio")
        #axes[2].axhline(y=1-std,xmin=0,xmax=3,c="red",linewidth=0.5,zorder=0)
        #axes[2].annotate(r'$\sigma$', xy=(time[2000], 1-std), xytext=(time[2000]+1, 1-std-0.0003),
        #arrowprops=dict(arrowstyle="->", connectionstyle="arc,angleA=0,armA=30,rad=10"),)
        #axes[2].axhline(y=1,xmin=0,xmax=3,c="red",linewidth=0.5,zorder=0)
        axes[2].text(time[2000], 1.0006, 'S/N = %.3f'%sn)
        axes[2].legend(loc=1, ncol=3, prop={'size':8})

        #plot the PDC curve

        axes[3].plot(pdc_time[pdc_transit_mask<2], flux[pdc_transit_mask<2], '.b', markersize=2)
        axes[3].plot(pdc_time[pdc_transit_mask>=2], flux[pdc_transit_mask>=2], '.k', markersize=2, label="Transit singal \n within 6 hrs window")
        axes[3].plot(pdc_time, pdc_mean_list, 'r-')
        axes[3].plot(pdc_time, pdc_mean_list-pdc_std_list, 'r-')
        #plt.setp( axes[3].get_yticklabels(), visible=False)
        axes[3].set_ylim(0.999,1.001)
        axes[3].yaxis.tick_right()
        axes[3].set_ylabel("pdc flux")
        axes[3].set_xlabel("time [BKJD]")
        axes[3].text(time[2000], 1.0006, 'S/N = %.3f'%pdc_sn)
        axes[3].legend(loc=1, ncol=3, prop={'size':8})

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)
        plt.suptitle('Kepler %d Quarter %d L2-Reg %.0e \n Fit Source[Initial:%d Number:%d CCD:%r] Test Region:%d-%d'%(kid, quarter, l2, offset+1, num, ccd, -margin, margin))
        plt.savefig('./kic/5088536/auto/lightcurve/lightCurve_%d_%d_%d_q%d_reg%.0e_pdc_outlier.png'%(kid, offset+1, num, quarter, l2), dpi=190)






