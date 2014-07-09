import kplr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import leastSquareSolver as lss
import time as tm

client = kplr.API()
Pixel = 17
Percent = 0
Fake_Po = 2000
Fake_Len = 20

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
        time = hdu_data["time"]
        flux = hdu_data["flux"]
        flux_err = hdu_data["flux_err"]
    print flux.shape
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

def get_kfold_train_mask(length, k):
    step = length//k
    train_mask = np.ones(length, dtype=int)
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

#fit the target kic with the pixels of the neighors in magnitude
def get_fit_matrix(kic, quarter, neighor_num=1, offset=0, ccd=True, normal=False, fake_po=0, fake_len=0, fake_strength=0, constant=True):
    origin_star = client.star(kic)
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

    #construct covariance matrix
    covar_list = np.zeros((flux_err.shape[1], flux_err.shape[0], flux_err.shape[0]))
    for i in range(0, flux_err.shape[1]):
        #covar_list[] = np.zeros((flux_err.shape[0], flux_err.shape[0]))
        for j in range(0, flux_err.shape[0]):
            #covar[j, j] = flux_err[j][i]
            covar_list[i, j, j] = flux_err[j][i]
        #covar_list.append(covar)
    print covar_list.shape
    for i in range(0, len(neighor_fluxes)):
        neighor_fluxes[i] = neighor_fluxes[i][epoch_mask>0, :]
    #neighor_flux_matrix = neighor_flux_matrix[epoch_mask>0, :]
    
    #insert fake signal into the data
    if fake_len != 0:
        target_flux = get_fake_data(target_flux, fake_po, fake_len, fake_strength)

    #normalize the data
    if normal:
        '''
        for i in range(0, len(neighor_fluxes)):
            neighor_fluxes[i] = (neighor_fluxes[i] - np.mean(neighor_fluxes[i]))/np.std(neighor_fluxes[i])
        target_flux = (target_flux - np.mean(target_flux))/np.std(target_flux)
        '''
    #construt the neighor flux matrix
    neighor_flux_matrix = np.float64(np.concatenate(neighor_fluxes, axis=1))
    target_flux = np.float64(target_flux)

    target_mean, target_std = None, None
    if normal:
        #neighor_flux_matrix = (neighor_flux_matrix - np.mean(neighor_flux_matrix))/np.var(neighor_flux_matrix)
        
        mean = np.mean(neighor_flux_matrix, axis=0)
        std = np.std(neighor_flux_matrix, axis=0)
        neighor_flux_matrix = (neighor_flux_matrix - mean)/std

        target_mean = np.mean(target_flux, axis=0)
        target_std = np.std(target_flux, axis=0)
        target_flux = (target_flux - target_mean)/target_std
        #print(np.max(target_flux[:, Pixel]), np.min(target_flux[:, Pixel]))


    #add the constant level
    if constant:
        neighor_flux_matrix = np.concatenate((neighor_flux_matrix, np.ones((neighor_flux_matrix.shape[0], 1))), axis=1)
    

    pixel_x = (Pixel+1)%target_pixel_mask.shape[2]
    pixel_y = (Pixel+1)//target_pixel_mask.shape[2]+1

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
    
    if train_mask is not None:
        axes[0].plot(time[train_mask>0], target_flux[train_mask>0, Pixel], '.b', markersize=1, label='train')
        axes[0].plot(time[train_mask==0], target_flux[train_mask==0, Pixel], '.r', markersize=1, label='test')
        axes[0].legend()
    else:
        axes[0].plot(time, target_flux[:, Pixel])
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
        num = 80
        l2 = 1e-3
        ccd = True
        normal = False
        constant = True
        fake_strength = 0
        
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, fake_strength, constant)
        covar = covar_list[Pixel]
        print len(covar_list)
        print np.max(covar)
        result = lss.leastSquareSolve(neighor_flux_matrix, target_flux, covar, l2, False)
        fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0], target_mean, target_std)
        title = 'Kepler %d Quarter %d Pixel(%d,%d) L2-Reg %.0e \n Fit Source[Initial:%d Number:%d CCD:%r] RMS Deviation:%f'%(kid, quarter, pixel_x, pixel_y, l2, offset+1, num, ccd, rms[Pixel])
        file_prefix = 'fit(%d,%d)_%d_%d_reg%.0e_nor%r_cons%r_norm'%(pixel_x, pixel_y, offset+1, num, l2, normal, constant)
        plot_threepannel(target_flux, neighor_flux_matrix, time, result[0], file_prefix, title, False, 0, target_mean, target_std)
        print_coe(result[0], neighor_kid, neighor_kplr_maskes, file_prefix, result[2][Pixel], constant)

#k-fold validation
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        l2 = 0
        ccd = True
        normal = False
        constant = True
        k = 10
        case_num = 48
        
        num_list = np.zeros(case_num)
        rms_list = []
        for num in range(1, case_num+1):
            if num >= 49:
                l2= 1
            neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant)
            covar = covar_list[Pixel]
            num_list[num-1] = neighor_flux_matrix.shape[1]
            mean_rms = 0
            kfold_mask = get_kfold_train_mask(neighor_flux_matrix.shape[0], k)
            for i in range(0, k):
                result = lss.leastSquareSolve(neighor_flux_matrix[kfold_mask!=i, :], target_flux[kfold_mask!=i, :], None, l2, False)
                fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix[kfold_mask==i, :], target_flux[kfold_mask==i, :], result[0])
                mean_rms += rms[Pixel]
            mean_rms /= k
            rms_list.append(mean_rms)
            print 'done %d'%num

        plt.plot(num_list, rms_list, 'bs')
        plt.ylim(ymin=0)
        plt.title('k-fold validation k=%d L2 Reg: %.0e'%(k, l2))
        plt.xlabel('Number of parameters')
        plt.ylabel('RMS Deviation')
        plt.savefig('rms-num_k%d_reg%.0e.png'%(k, l2), dpi=150)

#k-fold validation for l2
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 30
        l2 = 0
        ccd = True
        normal = False
        constant = True
        k = 40
        case_num = 13
        
        l2_list = np.zeros(case_num)
        rms_list = []
        
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, 0, constant)
        covar = covar_list[Pixel]
        for case in range(1, case_num+1):
            l2 = (1e-3)*(10**(case-1))
            l2_list[case-1] = case-4
            mean_rms = 0
            kfold_mask = get_kfold_train_mask(neighor_flux_matrix.shape[0], k)
            for i in range(0, k):
                result = lss.leastSquareSolve(neighor_flux_matrix[kfold_mask!=i, :], target_flux[kfold_mask!=i, :], None, l2, False)
                fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix[kfold_mask==i, :], target_flux[kfold_mask==i, :], result[0])
                mean_rms += rms[Pixel]
            mean_rms /= k
            rms_list.append(mean_rms)
            print 'done %d'%case
        
        plt.plot(l2_list, rms_list, 'bs')
        plt.ylim(ymin=0)
        plt.title('k-fold validation k=%d Number of stars: %d'%(k, num))
        plt.xlabel(r'Strength of Regularization($log \lambda$)')
        plt.ylabel('RMS Deviation')
        plt.savefig('rms-l2_k%d_num%d.png'%(k, num), dpi=150)

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
        bias = np.empty((case_num, 4), dtype=float)
        
        num = 30
        
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
        
        f, axes = plt.subplots(4, 4)
        axes[0, 0].plot(position_list, bias[:, 0], 'bs')
        axes[0, 0].set_title('l2-Reg:0')
        axes[0, 0].set_ylabel('Num of Stars: %d'%num)
        axes[0, 0].set_ylim(plot_range)
        plt.setp(axes[0, 0].get_xticklabels(), visible=False)
        axes[0, 1].plot(position_list, bias[:, 1], 'bs')
        axes[0, 1].set_title('l2-Reg:1e3')
        axes[0, 1].set_ylim(plot_range)
        plt.setp(axes[0, 1].get_xticklabels(), visible=False)
        plt.setp(axes[0, 1].get_yticklabels(), visible=False)
        axes[0, 2].plot(position_list, bias[:, 2], 'bs')
        axes[0, 2].set_title('l2-Reg:1e5')
        axes[0, 2].set_ylim(plot_range)
        plt.setp(axes[0, 2].get_xticklabels(), visible=False)
        plt.setp(axes[0, 2].get_yticklabels(), visible=False)
        axes[0, 3].plot(position_list, bias[:, 3], 'bs')
        axes[0, 3].set_title('l2-Reg:1e7')
        axes[0, 3].set_ylim(plot_range)
        plt.setp(axes[0, 3].get_xticklabels(), visible=False)
        plt.setp(axes[0, 3].get_yticklabels(), visible=False)
        print('done1')

        num = 20
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
        axes[1, 3].plot(position_list, bias[:, 3], 'bs')
        axes[1, 3].set_ylim(plot_range)
        plt.setp(axes[1, 3].get_xticklabels(), visible=False)
        plt.setp(axes[1, 3].get_yticklabels(), visible=False)
        print('done2')

        num = 10
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
        axes[2, 3].plot(position_list, bias[:, 3], 'bs')
        axes[2, 3].set_ylim(plot_range)
        plt.setp(axes[2, 3].get_xticklabels(), visible=False)
        plt.setp(axes[2, 3].get_yticklabels(), visible=False)
        print('done3')

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


        fig = plt.gcf()
        fig.set_size_inches(18.5,20.5)
        fig.text(0.5, 0.03, 'Location of the signal(time[BKJD])', ha='center', va='center')
        fig.text(0.02, 0.5, 'Relative bias from the true signal\n [(Mearsured-True)/True]', ha='center', va='center', rotation='vertical')

        plt.subplots_adjust(left=0.1, bottom=0.07, right=0.97, top=0.95,
                    wspace=0, hspace=0.12)

        plt.suptitle('Signal Strength:%.4f Train-and-Test: %r'%(fake_strength, True))


        plt.savefig('loc-bias_num%d_pixel(3,4)Test_cons%r_diff_str%.4f_range(-1,1).png'%(num, constant, fake_strength))
        plt.show()

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
        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, constant)
        
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
        plt.savefig('lightCurve_%d_%d_%d_q%d_reg%.0e_nor%r_pdc_test.png'%(kid, offset+1, num, quarter, l2, normal), dpi=190)
        plt.show()
        plt.clf()
    
    if False:
        t0 = tm.time()
        kid = 5088536
        quarter = 5
        offset = 0
        num = 30
        l2 = 0
        ccd = True
        normal = False
        constant = True

        neighor_flux_matrix, target_flux, covar_list, time, neighor_kid, neighor_kplr_maskes, target_kplr_mask, epoch_mask, pixel_x, pixel_y, target_mean, target_std = get_fit_matrix(kid, quarter, num, offset, ccd, normal, 0, 0, constant)

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
            
            covar = np.mean(covar_list, axis=0)
            covar = covar[covar_mask>0]
            train_length = np.sum(train_mask, axis=0)
            covar = covar.reshape(train_length, train_length)
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], covar, l2, False)[0]
            fit_flux[i*group_num:(i+1)*group_num] = np.dot(neighor_flux_matrix[group_mask == i], result)
            '''
            for pixel in range(0, target_flux.shape[1]):
                covar = covar_list[pixel]
                covar = covar[covar_mask>0]
                train_length = np.sum(train_mask, axis=0)
                covar = covar.reshape(train_length, train_length)
                result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0, pixel], covar, l2, False)[0]
                fit_flux[i*group_num:(i+1)*group_num, pixel] = np.dot(neighor_flux_matrix[group_mask == i], result)
            '''
            print('done%d'%i)

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
            fit_flux[group*group_num:] = np.dot(neighor_flux_matrix[group_mask == group], result)
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

        dev = light_curve_ratio - 1.
        rms = np.sqrt(np.mean(dev**2, axis=0))

        t = tm.time()
        print(t-t0)

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
        plt.suptitle('Kepler %d Quarter %d L2-Reg %.0e \n Fit Source[Initial:%d Number:%d CCD:%r] Test Region:%d-%d'%(kid, quarter,  l2, offset+1, num, ccd, -margin, margin))
        plt.savefig('lightCurve_%d_%d_%d_q%d_reg%.0e_nor%r_pdc_train%d.png'%(kid, offset+1, num, quarter, l2, normal, group_num), dpi=190)

        plt.clf()


        f, axes = plt.subplots(4, 1)
        axes[0].plot(time, light_curve)
        plt.setp( axes[0].get_xticklabels(), visible=False)
        plt.setp( axes[0].get_yticklabels(), visible=False)
        axes[0].set_ylabel("light curve")

        axes[1].plot(time, fit_light_curve)
        plt.setp( axes[1].get_xticklabels(), visible=False)
        plt.setp( axes[1].get_yticklabels(), visible=False)
        axes[1].set_ylabel("fit")

        axes[2].plot(time, light_curve_ratio)
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

        axes[3].plot(pdc_time, flux)
        plt.setp( axes[3].get_yticklabels(), visible=False)
        axes[3].set_ylabel("pdc flux")
        axes[3].set_xlabel("time [BKJD]")

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)
        plt.suptitle('Kepler %d Quarter %d L2-Reg %.0e \n Fit Source[Initial:%d Number:%d CCD:%r] Test Region:%d-%d'%(kid, quarter,  l2, offset+1, num, ccd, -margin, margin))
        plt.savefig('lightCurve_%d_%d_%d_q%d_reg%.0e_nor%r_pdc_train%d_line.png'%(kid, offset+1, num, quarter, l2, normal, group_num), dpi=190)

        plt.show()
        plt.clf()


