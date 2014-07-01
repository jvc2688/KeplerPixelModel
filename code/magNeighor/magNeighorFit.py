import kplr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import leastSquareSolver as lss

client = kplr.API()
Pixel = 17
Percent = 0
Fake_Po = 2000
Fake_Len = 20

#fint the neighors in magnitude
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

def get_fake_data(target_flux, position, length):
    #the sine distortion
    '''
    factor = np.arange(target_flux.shape[0])
    factor = (1+0.004*np.sin(12*np.pi*factor/factor[-1]))
    for i in range(0, target_flux.shape[0]):
        target_flux[i] = target_flux[i] * factor[i]
    '''
    #the fake transit
    target_flux[position:position+length, :] = target_flux[position:position+length, :]*(1+0.004)
    return target_flux

#fit the target kic with the pixels of the neighors in magnitude
def get_fit_matrix(kic, quarter, neighor_num=1, offset=0, ccd=True, normal=False, fake_po=0, fake_len=0, constant=False):
    origin_star = client.star(kic)
    origin_tpf = client.target_pixel_files(ktc_kepler_id=origin_star.kepid, sci_data_quarter=quarter)[0]
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
    covar = np.zeros((flux_err.shape[0], flux_err.shape[0]))
    for i in range(0, flux_err.shape[0]):
        covar[i, i] = flux_err[i][Pixel]
    for i in range(0, len(neighor_fluxes)):
        neighor_fluxes[i] = neighor_fluxes[i][epoch_mask>0, :]
    #neighor_flux_matrix = neighor_flux_matrix[epoch_mask>0, :]
    
    #insert fake signal into the data
    if fake_len != 0:
        target_flux = get_fake_data(target_flux, fake_po, fake_len)

    #normalize the data
    if normal:
        for i in range(0, len(neighor_fluxes)):
            neighor_fluxes[i] = (neighor_fluxes[i] - np.mean(neighor_fluxes[i]))/np.std(neighor_fluxes[i])
        target_flux = (target_flux - np.mean(target_flux))/np.std(target_flux)

    #construt the neighor flux matrix
    neighor_flux_matrix = np.float64(np.concatenate(neighor_fluxes, axis=1))

    '''
    if normal:
        #neighor_flux_matrix = (neighor_flux_matrix - np.mean(neighor_flux_matrix))/np.var(neighor_flux_matrix)
        mean = np.mean(neighor_flux_matrix, axis=0)
        std = np.std(neighor_flux_matrix, axis=0)
        neighor_flux_matrix = (neighor_flux_matrix - mean)/std
        print(mean, mean.shape)
        print(std, std.shape)
        print(neighor_flux_matrix.shape)
        target_flux = (target_flux - np.mean(target_flux, axis=0))/np.std(target_flux, axis=0)
    '''

    
    #add the constant level
    if constant:
        neighor_flux_matrix = np.concatenate((neighor_flux_matrix, np.ones((neighor_flux_matrix.shape[0], 1))), axis=1)

    pixel_x = (Pixel+1)%target_pixel_mask.shape[2]
    pixel_y = (Pixel+1)//target_pixel_mask.shape[2]+1

    return neighor_flux_matrix, target_flux, covar, time, neighor_kid, neighor_kplr_maskes, epoch_mask, pixel_x, pixel_y


def get_fit_result(neighor_flux_matrix, target_flux, fit_coe):
    fit_flux = np.dot(neighor_flux_matrix, fit_coe)
    ratio = np.divide(target_flux, fit_flux)
    dev = ratio - 1.0
    rms = np.sqrt(np.mean(dev**2, axis=0))

    return fit_flux, ratio, rms

def plot_threepannel(target_flux, neighor_flux_matrix, time, fit_coe, prefix, title, fake=False):
    f, axes = plt.subplots(3, 1)
    
    axes[0].plot(time, target_flux[:, Pixel])
    plt.setp( axes[0].get_xticklabels(), visible=False)
    plt.setp( axes[0].get_yticklabels(), visible=False)
    axes[0].set_ylabel("flux of tpf")
    
    fit_flux = np.dot(neighor_flux_matrix, fit_coe)
    axes[1].plot(time, fit_flux[:, Pixel])
    plt.setp( axes[1].get_xticklabels(), visible=False)
    plt.setp( axes[1].get_yticklabels(), visible=False)
    axes[1].set_ylabel("flux of fit")
    
    ratio = np.divide(target_flux, fit_flux)
    
    result.append((np.mean(ratio[Fake_Po:Fake_Po+20, :], axis=0)-1.004)/0.004)
    axes[2].plot(time, np.divide(target_flux[:, Pixel], fit_flux[:, Pixel]))
    axes[2].set_ylim(0.999,1.001)
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("ratio of data and fit")
    #sin = 1+0.004*np.sin(12*np.pi*(time-time[0])/(time[-1]-time[0]))
    #axes[2].plot(time, sin, color='red')
    if fake:
        axes[2].axhline(y=1.004,xmin=0,xmax=3,c="red",linewidth=0.5,zorder=0)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0, hspace=0)
    plt.suptitle('%s'%title)
    plt.savefig('%s.png'%prefix)
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
        
        case_num = 20
        
        num_list = np.arange(case_num)
        bias = np.empty_like(num_list, dtype=float)
        
        for num in range(1, case_num+1):
            neighor_flux_matrix, target_flux, covar, time, neighor_kid, neighor_kplr_maskes, epoch_mask, pixel_x, pixel_y = get_fit_matrix(kid, quarter, num, offset, ccd, normal, Fake_Po, Fake_Len)
            num_list[num-1] = neighor_flux_matrix.shape[1]
            result = lss.leastSquareSolve(neighor_flux_matrix, target_flux, None, l2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[num-1] = (np.mean(ratio[Fake_Po:Fake_Po+Fake_Len, Pixel]) - 1.004)/0.004

        plt.plot(num_list, bias,'bs')
        #plt.title('test-train')
        plt.xlabel('Number of parameters')
        plt.ylabel('relative bias from the true singal')
        #plt.ylim(ymax=1)
        #plt.ylim(0.99, 1.01)
        plt.savefig('paraNum-bias_pixel(3,4).png')

#bias vs number of parameters in trian-and-test framework
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 20
        l2 = 0
        ccd = True
        normal = False
        
        case_num = 20
        
        num_list = np.arange(case_num)
        bias = np.empty_like(num_list, dtype=float)
        
        for num in range(1, case_num+1):
            neighor_flux_matrix, target_flux, covar, time, neighor_kid, neighor_kplr_maskes, epoch_mask, pixel_x, pixel_y = get_fit_matrix(kid, quarter, num, offset, ccd, normal, Fake_Po, Fake_Len)
            train_mask = get_train_mask(neighor_flux_matrix.shape[0], Percent, True, Fake_Po-24, Fake_Len+48)
            num_list[num-1] = neighor_flux_matrix.shape[1]
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], None, l2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[num-1] = (np.mean(ratio[Fake_Po:Fake_Po+Fake_Len, Pixel]) - 1.004)/0.004
        
        plt.plot(num_list, bias,'bs')
        plt.title('test-and-train')
        plt.xlabel('Number of parameters')
        plt.ylabel('relative bias from the true singal')
        plt.savefig('paraNum-bias_pixel(3,4)Test.png')

#Single fitting plot
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 5
        l2 = 0
        ccd = True
        normal = False
        neighor_flux_matrix, target_flux, covar, time, neighor_kid, neighor_kplr_maskes, epoch_mask, pixel_x, pixel_y = get_fit_matrix(kid, quarter, num, offset, ccd, normal)
        result = lss.leastSquareSolve(neighor_flux_matrix, target_flux, covar, l2, False)
        fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
        title = 'Kepler %d Quarter %d Pixel(%d,%d) L2-Reg %.0e \n Fit Source[Initial:%d Number:%d CCD:%r] RMS Deviation:%f'%(kid, quarter, pixel_x, pixel_y, l2, offset+1, num, ccd, rms[Pixel])
        file_prefix = 'fit(%d,%d)_%d_%d_reg%.0e_nor%r'%(pixel_x, pixel_y, offset+1, num, l2, normal)
        plot_threepannel(target_flux, neighor_flux_matrix, time, result[0], file_prefix, title)
        print_coe(result[0], neighor_kid, neighor_kplr_maskes, file_prefix, result[2][Pixel])

#k-fold validation
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        l2 = 0
        ccd = True
        normal = False
        k = 15
        
        case_num = 40
        
        num_list = np.zeros(case_num)
        rms_list = []
        for num in range(1, case_num+1):
            neighor_flux_matrix, target_flux, covar, time, neighor_kid, neighor_kplr_maskes, epoch_mask, pixel_x, pixel_y = get_fit_matrix(kid, quarter, num, offset, ccd, normal)
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
        plt.title('k-fold validation k=%d'%k)
        plt.xlabel('Number of parameters')
        plt.ylabel('RMS Deviation')
        plt.savefig('rms-num_k%dr.png'%k)

#signal bias vs l2 regularization
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 20
        l2 = 0
        ccd = True
        normal = False

        strength = np.arange(8)
        bias = np.empty_like(strength, dtype=float)
        
        neighor_flux_matrix, target_flux, covar, time, neighor_kid, neighor_kplr_maskes, epoch_mask, pixel_x, pixel_y = get_fit_matrix(kid, quarter, num, offset, ccd, normal, Fake_Po, Fake_Len)
        
        for i in strength:
            l2 = (1e3)*(10**strength[i])
            result = lss.leastSquareSolve(neighor_flux_matrix, target_flux, None, l2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[i] = (np.mean(ratio[Fake_Po:Fake_Po+Fake_Len, Pixel]) - 1.004)/0.004
            strength[i] += 3
        
        plt.clf()
        plt.plot(strength, bias, 'bs')
        plt.title('Number of stars:%d'%num)
        plt.xlabel(r'Strength of Regularization($log \lambda$)')
        plt.ylabel('relative bias from the true singal')
        plt.savefig('l2-bias_num%d_pixel(3,4).png'%num)

#signal with different location
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 40
        l2 = 0
        ccd = True
        normal = False
        
        case_num = 22
        
        position_list = np.arange(case_num)
        bias = np.empty_like(position_list, dtype=float)
        
        for i in range(0, case_num):
            position = Fake_Len*2+i*200
            neighor_flux_matrix, target_flux, covar, time, neighor_kid, neighor_kplr_maskes, epoch_mask, pixel_x, pixel_y = get_fit_matrix(kid, quarter, num, offset, ccd, normal, position, Fake_Len)
            position_list[i] = time[position]
            result = lss.leastSquareSolve(neighor_flux_matrix, target_flux, None, l2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[i] = (np.mean(ratio[position:position+Fake_Len, Pixel]) - 1.004)/0.004
        
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
        num = 5
        l2 = 0
        ccd = True
        normal = False
        
        case_num = 22
        
        position_list = np.arange(case_num)
        bias = np.empty_like(position_list, dtype=float)
        
        for i in range(0, case_num):
            position = Fake_Len*2+i*200
            neighor_flux_matrix, target_flux, covar, time, neighor_kid, neighor_kplr_maskes, epoch_mask, pixel_x, pixel_y = get_fit_matrix(kid, quarter, num, offset, ccd, normal, position, Fake_Len)
            position_list[i] = time[position]
            train_mask = get_train_mask(neighor_flux_matrix.shape[0], Percent, True, position-24, Fake_Len+48)
            result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], None, l2, False)
            fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
            bias[i] = (np.mean(ratio[position:position+Fake_Len, Pixel]) - 1.004)/0.004
        
        plt.clf()
        plt.plot(position_list, bias, 'bs')
        plt.title('Number of stars:%d \n trian-and-test'%num)
        plt.xlabel('location of the signal(time[BKJD])')
        plt.ylabel('relative bias from the true singal')
        plt.savefig('loc-bias_num%d_pixel(3,4)Test.png'%num)

#bias from 1 with different location and train-and-test framework
    if False:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 5
        l2 = 0
        ccd = True
        normal = False
        
        case_num = 22
        
        position_list = np.arange(case_num)
        bias = np.empty_like(position_list, dtype=float)
        
        for i in range(0, case_num):
            position = Fake_Len*2+i*200
            neighor_flux_matrix, target_flux, covar, time, neighor_kid, neighor_kplr_maskes, epoch_mask, pixel_x, pixel_y = get_fit_matrix(kid, quarter, num, offset, ccd, normal)
            position_list[i] = time[position]
            train_mask = get_train_mask(neighor_flux_matrix.shape[0], Percent, True, position-24, Fake_Len+48)
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
    if True:
        kid = 5088536
        quarter = 5
        offset = 0
        num = 5
        l2 = 0
        ccd = True
        normal = True
        constant = True
        neighor_flux_matrix, target_flux, covar, time, neighor_kid, neighor_kplr_maskes, epoch_mask, pixel_x, pixel_y = get_fit_matrix(kid, quarter, num, offset, ccd, normal, constant)
        result = lss.leastSquareSolve(neighor_flux_matrix, target_flux, covar, l2, False)
        fit_flux, ratio, rms = get_fit_result(neighor_flux_matrix, target_flux, result[0])
        title = 'Kepler %d Quarter %d Pixel(%d,%d) L2-Reg %.0e \n Fit Source[Initial:%d Number:%d CCD:%r] RMS Deviation:%f'%(kid, quarter, pixel_x, pixel_y, l2, offset+1, num, ccd, rms[Pixel])
        file_prefix = 'fit(%d,%d)_%d_%d_reg%.0e_nor%r_cons%r'%(pixel_x, pixel_y, offset+1, num, l2, normal, constant)
        plot_threepannel(target_flux, neighor_flux_matrix, time, result[0], file_prefix, title, False)
        print_coe(result[0], neighor_kid, neighor_kplr_maskes, file_prefix, result[2][Pixel], constant)
