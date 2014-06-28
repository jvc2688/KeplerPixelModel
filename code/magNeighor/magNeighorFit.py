import kplr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import leastSquareSolver as lss

client = kplr.API()
Pixel = 17
Percent = 0

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
    time = time[epoch_mask>0]
    flux[pixel_mask==0] = 0
    shape = flux.shape
    flux = flux[epoch_mask>0,:]
    flux = flux[:,kplr_mask>0]
    flux = flux.reshape((flux.shape[0], -1))

    #mask = np.array(np.sum(np.isfinite(flux), axis=0), dtype=bool)
    #flux = flux[:, mask]

    '''
    #interpolate the bad points
    for i in range(flux.shape[1]):
        interMask = np.isfinite(flux[:,i])
        flux[~interMask,i] = np.interp(time[~interMask], time[interMask], flux[interMask,i])
    '''
    print('time:%f'%(time[2]-time[1]))
    return time, flux, pixel_mask, kplr_mask

def get_train_mask(target_flux, percent=0.1, specify=False, initial=0, length=0):
    if specify:
        train_mask = np.ones(target_flux.shape[0])
        train_mask[initial:initial+length] = 0
    else:
        train_mask = np.ones(target_flux.shape[0])
        length = int(target_flux.shape[0] * percent)
        initial = int(target_flux.shape[0] * (0.5-percent/2.0))
        print length
        train_mask[initial:initial+length] = 0
    return train_mask

def get_fake_data(target_flux, position, length):
    #the sine distortion
    
    factor = np.arange(target_flux.shape[0])
    factor = (1+0.004*np.sin(12*np.pi*factor/factor[-1]))
    for i in range(0, target_flux.shape[0]):
        target_flux[i] = target_flux[i] * factor[i]
    
    #the fake transit
    #target_flux[position:position+length, :] = target_flux[position:position+length, :]*(1+0.004)
    return target_flux

#fit the target kic with the pixels of the neighors in magnitude
def neighor_fit(kic, quarter, neighor_num=1, offset=0, ccd=True, l2=0, plot=True):
    origin_star = client.star(kic)
    origin_tpf = client.target_pixel_files(ktc_kepler_id=origin_star.kepid, sci_data_quarter=quarter)[0]
    neighor = find_mag_neighor(origin_tpf, neighor_num, offset, ccd)

    time, target_flux, target_pixel_mask, target_kplr_mask = load_data(origin_tpf)

    neighor_kid, neighor_fluxes, neighor_pixel_maskes, neighor_kplr_maskes = [], [], [], []

    for key, tpf in neighor.items():
        neighor_kid.append(key)
        tmpResult = load_data(tpf)
        neighor_fluxes.append(tmpResult[1])
        neighor_pixel_maskes.append(tmpResult[2])
        neighor_kplr_maskes.append(tmpResult[3])
    
    neighor_flux_matrix = np.float64(np.concatenate(neighor_fluxes, axis=1))

    target_flux = get_fake_data(target_flux, 2000, 20)

    train_mask = get_train_mask(target_flux, Percent)
    #train_mask = get_train_mask(target_flux, Percent, True, 1986, 68)
    target_train_set = target_flux[train_mask>0, :]
    neighor_train_set = neighor_flux_matrix[train_mask>0, :]
    target_test_set = target_flux[train_mask==0, :]
    neighor_test_set = neighor_flux_matrix[train_mask==0, :]
    time_test = time[train_mask==0]

    result = lss.leastSquareSolve(neighor_train_set, target_train_set, l2)

    pixel_x = (Pixel+1)%target_pixel_mask.shape[2]
    pixel_y = (Pixel+1)//target_pixel_mask.shape[2]+1

    test_fit = np.dot(neighor_test_set, result[0])
    test_ratio = np.divide(target_test_set, test_fit)
    test_dev = test_ratio - 1.0
    test_rms = np.sqrt(np.mean(test_dev**2, axis=0))
    result.append(test_rms)

#plot the three pannel figure of the test set
    if False:
        f, axes = plt.subplots(3, 1)
        
        axes[0].plot(time_test, target_test_set[:, Pixel])
        plt.setp( axes[0].get_xticklabels(), visible=False)
        plt.setp( axes[0].get_yticklabels(), visible=False)
        axes[0].set_ylabel("flux of tpf")
        
        axes[1].plot(time_test, test_fit[:, Pixel])
        plt.setp( axes[1].get_xticklabels(), visible=False)
        plt.setp( axes[1].get_yticklabels(), visible=False)
        axes[1].set_ylabel("flux of fit")
        
        axes[2].plot(time_test, test_ratio[:, Pixel])
        axes[2].set_ylim(0.999,1.001)
        axes[2].set_xlabel("time")
        axes[2].set_ylabel("ratio of data and fit")
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)
        plt.suptitle('Kepler %d Quarter %d Pixel(%d,%d) \n Fit Source[Initial:%d Number:%d CCD:%r] RMS Deviation:%f'%(origin_tpf.ktc_kepler_id, origin_tpf.sci_data_quarter, pixel_y, pixel_x, offset+1, neighor_num, ccd, test_rms[Pixel]))
        plt.savefig('fit(%d,%d)_%d_%d_ccd%r_test.png'%(pixel_y, pixel_x, offset+1,neighor_num,ccd))
        plt.clf()


    #plot the three pannel figure of target pixel
    if True:
        f, axes = plt.subplots(3, 1)

        axes[0].plot(time, target_flux[:, Pixel])
        plt.setp( axes[0].get_xticklabels(), visible=False)
        plt.setp( axes[0].get_yticklabels(), visible=False)
        axes[0].set_ylabel("flux of tpf")

        fit_flux = np.dot(neighor_flux_matrix, result[0])
        axes[1].plot(time, fit_flux[:, Pixel])
        plt.setp( axes[1].get_xticklabels(), visible=False)
        plt.setp( axes[1].get_yticklabels(), visible=False)
        axes[1].set_ylabel("flux of fit")
        
        ratio = np.divide(target_flux, fit_flux)
        result.append(np.mean(ratio[2000:2020, :], axis=0)/1.004)
        axes[2].plot(time, np.divide(target_flux[:, Pixel], fit_flux[:, Pixel]))
        axes[2].set_ylim(0.990,1.010)
        axes[2].set_xlabel("time")
        axes[2].set_ylabel("ratio of data and fit")
        sin = 1+0.004*np.sin(12*np.pi*(time-time[0])/(time[-1]-time[0]))
        axes[2].plot(time, sin, color='red')
        #axes[2].axhline(y=1.004,xmin=0,xmax=3,c="red",linewidth=0.5,zorder=0)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)
        plt.suptitle('Kepler %d Quarter %d Pixel(%d,%d) \n Fit Source[Initial:%d Number:%d CCD:%r] RMS Deviation:%f'%(origin_tpf.ktc_kepler_id, origin_tpf.sci_data_quarter, pixel_y, pixel_x, offset+1, neighor_num, ccd, result[2][Pixel]))
        plt.savefig('fit(%d,%d)_%d_%d_ccd%rTran_reg%.0eSin.png'%(pixel_y, pixel_x, offset+1,neighor_num,ccd, l2))
        plt.clf()

    #print the fitting coefficient
    f = open('coe(2,4)_%d_%d_ccd%r.dat'%(offset+1,neighor_num,ccd), 'w')
    loc = 0
    for n in range(0, neighor_num):
        kplr_mask = neighor_kplr_maskes[n].flatten()
        #coe = result[0][:, Pixel]
        coe = np.zeros_like(kplr_mask, dtype=float)
        #coe = np.zeros_like(neighor_pixel_maskes[n], dtype=float)
        #coe = np.ma.masked_equal(coe,0)
        coe[kplr_mask>0] = result[0][loc:loc+neighor_fluxes[n].shape[1], Pixel]
        loc += neighor_fluxes[n].shape[1]
        coe = coe.reshape((neighor_pixel_maskes[n].shape[1],neighor_pixel_maskes[n].shape[2]))

        f.write('fit coefficient of the pixels of kepler %d\n'%neighor_kid[n])
        f.write('================================================\n')
        for i in range(coe.shape[0]):
            for j in range(coe.shape[1]):
                f.write('%8.5f   '%coe[i,j])
            f.write('\n')
        f.write('================================================\n')
    f.write('RMS Deviation:%f'%result[2][Pixel])
    f.close()
    return result

if __name__ == "__main__":
    if False:
        neighor_num = np.arange(20)+1
        ratio = np.empty_like(neighor_num, dtype=float)
        for i in neighor_num:
            result = neighor_fit(5088536, 5, i, 0, True, 0)
            ratio[i-1] = result[4][Pixel]
            neighor_num[i-1] = result[0].shape[0]
        plt.plot(neighor_num,ratio,'bs')
        plt.title('test-train')
        plt.xlabel("Number of parameters")
        plt.ylabel("ratio of the fitting and true singal")
        #plt.ylim(ymax=1)
        plt.ylim(0.99, 1.01)
        plt.savefig('paraNum-ratio_pixel(3,4)Train.png')

    if False:
        neighor_num = np.arange(12)+1
        residuals = np.empty_like(neighor_num, dtype=float)
        rms = np.empty_like(neighor_num, dtype=float)
        for i in neighor_num:
            result = neighor_fit(5088536, 5, i, 0, True, 0)
            residuals[i-1] = result[1][Pixel]
            rms[i-1] = result[3][Pixel]
            neighor_num[i-1] = result[0].shape[0]
        plt.clf()
        plt.plot(neighor_num,residuals,'bs')
        plt.xlabel("Number of parameters")
        plt.ylabel("Total squared residuals")
        plt.savefig('paraNum-res.png')
        plt.clf()
        plt.title('%d%% test data'%int(Percent*100))
        plt.plot(neighor_num,rms,'bs')
        plt.xlabel("Number of parameters")
        plt.ylabel("RMS Deviation")
        plt.ylim(ymin=0)
        plt.savefig('paraNum-rms_pixel(2,4)_%.1f.png'%Percent)

    if False:
        neighor_fit(5088536, 5, 1, 0, True)
        neighor_fit(5088536, 5, 2, 0, True)
        neighor_fit(5088536, 5, 1, 1, True)
        neighor_fit(5088536, 5, 1, 0, False)

    if True:
        result = neighor_fit(5088536, 5, 5, 0, True, 0, True)

    if False:
        num = 5
        strength = np.arange(8)
        ratio = np.empty_like(strength, dtype=float)
        for i in strength:
            result = neighor_fit(5088536, 5, num, 0, True, (1e3)*(10**strength[i]), True)
            ratio[i] = result[4][Pixel]
            strength[i] += 3
        plt.clf()
        plt.plot(strength, ratio, 'bs')
        plt.title('Number of stars:%d'%num)
        plt.xlabel(r'Strength of Regularization($log \lambda$)')
        plt.ylabel("ratio of the fitting and true singal")
        plt.ylim(ymax=1.0)
        plt.savefig('l2-ratio_num%d_pixel(3,4).png'%num)

    if False:
        strength = np.arange(7)
        rms = np.empty_like(strength, dtype=float)
        for i in strength:
            result = neighor_fit(5088536, 5, 5, 0, True, (1e-8)*(10**strength[i]))
            rms[i] = result[2][Pixel]
            strength[i] -= 8
        plt.clf()
        plt.plot(strength,rms,'bs')
        plt.xlabel(r'Strength of Regularization($log \lambda$)')
        plt.ylabel("RMS Deviation")
        plt.ylim(ymin=0)
        plt.savefig('l2-rms.png')

    if False:
        neighor_num = np.arange(5)+1
        para_num = np.empty_like(neighor_num)
        print neighor_num
        strength = np.arange(5)
        ratio = np.empty((strength.shape[0], neighor_num.shape[0]), dtype=float)
        for i in strength:
            for j in neighor_num:
                print(i, j)
                result = neighor_fit(5088536, 5, j, 0, True, (10**strength[i]), True)
                ratio[i, j-1] = result[4][Pixel]
                para_num[j-1] = result[0].shape[0]
        plt.clf()
        im = plt.imshow(ratio, interpolation='bilinear', origin='lower', cmap=cm.gray, extent=(-3,3,-2,2))
        plt.contourf(strength, para_num, ratio)
        #plt.title('Number of stars:%d'%num)
        plt.xlabel(r'Strength of Regularization($log \lambda$)')
        plt.ylabel("Number of parameters")
        plt.flag()
        plt.colorbar(im, orientation='horizontal', shrink=0.8)
        #plt.ylim(ymax=1.0)
        plt.savefig('l2-num-ratio_pixel(3,4).png')

