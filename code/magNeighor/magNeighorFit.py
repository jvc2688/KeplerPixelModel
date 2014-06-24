import kplr
import numpy as np
import matplotlib.pyplot as plt
import leastSquareSolver as lss

client = kplr.API()

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
    time, flux = [], []
    with tpf.open() as file:
        hdu_data = file[1].data
        kplr_mask = file[2].data
        time = hdu_data["time"]
        flux = hdu_data["flux"]
    pixel_mask = get_pixel_mask(flux, kplr_mask)
    epoch_mask = get_epoch_mask(pixel_mask)
    time = time[epoch_mask>0]
    shape = flux.shape
    flux[pixel_mask==0] = 0
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
    return time, flux, pixel_mask, kplr_mask

#fit the target kic with the pixels of the neighors in magnitude
def neighor_fit(kic, quarter, neighor_num=1, offset=0, ccd=True, l2=0, plot=True):
    origin_star = client.star(kic)
    origin_tpf = client.target_pixel_files(ktc_kepler_id=origin_star.kepid, sci_data_quarter=quarter)[0]
    neighor = find_mag_neighor(origin_tpf, neighor_num, offset, ccd)

    time, target_flux, targetMask, targetShape = load_data(origin_tpf)

    neighor_kid, neighor_fluxes, neighor_pixel_maskes, neighor_kplr_maskes = [], [], [], []

    for key,tpf in neighor.items():
        neighor_kid.append(key)
        tmpResult = load_data(tpf)
        neighor_fluxes.append(tmpResult[1])
        neighor_pixel_maskes.append(tmpResult[2])
        neighor_kplr_maskes.append(tmpResult[3])
    
    neighor_flux_matrix = np.float64(np.concatenate(neighor_fluxes, axis=1))

    result = lss.leastSquareSolve(neighor_flux_matrix, target_flux, l2)

    #plot the three pannel figure of target pixel
    if plot:
        f, axes = plt.subplots(3, 1)

        axes[0].plot(time, target_flux[:,10])
        plt.setp( axes[0].get_xticklabels(), visible=False)
        plt.setp( axes[0].get_yticklabels(), visible=False)
        axes[0].set_ylabel("flux of tpf")

        fit_flux = np.dot(neighor_flux_matrix, result[0])
        axes[1].plot(time, fit_flux[:,10])
        plt.setp( axes[1].get_xticklabels(), visible=False)
        plt.setp( axes[1].get_yticklabels(), visible=False)
        axes[1].set_ylabel("flux of fit")

        axes[2].plot(time, np.divide(target_flux[:,10], fit_flux[:,10]))
        #plt.setp( axes[2].get_xticklabels(), visible=False)
        #plt.setp( axes[2].get_yticklabels(), visible=False)
        axes[2].set_ylim(0.999,1.001)
        axes[2].set_xlabel("time")
        axes[2].set_ylabel("ratio of data and fit")

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0)
        plt.suptitle('Kepler %d Quarter %d Pixel(2,4) \n Fit Source[Initial:%d Number:%d CCD:%r] RMS Deviation:%f'%(origin_tpf.ktc_kepler_id, origin_tpf.sci_data_quarter, offset+1, neighor_num, ccd, result[2][10]))
        plt.savefig('fit(2,4)_%d_%d_ccd%r.png'%(offset+1,neighor_num,ccd))
        plt.clf()

    #print the fitting coefficient
    f = open('coe(2,4)_%d_%d_ccd%r.dat'%(offset+1,neighor_num,ccd), 'w')
    loc = 0
    for n in range(0, neighor_num):
        kplr_mask = neighor_kplr_maskes[n].flatten()
        #coe = result[0][:,10]
        coe = np.zeros_like(kplr_mask, dtype=float)
        #coe = np.zeros_like(neighor_pixel_maskes[n], dtype=float)
        #coe = np.ma.masked_equal(coe,0)
        coe[kplr_mask>0] = result[0][loc:loc+neighor_fluxes[n].shape[1],10]
        loc += neighor_fluxes[n].shape[1]
        coe = coe.reshape((neighor_pixel_maskes[n].shape[1],neighor_pixel_maskes[n].shape[2]))

        f.write('fit coefficient of the pixels of kepler %d\n'%neighor_kid[n])
        f.write('================================================\n')
        for i in range(coe.shape[0]):
            for j in range(coe.shape[1]):
                f.write('%8.5f   '%coe[i,j])
            f.write('\n')
        f.write('================================================\n')
    f.write('RMS Deviation:%f'%result[2][10])
    f.close()
    return result

if __name__ == "__main__":
    if False:
        neighor_num = np.arange(12)+1
        residuals = np.empty_like(neighor_num, dtype=float)
        rms = np.empty_like(neighor_num, dtype=float)
        for i in neighor_num:
            result = neighor_fit(5088536, 5, i, 0, True, 0)
            residuals[i-1] = result[1][10]
            rms[i-1] = result[2][10]
            neighor_num[i-1] = result[0].shape[0]
        plt.clf()
        plt.plot(neighor_num,residuals,'bs')
        plt.xlabel("Number of parameters")
        plt.ylabel("Total squared residuals")
        plt.savefig('paraNum-res.png')
        plt.clf()
        plt.plot(neighor_num,rms,'bs')
        plt.xlabel("Number of parameters")
        plt.ylabel("RMS Deviation")
        plt.ylim(ymin=0)
        plt.savefig('paraNum-rms.png')

    if False:
        neighor_fit(5088536, 5, 1, 0, True)
        neighor_fit(5088536, 5, 2, 0, True)
        neighor_fit(5088536, 5, 1, 1, True)
        neighor_fit(5088536, 5, 1, 0, False)

    if True:
        neighor_fit(5088536, 5, 5, 0, True, 0)

    if False:
        strength = np.arange(7)
        rms = np.empty_like(strength, dtype=float)
        for i in strength:
            result = neighor_fit(5088536, 5, 5, 0, True, (1e-8)*(10**strength[i]))
            rms[i] = result[2][10]
            strength[i] -= 8
        plt.clf()
        plt.plot(strength,rms,'bs')
        plt.xlabel(r'Strength of Regularization($log \lambda$)')
        plt.ylabel("RMS Deviation")
        plt.ylim(ymin=0)
        plt.savefig('l2-rms.png')