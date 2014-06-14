import kplr
import numpy as np
import matplotlib.pyplot as plt

client = kplr.API()

def findMagNeighor(originTpf, num):
    starsOver = client.target_pixel_files(kic_kepmag=">=%f"%originStar.kic_kepmag, sci_data_quarter=originTpf.sci_data_quarter, sci_channel=originTpf.sci_channel, sort=("kic_kepmag", 1),ktc_target_type="LC", max_records=num+1)
    starsUnder = client.target_pixel_files(kic_kepmag="<=%f"%originStar.kic_kepmag, sci_data_quarter=originTpf.sci_data_quarter, sci_channel=originTpf.sci_channel, sort=("kic_kepmag", -1),ktc_target_type="LC", max_records=num+1)
    
    stars = {}
    stars[originTpf.ktc_kepler_id] = originTpf
    
    i=0
    j=0
    while len(stars) <num+1:
        while starsOver[i].ktc_kepler_id in stars:
            i+=1
        tmpOver = starsOver[i]
        while starsUnder[j].ktc_kepler_id in stars:
            j+=1
        tmpUnder = starsUnder[j]
        if tmpOver.kic_kepmag-originStar.kic_kepmag > originStar.kic_kepmag-tmpUnder.kic_kepmag:
            stars[tmpUnder.ktc_kepler_id] = tmpUnder
            j+=1
        elif tmpOver.kic_kepmag-originStar.kic_kepmag < originStar.kic_kepmag-tmpUnder.kic_kepmag:
            stars[tmpOver.ktc_kepler_id] = tmpOver
            j+=1
        else:
            stars[tmpUnder.ktc_kepler_id] = tmpUnder
            stars[tmpOver.ktc_kepler_id] = tmpOver
            i+=1
            j+=1

    stars.pop(originTpf.ktc_kepler_id)

    return stars

def convertData(tdf):
    time, flux = [], []
    with tdf.open() as file:
        hdu_data = file[1].data
        time = hdu_data["time"]
        flux = hdu_data["flux"]
    shape = flux.shape
    flux = flux.reshape((flux.shape[0], -1))
    mask = np.array(np.sum(np.isfinite(flux), axis=0), dtype=bool)
    flux = flux[:, mask]

    for i in range(flux.shape[1]):
        interMask = np.isfinite(flux[:,i])
        flux[~interMask,i] = np.interp(time[~interMask], time[interMask], flux[interMask,i])

    return time, flux, mask, shape


originStar = client.star(5088536)
originTpf = client.target_pixel_files(ktc_kepler_id=originStar.kepid, sci_data_quarter=5)[0]
neighor = findMagNeighor(originTpf, 1)

time, targetFlux, targetMask, targetShape = convertData(originTpf)

neighorKID, neighorFluxes, neighorMaskes, neighorShapes = [], [], [], []

for key,tpf in neighor.items():
    neighorKID.append(key)
    tmpResult = convertData(tpf)
    neighorFluxes.append(tmpResult[1])
    neighorMaskes.append(tmpResult[2])
    neighorShapes.append(tmpResult[3])
neighorFluxMatrix = np.concatenate(neighorFluxes, axis=1)

result  = np.linalg.lstsq(neighorFluxMatrix, targetFlux)

f, axes = plt.subplots(3, 1)

axes[0].plot(time, targetFlux[:,10])
plt.setp( axes[0].get_xticklabels(), visible=False)
plt.setp( axes[0].get_yticklabels(), visible=False)
axes[0].set_ylabel("flux of tpf")

fitFlux = np.dot(neighorFluxMatrix, result[0])
axes[1].plot(time, fitFlux[:,10])
plt.setp( axes[1].get_xticklabels(), visible=False)
plt.setp( axes[1].get_yticklabels(), visible=False)
axes[1].set_ylabel("flux of fit")

axes[2].plot(time, np.divide(targetFlux[:,10], fitFlux[:,10]))
#plt.setp( axes[2].get_xticklabels(), visible=False)
#plt.setp( axes[2].get_yticklabels(), visible=False)
axes[2].set_ylim(0.999,1.001)
axes[2].set_xlabel("time")
axes[2].set_ylabel("ratio of data and fit")

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0, hspace=0)
plt.suptitle('Kepler %d Quarter %d CCD Channel %d\n Kepler magnitude %f'%(originTpf.ktc_kepler_id, originTpf.sci_data_quarter, originTpf.sci_channel, originTpf.kic_kepmag))
plt.savefig('fitPixel(2,4).png')
plt.clf()

coe = np.zeros_like(neighorMaskes[0], dtype=float)
coe = np.ma.masked_equal(coe,0)
coe[neighorMaskes[0]] = result[0][:,10]
coe = coe.reshape((neighorShapes[0][1],neighorShapes[0][2]))

f = open('coe.dat', 'w')
f.write('fit coefficient of the pixels of kepler %d\n'%neighorKID[0])
f.write('================================================\n')
for i in range(coe.shape[0]):
    for j in range(coe.shape[1]):
        f.write('%8.5f   '%coe[i,j])
    f.write('\n')
f.close()




