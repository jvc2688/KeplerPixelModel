import kplr
import numpy as np
import matplotlib.pyplot as plt

qua = 5

client = kplr.API()

# Find the target KOI.
koi = client.koi(282.02)

originStar = koi.star

# Find potential targets by Kepler magnitude
koisOver = client.kois(where="koi_kepmag between %f and %f"%(originStar.kic_kepmag, originStar.kic_kepmag+0.1), sort=("koi_kepmag",1))
koisUnder = client.kois(where="koi_kepmag between %f and %f"%(originStar.kic_kepmag-0.1, originStar.kic_kepmag), sort=("koi_kepmag",1))
koisUnder.reverse()

stars = []
stars.append(originStar.kepid)

#Find 16 stars that are closest to the origin star in terms of Kepler magnitude
i=0
j=0
while len(stars) <17:
    while koisOver[i].kepid in stars:
        i+=1
    tmpOver = koisOver[i].star
    while koisUnder[j].kepid in stars:
        j+=1
    tmpUnder =koisUnder[j].star
    if tmpOver.kic_kepmag-originStar.kic_kepmag > originStar.kic_kepmag-tmpUnder.kic_kepmag:
        stars.append(tmpUnder.kepid)
        j+=1
    elif tmpOver.kic_kepmag-originStar.kic_kepmag < originStar.kic_kepmag-tmpUnder.kic_kepmag:
        stars.append(tmpOver.kepid)
        j+=1
    else:
        stars.append(tmpUnder.kepid)
        stars.append(tmpOver.kepid)
        i+=1
        j+=1


for tmp in stars:
    star = client.star(tmp)
# Get a list of light curve datasets.
    tpfs = star.get_target_pixel_files(short_cadence=False)

    time, flux = [], []

    for tpf in tpfs:
        with tpf.open() as f:
            hdu_data = f[1].data
            time.append(hdu_data["time"])
            flux.append(hdu_data["flux"])

    t = time[qua]

    data = flux[qua]
    data = np.nan_to_num(data)
    data = np.ma.masked_equal(data,0)

    shape = data.shape
    td = shape[0]
    x = shape[1]
    y = shape[2]

# Plot the data
    f, axes = plt.subplots(x, y)

    for i in range(0,x):
        for j in range(0,y):
            axes[i,j].plot(t,data[0:td:1,i,j])
            plt.setp( axes[i,j].get_xticklabels(), visible=False)
            plt.setp( axes[i,j].get_yticklabels(), visible=False)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0, hspace=0)
    plt.suptitle('Kepler %d Quarter %d\n Kepler magnitude %f'%(star.kepid, qua, star.kic_kepmag))
    plt.savefig('%d-%d.png'%(star.kepid, qua))
    plt.clf()


