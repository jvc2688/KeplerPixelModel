import kplr
import numpy as np
import matplotlib.pyplot as plt

qua = 5

client = kplr.API()

# Find the target KOI.
koi = client.koi(282.02)

originStar = koi.star
originTpf = client.target_pixel_files(ktc_kepler_id=originStar.kepid, sci_data_quarter=qua)[0]

# Find potential targets by Kepler magnitude
starsOver = client.target_pixel_files(kic_kepmag=">=%f"%originStar.kic_kepmag, sci_data_quarter=qua, sort=("kic_kepmag", 1),ktc_target_type="LC", max_records=17)
starsUnder = client.target_pixel_files(kic_kepmag="<=%f"%originStar.kic_kepmag, sci_data_quarter=qua, sort=("kic_kepmag", -1),ktc_target_type="LC", max_records=17)

stars = {}
stars[originTpf.ktc_kepler_id] = originTpf

#Find 16 stars that are closest to the origin star in terms of Kepler magnitude
i=0
j=0
while len(stars) <17:
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

for key, tpf in stars.items():
# Get a list of light curve datasets.
    time, data = [], []

    with tpf.open() as f:
        hdu_data = f[1].data
        time = hdu_data["time"]
        data = hdu_data["flux"]

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
            axes[i,j].plot(time,data[0:td:1,i,j])
            plt.setp( axes[i,j].get_xticklabels(), visible=False)
            plt.setp( axes[i,j].get_yticklabels(), visible=False)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0, hspace=0)
    plt.suptitle('Kepler %d Quarter %d\n Kepler magnitude %f'%(tpf.ktc_kepler_id, qua, tpf.kic_kepmag))
    plt.savefig('%d-%d.png'%(tpf.ktc_kepler_id, qua))
    plt.clf()


