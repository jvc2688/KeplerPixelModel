import kplr
import numpy as np
import matplotlib.pyplot as plt

qua = 5

client = kplr.API()

koi = client.koi(282.02)

originStar = koi.star
#find the tpfs around the origin star and get the nearest 6 targets for specific quarter
starsNear = client.target_pixel_files(ra=originStar.kic_degree_ra, dec=originStar.kic_dec, radius=5, sci_data_quarter=5, sort=("ang_sep", 1),ktc_target_type="LC", max_records=6)

for tpf in starsNear:
    # Get a list of light curve datasets.
    time, flux = [], []
    with tpf.open() as f:
        hdu_data = f[1].data
        time = hdu_data["time"]
        flux = hdu_data["flux"]
    
    flux = np.nan_to_num(flux)
    flux = np.ma.masked_equal(flux,0)
    
    shape = flux.shape
    td = shape[0]
    x = shape[1]
    y = shape[2]

    # Plot the data
    f, axes = plt.subplots(x, y)
    
    for i in range(0,x):
        for j in range(0,y):
            axes[i,j].plot(time,flux[0:td:1,i,j])
            plt.setp( axes[i,j].get_xticklabels(), visible=False)
            plt.setp( axes[i,j].get_yticklabels(), visible=False)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0, hspace=0)
    plt.suptitle('Kepler %d Quarter %d\n RA %f DEC %f Separation from origin %f'%(tpf.ktc_kepler_id, qua, tpf.sci_ra, tpf.sci_dec, tpf.angular_separation))
    plt.savefig('%d-%d.png'%(tpf.ktc_kepler_id, qua))
    plt.clf()