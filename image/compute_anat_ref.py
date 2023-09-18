import time
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import dask.array as da
from fish.image import zds
from glob import glob
from scipy.ndimage import median_filter, sobel, gaussian_filter
import os, sys


if __name__ == '__main__':

    base_dir = sys.argv[1]
    exp_num = sys.argv[2]
    print(base_dir)
    print('this is exp num: {0}'.format(exp_num))
    experiment = 'exp' + str(exp_num) + '/'

    dirs = fs.get_subfolders(base_dir + experiment)
    print(dirs.keys())
    imag_folder = sorted(glob(dirs['imag'] + '*'))[0] + '/'
    dset = zds.ZDS(imag_folder)
    fs_im = dset.metadata['volume_rate']
    print('%f fs_im', fs_im)

    data = dset.data
    try:
        dirs['anat']
        print('\nanat folder found')
    except:
        print('no folder found \ncreating...')
        dirs['anat'] = dirs['main'] + 'anat/'
        os.mkdir(dirs['anat'])

    anat_ref_file_path = dirs['anat'] + 'anat_ref.npy'
    if os.path.isfile(anat_ref_file_path):
        anat_ref = np.load(anat_ref_file_path)
        print('\nloaded anatomical reference')
    else:
        print('\nno anatomical reference found - \ncalculating...')
        anat_ref = data[data.shape[0]//2 + np.arange(-5,5)].compute(scheduler='threads').mean(0)
        np.save(dirs['anat'] + 'anat_ref.npy', anat_ref)
        tf.imsave(dirs['anat'] + 'anat_ref.tif', data = np.expand_dims(anat_ref.astype('float32'), axis = 1), imagej = True,\
              metadata={'axes': 'ZCYX'})
        print('anat reference saved')
