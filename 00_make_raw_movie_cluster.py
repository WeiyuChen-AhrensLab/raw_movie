import os, sys
sys.path.append('/groups/ahrens/home/chenw2/code/zfish/')
import time
import numpy as np
from glob import glob
from sys import getsizeof
import tifffile as tf
import matplotlib.pyplot as pl
import dask.array as da
from dask.distributed import Client
from zfish.image import vol as vo
from zfish.util import filesys as fs
from zfish.image import zds
from zfish.util.distributed import get_jobqueue_cluster
from zfish.util.distributed import get_dashboard_link
from scipy.ndimage import median_filter


def remove_anat(im, anat_ref):
    return im.astype('float32')-anat_ref


if __name__ == '__main__':

    # callcluster = True
    callcluster = False
    
    base_dir = sys.argv[1]
    exp_num = sys.argv[2]
    imag_num = int(sys.argv[3])
    plane_start = int(sys.argv[4])
    plane_end = int(sys.argv[5])
    print(base_dir)
    print('processing experiment number: {0}'.format(exp_num))
    experiment = 'exp' + str(exp_num) + '/'
    dirs = fs.get_subfolders(base_dir + experiment)
    if imag_num == 1: # if imag_num is 1:
        print(imag_num)
        imag_folder = sorted(glob(dirs['imag'] + '*'))[0] + '/'
        raw_folder = dirs['raw']
        
    if imag_num == 2: # if imag_num is 2
        print(imag_num)
        imag_folder = sorted(glob(dirs['imag2'] + '*'))[0] + '/'
        raw_folder = dirs['raw2']
    

    anat_ref_file_path = dirs['anat'] + 'anat_ref.npy'
    print('processing %s' % imag_folder)
    print('saving to: %s' % raw_folder)
    
    
    dset = zds.ZDS(imag_folder)
    fps = dset.metadata['volume_rate']
    
    ds_xy = 4#4
    ds_t = 1#1
    ds_z = 2#1
    print('downsampling in t, z, x/y by: %d, %d, %d' %(ds_t, ds_z, ds_xy))
    
    
    print('fps: {}'.format(fps))
    print('\noriginal dataset shape: {}'.format(dset.data.shape))

    # data = dset.data[::ds_t,::ds_z]
    
    data = dset.data[::ds_t,plane_start:plane_end]
    # data = dset.data[300:750,plane_start:plane_end]
    
    
    print("data type of data:", type(data))
    data_size_gb = data.nbytes / (1024**3)
    print("Size of data:", data_size_gb, "GB")

     

    # data = dset.data[:300,plane_start:plane_end]
    # data = dset.data[:100]
#     data_txf = data.map_blocks(remove_anat, anat_ref, dtype='float32').map_blocks(lambda v: median_filter(v, size=(1,1,3,3)))
    

    data_txf = data.map_blocks(lambda v: median_filter(v, size=(1,1,3,3)), dtype='float32')
    data_ds = da.coarsen(np.mean, data_txf, {2: ds_xy, 3: ds_xy})

    data_txf_size_gb = data_txf.nbytes / (1024**3)
    data_ds_size_gb = data_ds.nbytes / (1024**3)
    print("Size of data_txf:", data_txf_size_gb, "GB")
    print("Size of data_ds:", data_ds_size_gb, "GB")

    
    
    
#     rechunked = data_ds.rechunk(chunks=(-1, 'auto', 'auto','auto'))
#     data_dff = rechunked.map_blocks(lambda v: mydff(v, fs_im=fps), dtype='float32')
#     print(data_dff.shape)

    
    
    ### to call the cluster
    if callcluster:
        cluster = get_jobqueue_cluster()
        client = Client(cluster)
        get_dashboard_link(client)
        # cluster.start_workers(25)
        cluster.scale(100)
        
    

    x = data_ds.compute()

    ## calculate the size of x
    dtype_size = x.dtype.itemsize
    total_elements = x.size
    size_in_bytes = dtype_size * total_elements
    size_in_gb = size_in_bytes / 1024**3
    print(f"Size of x: {size_in_gb:.2f} GB")

    print('data computed!')
    name = 'data_ds' + str(ds_xy) + '_dst' + str(ds_t) + '_dz' + str(ds_z)
    fs.save_as_tiff(x, raw_folder + name) # works
    print('data saved!')

    t = x.shape[0]
    anat = np.median(x[t//2-10:t//2+10], axis = 0)
    # dfx = x-anat[None] # making df tiff
    
    fs.save_as_tiff(anat, raw_folder + name + '_anat')
    # fs.save_as_tiff(dfx, raw_folder + name + '_df') # making df tiff
    
    # xmax = x.max(1)
    # xdfmax = dfx.max(1)
    # fs.save_as_tiff(xdfmax, raw_folder + name + '_df_maxz')
    # fs.save_as_tiff(xmax, raw_folder + name + '_maxz')
    
    
    # print('making volume projections')
    # vol = vo.make_proj(x, rep = 10)
    # fs.save_as_tiff(vol, raw_folder + name + '_proj') # works
    # vol = vo.make_proj(dfx, rep = 10)
    # fs.save_as_tiff(vol, raw_folder + name + '_df_proj') # works
    
    
    
    print('saved data')
    print('saved movie')
    if callcluster:
        # cluster.stop_all_jobs()
        cluster.scale(0)
        print('workers stopped')
