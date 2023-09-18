import os, sys
import numpy as np
import matplotlib.pyplot as pl
import h5py
from zfish.image import basic_ana as ba
from glob import glob
import tifffile as tf


class Mika():
    def __init__(self, imfolder_path = None, dirs = None):
        self.imfolder_path = imfolder_path
        
        self.dirs = dirs
        self.initialise()
        
        
    def initialise(self,):
       
        fname = 'cells0_clust.hdf5' ### for brain NMF of brain segements
    
        if self.dirs is not None:
            self.fig_path = self.dirs['plots'] + 'mikas_'
            self.cell_fpath = sorted(glob(self.dirs['mika'] + 'cells0_clean*'))[0]
            self.transforms_fpath = self.dirs['mika'] + 'transforms/transforms0.hdf5'
            self.clust_fpath = self.dirs['mika'] +  fname
            self.anatref_fpath = self.dirs['anat'] +  'anat_ref.npy'
            if os.path.isfile(self.anatref_fpath):
                self.anat_ref = np.load(self.anatref_fpath)
                print('loaded anatomical reference')
        else:
            self.fig_path = './mika_'
            self.cell_file = glob(self.imfolder_path + 'cells0_clean*')[0]
            print('no file to save analysis results...')
            self.clust_fpath = self.imfolder_path +  fname
        self.load_voldata()
        
        
    def load_celldata(self, offset = 1, camera_max = 900):
        f = h5py.File(self.cell_fpath, 'r')
        self.background = f['background'][()] # this is camera background
        self.fl = f['cell_timeseries'][()] # this is detrended time series
        self.fl = self.fl.clip(self.background.min(), camera_max)
        
        
        
        self.baseline = f['cell_baseline'][()] 
        self.baseline = self.baseline.clip(self.background.min(), None)
        
        #remove dead pixels
        maxn2 = self.fl.max(axis = 1)
        self.non_valid = np.argwhere(maxn2 == camera_max).flatten()
        print('# non valid: {}'.format(len(self.non_valid)))
#         self.fl[self.non_valid] = self.background.min()
#         self.baseline[self.non_valid] = self.background.min()


        self.df = self.fl-self.baseline
        threshold = np.percentile(self.df, .0005)
        print('threshold: {}'.format(threshold))
        self.df = self.df.clip(threshold, None)

        camera_offset = np.min(self.baseline)
        self.dff = self.df/(self.baseline - camera_offset + offset)
        f.close()
        print('loaded df')
        print('dimension: {0}'.format(self.df.shape))
        return self.dff


    def load_transforms(self, plot = True):
        f = h5py.File(self.transforms_fpath, 'r')
        self.transforms = f['transforms'][()]
        if plot:
            plotTransforms(self.transforms, self.fig_path)
            print('transforms loaded, plotted and saved')
        
    def make_segment_mask(self,):
        data_tn = np.random.uniform(0, 1, self.n)*256
        vol = self.fill_vol(data_tn)
        
        pl.figure(figsize = (12, 10))
        pl.title('segement map')
        pl.imshow(vol.max(0).T)
                  
        figpath = self.fig_path + 'segement_map.png'
        pl.savefig(figpath)
                  
        ds = 2
        anat_mkref = self.anat_ref[:,::ds,::ds]
        rgb = np.concatenate([vol.transpose([0,2,1])[:,None], anat_mkref[:,None]], axis = 1)
        path_anat = self.dirs['anat'] + 'anat_mkref'
        ext = '.tif'
        tf.imsave(path_anat + ext, data = rgb.astype('float32'), imagej = True)
        print('saved segment mask')
        
            
    def load_voldata(self):    
        f = h5py.File(self.cell_fpath, 'r')
        self.x = f['x'][()]
        self.y = f['y'][()]
        self.z = f['z'][()]
        self.t = f['t'][()]
        self.n = f['n'][()]   
        self.dims = (self.z, self.x, self.y)

        self.cell_x = f['cell_x'][()]         
        self.cell_y = f['cell_y'][()]         
        self.cell_z = f['cell_z'][()]        
        self.coords = np.array([self.cell_z, self.cell_x, self.cell_y])
        self.coords[self.coords ==-1] = 0
        
        self.linear_coords = np.ravel_multi_index(self.coords.reshape([3, -1]), dims = self.dims).reshape([self.n, -1])
        
        f.close()
        self.valid_cx = self.cell_x>0
        self.max_cellx = self.cell_x.shape[1]
        
        print('basic volume paramters loaded...\n')
        
    
    def fill_one_vol(self, data):
        vol = np.zeros([self.z, self.x, self.y])
        data = np.repeat(data[:,None], self.max_cellx, axis = 1)
        vol[self.cell_z[self.valid_cx].flatten()-1, \
            self.cell_x[self.valid_cx].flatten()-1, \
            self.cell_y[self.valid_cx].flatten()-1] = data[self.valid_cx]
        return vol
            
    def fill_vol(self, data_tn):
        if len(data_tn.shape) == 1:
            vol = self.fill_one_vol(data_tn)
        else:
            t, n = data_tn.shape
            vol = np.array([self.fill_one_vol(data_tn[i]) for i in range(t)])
        return vol
        
    def clean_up(self, data_nt, indices, post = 5, maxT = None):
        tmp = np.copy(data_nt)
        for i in indices:
            tmp[:, i:i+post] = data_nt[:,i-post*2:i-post].mean(axis = 1)[:,None]
        if maxT is not None:
            tmp = tmp[:,:maxT]
        return tmp

    def applyRasterMap(self, data_nt, k = 30, save = True, name = ''):
        raster, order = ba.applyRasterMap(data_nt, n_X = k)
        self.raster = raster
        self.order = order
        plotRaster(self.raster, self.fig_path + 'rastermap_' + name)
        if save:
            f = h5py.File(self.clust_fpath, 'a')
            keys = f.keys()
            key = name + '_order'
            if key in keys:
                del f[key]
                print('deleted key')
            f[key] = order
            f.close
        return raster, order
        
    def applyNMF(self, data_nt, k = 30, save = True, init = 'random'):
        from sklearn.decomposition import NMF
        nmf = NMF(n_components=k, init = init)
        w = nmf.fit_transform(data_nt.T)
        if save:
            f = h5py.File(self.clust_fpath, 'a')
            try:
                del f['W']
            except:
                pass
            try:
                del f['H']
            except:
                pass
            f['W'] = w
            f['H'] = nmf.components_
            f.close
            print('{0} clusters computed and saved'.format(k))
        self.w = w
        self.h = nmf.components_
        return w, nmf.components_
    

    
def plotTransforms(transforms, figpath):
    fig = pl.figure(figsize = (21, 10))
    t, w = transforms.shape
    for i in range(w):
        pl.plot(transforms[:,i], label = i, lw = 3)
    pl.legend()
    pl.xlabel('time [frames]')
    pl.ylabel('distance [pixels]')
    pl.title('ITK tranforms')
    pl.ylim([-8, 8])
    figpath = figpath + 'transforms.png'
    pl.savefig(figpath)


def plotRaster(raster, figpath):
    fig = pl.figure(figsize = (21, 14))
    pl.imshow(raster, aspect = .1)
    pl.clim([0,1])
    pl.savefig(figpath)