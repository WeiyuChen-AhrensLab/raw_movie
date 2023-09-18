import numpy as np

def fill_vol_index(indices, list_inds, weights = None, rep = 8):
    ncells = int(len(list_inds['linds'])/list_inds['n'][0])
    data_nt = np.zeros([ncells,1])
    if weights is not None:
        data_nt[indices] = weights.squeeze()[:,None]
    else:
        data_nt[indices] = 1
    vol = fill_vol(data_nt, list_inds)
    proj = make_proj(vol, rep = rep)
    return vol, proj


def fill_vol(data_nt, list_inds_zyx):
    import numpy as np
    zd,yd,xd = list_inds_zyx['range']
    n = list_inds_zyx['n'][0]
   
    if len(data_nt.shape)==1:
        data_nt = np.repeat(data_nt.reshape([-1,1]), n,axis = 1)
        N = data_nt.shape[0]
        T=1
    else:
        N,T = data_nt.shape
        data_nt = np.repeat(np.expand_dims(data_nt,axis = 1), n ,axis = 1).reshape([data_nt.shape[0]*n, -1])
        
    if T==1:
        vol = np.zeros([zd, yd, xd]).squeeze().flatten()
        vol[list_inds_zyx['linds']] = data_nt.squeeze().flatten()
        vol = vol.reshape([zd, yd, xd])
    if T>1:
        vol = np.zeros([zd, yd, xd, T]).reshape([-1,T])
        vol[list_inds_zyx['linds'],:] = data_nt
        vol = np.transpose(vol.reshape([zd, yd, xd,T]),[3,0,1,2])
        print(vol.shape)
        
    return vol




def make_proj(hyp_Vol,txyz = [0,3,2,1], proj_func = np.max, proj_func2 = np.min,  tag = 0,rep = 4):
        """
        Returns projected volume, with
        """
        
        import numpy as np
        
        if len(hyp_Vol.shape) ==3:
            hyp_Vol = np.expand_dims(hyp_Vol, 0)
        hyp_Vol = hyp_Vol.transpose([txyz[0], txyz[3],txyz[2],txyz[1]])
        vol = hyp_Vol.shape
        x = vol[-1]
        y = vol[-2]
        z = vol[-3]


        hyp_Volxy = proj_func(hyp_Vol, axis = 1) + proj_func2(hyp_Vol, axis = 1)
        hyp_Volxz = proj_func(hyp_Vol, axis = 2) + proj_func2(hyp_Vol, axis = 2)
        hyp_Volyz = proj_func(hyp_Vol, axis = 3) + proj_func2(hyp_Vol, axis = 3)

        #expand
        hyp_Volyz_tmp = np.zeros([hyp_Volyz.shape[0], hyp_Volyz.shape[1]*rep, hyp_Volyz.shape[2]])
        for i in range(0, hyp_Volyz.shape[1]):
            ind = i*rep
            for r in range(0,rep):
                hyp_Volyz_tmp[:,ind+r,:] = hyp_Volyz[:,i,:]

        hyp_Volxz_tmp = np.zeros([hyp_Volxz.shape[0], hyp_Volxz.shape[1]*rep, hyp_Volxz.shape[2]])
        for i in range(0, hyp_Volxz.shape[1]):
            ind = i*rep
            for r in range(0,rep):
                hyp_Volxz_tmp[:,ind+r,:] = hyp_Volxz[:,i,:]

        VV = np.zeros([hyp_Vol.shape[0], y+z*rep+1, x+z*rep])
        if not tag ==0 :
            for ind in range(0,tag.shape[0]):
                VV[ind,y+1:y+z*rep+1,x:x+z*rep] = tag[ind]
        VV[:,0:y,0:x] = hyp_Volxy[:,:,:]
        # VV[:,y+1:y+z*rep+1,0:x] = hyp_Volxz_tmp[:,::-1,:]
#         VV[:,y+1:y+z*rep+1,0:x] = hyp_Volxz_tmp[:,::-1,:]
        VV[:,y+1:y+z*rep+1,0:x] = hyp_Volxz_tmp[:,::,:] # to flip bottom
        VV[:,0:y,x:x+z*rep] = np.transpose(hyp_Volyz_tmp[:,:,:],[0,2,1])[:,:,::-1]# to fpli side
        return VV.squeeze()

    
    
def local_corr(images, offset=[0,1,1]):
    """
    Correlate each image in a distributed set of images with a shifted copy of itself. Returns an rdd of
    correlation coefficients.

    images : thunder.images object
    offset : the shift, in pixels, to apply to each image before correlation

    """
    from scipy.ndimage.interpolation import shift

    def correlate_signals(s1, s2):
        from numpy import corrcoef
        return corrcoef(s1, s2)[0][1]

    images_shifted = images.map(lambda v: shift(v.astype('float32'), offset, mode='reflect')).astype('float16')
    joined = images.toseries().tordd().join(images_shifted.toseries().tordd())

    return joined.mapValues(lambda v: correlate_signals(v[0], v[1]))


def baseline(data, window, percentile, downsample=1, axis=-1):
    """
    Get the baseline of a numpy array using a windowed percentile filter with optional downsampling

    data : Numpy array
        Data from which baseline is calculated

    window : int
        Window size for baseline estimation. If downsampling is used, window shrinks proportionally

    percentile : int
        Percentile of data used as baseline

    downsample : int
        Rate of downsampling used before estimating baseline. Defaults to 1 (no downsampling).

    axis : int
        For ndarrays, this specifies the axis to estimate baseline along. Default is -1.

    """
    from scipy.ndimage.filters import percentile_filter
    from scipy.interpolate import interp1d
    from numpy import ones

    size = ones(data.ndim, dtype='int')
    size[axis] *= window//downsample

    slices = [slice(None)] * data.ndim
    slices[axis] = slice(0, None, downsample)

    if downsample == 1:
        bl = percentile_filter(data, percentile=percentile, size=size)

    else:
        data_ds = data[slices]
        baseline_ds = percentile_filter(data_ds, percentile=percentile, size=size)
        interper = interp1d(range(0, data.shape[axis], downsample), baseline_ds, axis=axis, fill_value='extrapolate')
        bl = interper(range(data.shape[axis]))

    return bl


def dff(data, window, percentile, baseline_offset, downsample=1, axis=-1, return_bl = False):
    """
    Estimate normalized change in fluorescence (dff) with the option to estimate baseline on downsampled data.
    Returns a vector with the same size as the input.

    If downsampling is required, the input data will be downsampled before baseline
    is estimated with a percentile filter. The baseline is then linearly interpolated to match the size of data.

    data : Numpy array
        Data to be processed

    window : int
        Window size for baseline estimation. If downsampling is used, window will shrink proportionally

    percentile : int
        Percentile of data used as baseline

    baseline_offset : float or int
        Value added to baseline before normalization, to prevent division-by-zero issues.

    downsample : int
        Rate of downsampling used before estimating baseline. Defaults to 1 (no downsampling).

    axis : int
        For ndarrays, this specifies the axis to estimate baseline along. Default is -1.
    """

    bl = baseline(data, window, percentile, downsample=downsample, axis=axis)
    if return_bl:
        print('returning baseline')
        return (data - bl) / (bl + baseline_offset), bl
    else:
        return (data - bl) / (bl + baseline_offset)


def filter_flat(vol, mask):
    """
    Flatten an array and return a list of the elements at positions where the binary mask is True.

    vol : ndarray
    mask : binary ndarray or function. If a function, mask must take an ndarray as an argument and return a
    binary mask.
    """
    vol_flat = vol.ravel()

    # if mask is a function, call it on vol to make the mask
    if hasattr(mask, '__call__'):
        mask_flat = mask(vol).ravel()
    else:
        mask_flat = mask.ravel()

    return vol_flat[mask_flat]


def unfilter_flat(vec, mask):
    """
    Reverse the effect of filter_flat by taking a 1d ndarray and assigning each value to a position in an ndarray

    vec : 1-dimensional ndarray
    mask : binary ndarray
    """

    from numpy import zeros
    """
    Fill a binary mask with the values in vec
    """
    mask_flat = mask.ravel()
    vol = zeros(mask.shape).ravel()
    vol[mask_flat == True] = vec

    return vol.reshape(mask.shape)


def kvp_to_array(dims, data, ind=0, baseline=0):
    """ Convert a list of key-value pairs to a volume.

    :param dims: dimensions of the volume to fill with values
    :param data: list of iterables with 2 values, 2-tuples or 2-item lists.
    :param ind: if the value in each key-value pair is itself an iterable, ind specifies which element to use
    :param baseline: fill value for empty spots in the volume
    :return: vol, an ndarray
    """
    from numpy import zeros, array

    vol = zeros(dims, dtype=data[0][1].dtype) + array(baseline).astype(data[0][1].dtype)

    for k, v in data:
        # check if data contains a single value or an iterable
        if hasattr(v, '__iter__'):
            vol[k] = v[ind]
        else:
            vol[k] = v
    return vol


def sub_proj(im, ax, func, chop=16):
    """
    Project a volume in chunks along an axis.

    im : numpy array, data to be projected

    ax : int, axis to project along

    func : function that takes an axis as an argument, e.g. np.max

    chop : int, number of projections to generate

    """
    from numpy import rollaxis

    extra = im.shape[ax] % chop
    montage_dims = list(im.shape)
    montage_dims[ax] //= chop
    montage_dims.insert(ax, chop)

    slices_crop = [slice(None) for x in im.shape]
    slices_crop[ax] = slice(0, extra + 1, 1)

    # remove trailing data by projecting it down to a single plane
    im[slices_crop] = im[slices_crop].max(ax, keepdims=True)

    slices_keep = [slice(None) for x in im.shape]
    slices_keep[ax] = slice(extra, None)

    im_proj = func(im[slices_keep].reshape(montage_dims), axis=ax + 1)
    # stick the axis of projections in the front
    im_proj = rollaxis(im_proj, ax, 0)

    return im_proj


def redim(array, ndim, shape=None):
    """
    Add or remove trailing dimensions from an array by reshaping. Useful for turning N-dimensional data into the 2D
    shape required for matrix factorization / clustering, and for reversing this transformation. Returns a view of
    the input array.

    array : numpy array with 2 or more dimensions.

    ndim : int, desired number of dimensions when contracting dimensions.

    shape : tuple, desired shape when expanding dimensions.

    """

    from numpy import prod

    result = None
    if (ndim > array.ndim) and (shape is None):
        raise ValueError('Cannot expand dimensions without supplying a shape argument')

    if ndim < array.ndim:
        new_shape = (*array.shape[:(ndim - 1)], prod(array.shape[(ndim - 1):]))
        result = array.reshape(new_shape)

    elif ndim > array.ndim:
        new_shape = shape
        result = array.reshape(new_shape)

    elif ndim == array.ndim:
        new_shape = array.shape
        result = array.reshape(new_shape)

    return result


class InterpArray:

    def __init__(self, x, y, full_shape, interpolation_axis):
        """
        Create an array of numeric values representing a downsampled version of a larger array. This object is
        initialized with ``x`` and ``y`` values which define downsampled data used for interpolation and the points those
        values correspond to in the dimensions of the full, original array. Indexing this object with an index that is
        not an element of y returns values interpolated from x. Indexing is only supported on the first axis.

        x: list or numpy array of sorted integers matching the length of x. These values represent the indices at which
        the values of x (along its first axis) were sampled.

        y: numpy or dask array. Length of the first axis must match length of x. These are the observed values that will
        be interpolated to yield values when indexing the InterpArray

        full_shape: the full shape of the data that generated x. If x was generated by some larger numpy array z,
        then x == z[y], and full_shape == z.shape
        """
        from numpy import array

        self.x = array(x)
        self.full_shape = full_shape
        self.y = y
        self.interpolation_axis = interpolation_axis

    def __repr__(self):
        return f'An interpolated array with size {self.full_shape} sampled' \
               f' at {self.x} along axis {self.interpolation_axis}'

    def _instantiate_slice_indices(self, slc, axis):
        """
        Replace a slice object ``slc`` along ``axis`` with a numpy array that spans the same range
        """
        from numpy import arange
        return arange(self.full_shape[axis])[slc]

    def _get_interpolated_value(self, idx):
        from numpy import where, diff, array

        ipax = self.interpolation_axis

        try:
            idx_ = list(idx)
        except TypeError:
            idx_ = [idx]

        idx_interp = idx_[ipax]

        # if hasattr(idx_interp, '__getitem__'):
        # make dists a column vector so we can potentially subtract a row vector
        dists = self.x.reshape(-1, 1) - idx_interp
        result = []

        for d_ in dists.T:
            idx_inner = idx_
            ds_ind_pre = where(d_ >= 0)[0][0] - 1

            if ds_ind_pre == -1:
                idx_inner[ipax] = 0
                result.append(self.y[tuple(idx_inner)])
                continue

            interval = diff(self.x)[ds_ind_pre]
            coeffs = array([interval - abs(d_[ds_ind_pre]), abs(d_[ds_ind_pre])]) / interval

            idx_inner[ipax] = slice(ds_ind_pre, ds_ind_pre + 2)

            # non-slice indices (ints) will drop an axis
            keep_axes = [ind for ind in range(len(idx_inner)) if isinstance(idx_inner[ind], slice)]

            y_ = self.y[tuple(idx_inner)]

            new_coeff_shape = [1] * y_.ndim
            new_coeff_shape[keep_axes.index(ipax)] = coeffs.size

            # perform linear interpolation
            result.append((coeffs.reshape(new_coeff_shape) * y_).sum(keep_axes.index(ipax)))

        return self._concat_arrays(result)

    def _concat_arrays(self, input):
        from numpy import array
        from dask.array import stack
        from dask.array.core import Array as DaskArray
        from numpy import ndarray as NPArray

        if len(input) == 1:
            return input[0]
        else:
            if isinstance(input[0], DaskArray):
                return stack(input)
            elif isinstance(input[0], NPArray):
                return array(input)
            else:
                return input

    def __getitem__(self, idx):
        ipax = self.interpolation_axis

        if not isinstance(idx, tuple):
            # instantiate slice and put in tuple with implicit slices
            full_idx = [slice(None)] * len(self.full_shape)
            full_idx[ipax] = self._instantiate_slice_indices(idx, ipax)
            idx = tuple(full_idx)
        else:
            # instantiate slice that is already in a tuple
            tmp_idx = list(idx)
            interp_idx = tmp_idx[ipax]
            tmp_idx[ipax] = self._instantiate_slice_indices(interp_idx, ipax)
            idx = tuple(tmp_idx)

        result = self._get_interpolated_value(idx)

        return result
