import sys
sys.path.append('/groups/ahrens/home/ruttenv/code/zfish/')

def _tif_reader(tif_path, roi=None):
    from skimage.io import imread

    if roi is not None:
        raise NotImplementedError

    return imread(tif_path)


def _tif_writer(tif_path, image):
    from skimage.io import imsave

    imsave(tif_path, image)


def _stack_reader(stack_path, roi=None):
    from numpy import fromfile, memmap
    from os.path import sep, split
    from zfish.image.zds import get_metadata
    from glob import glob

    param_file = split(stack_path)[0] + sep + 'ch0*'
    param_file = glob(param_file)[0]
    dims = get_metadata(param_file)["dimensions"][::-1]

    if roi is not None:
        im = memmap(stack_path, dtype="uint16", shape=dims, mode="r")[roi]
    else:
        im = fromfile(stack_path, dtype="uint16").reshape(dims)

    return im


def _stack_writer(stack_path, image):
    raise NotImplementedError


def _klb_reader(klb_path, roi=None):
    from pyklb import readfull

    if roi is not None:
        raise NotImplementedError

    # pyklb whines if it doesn't get a python string
    return readfull(str(klb_path))


def _klb_writer(klb_path, image):
    from pyklb import writefull

    writefull(image, str(klb_path))


def _h5_reader(h5_path, roi=None):
    from h5py import File

    if roi is None:
        roi = slice(None)

    with File(h5_path, "r", libver="latest") as f:
        return f["default"][roi]


def _h5_writer(h5_path, data):
    from h5py import File
    from os import remove
    from os.path import exists

    if exists(h5_path):
        remove(h5_path)

    with File(h5_path, "w") as f:
        f.create_dataset(
            "default", data=data, compression="gzip", chunks=True, shuffle=True
        )
        f.close()


def _jp2_reader(jp2_path, roi=None):
    from glymur import Jp2k

    return Jp2k(jp2_path).read()[roi]


def _jp2_writer(jp2_path, image):
    raise NotImplementedError


readers = dict()
readers["stack"] = _stack_reader
readers["tif"] = _tif_reader
readers["klb"] = _klb_reader
readers["h5"] = _h5_reader
readers["jp2"] = _jp2_reader

writers = dict()
writers["stack"] = _stack_writer
writers["tif"] = _tif_writer
writers["klb"] = _klb_writer
writers["h5"] = _h5_writer
writers["jp2"] = _jp2_writer


def read_image(fname, roi=None, parallelism=1):
    """
    Load .stack, .tif, .klb, .h5, or jp2 data and return as a numpy array

    fname : string, path to image file

    roi : tuple of slice objects. For data in hdf5 format, passing an roi allows the rapid loading of a chunk of data.

    parallelism : int, defines the number of cores to use for loading multiple images. Set to -1 to use all cores.

    """

    from functools import partial
    from numpy import array, ndarray
    from multiprocessing import Pool, cpu_count

    if isinstance(fname, str):
        fmt = fname.split(".")[-1]
        reader = partial(readers[fmt], roi=roi)
        result = reader(fname)

    elif isinstance(fname, (tuple, list, ndarray)):
        fmt = fname[0].split(".")[-1]
        reader = partial(readers[fmt], roi=roi)

        if parallelism == 1:
            result = array([reader(f) for f in fname])

        else:
            if parallelism == -1:
                num_cores = cpu_count()
            else:
                num_cores = min(parallelism, cpu_count())

            with Pool(num_cores) as pool:
                result = array(pool.map(reader, fname))
    else:
        raise TypeError(
            "First argument must be string for a one file or (tuple, list, ndarray) for many files"
        )

    return result


def write_image(fname, data):
    """
    Write a numpy array as .stack, .tif, .klb, or .h5 file

    fname : string, path to image file
    
    data : numpy array to be saved to disk
    
    """
    # Get the file extension for this file, assuming it is the last continuous string after the last period
    fmt = fname.split(".")[-1]
    return writers[fmt](fname, data)


def to_dask(fnames):
    """
    Return a dask array constructued from an collection of ndarrays distributed across multiple files.

    fnames : iterable of sorted filenames
    """
    from dask.array import from_delayed, from_array, stack
    from h5py import File
    from dask.delayed import delayed
    from numpy import memmap

    fmt = fnames[0].split(".")[-1]
    s = read_image(fnames[0])

    def delf(fn):
        return File(fn, mode="r", libver="latest")["default"][:]

    if fmt == "h5":
        result = stack(
            [from_delayed(delayed(delf)(fn), s.shape, s.dtype) for fn in fnames]
        )
        return result

    elif fmt == "stack":
        from os.path import split, sep

        mems = [memmap(fn, dtype=s.dtype, shape=s.shape, mode="r") for fn in fnames]
        result = stack([from_array(mem, chunks=s.shape) for mem in mems])
        return result

    elif fmt in ("tif", "jp2"):
        rdr = delayed(read_image)
        result = stack(
            [from_delayed(rdr(fn), shape=s.shape, dtype=s.dtype) for fn in fnames]
        )
        return result

    else:
        raise NotImplementedError("{0} files not supported at this time".format(fmt))


def image_conversion(source_path, dest_fmt, wipe=False):
    """
    Convert image from one format to another, optionally erasing the source image

    image_path : string
        Path to image to be converted.
    wipe : bool
        If True, delete the source image after successful conversion

    """

    from numpy import array_equal
    from os import remove

    # the name of the file before format extension
    source_name = source_path.split(".")[0]

    dest_path = source_name + "." + dest_fmt
    source_image = read_image(source_path)
    write_image(dest_path, source_image)

    if wipe:
        check_image = read_image(dest_path)
        if array_equal(check_image, source_image):
            remove(source_path)
        else:
            print(
                "{0} and {1} differ... something went wrong!".format(
                    source_path, dest_path
                )
            )
