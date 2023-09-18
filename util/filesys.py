def timeit(t0):
    ''' function to print elapsed time since t0'''
    import time
    var = (time.time()-t0)/60
    print('time: %.3f mins' % var)

    
def save_as_tiff(X, path):
    import tifffile as tf
    ext = '.tif'
    if len(X.shape) == 2:
        data = X[None].astype('float32')
    if len(X.shape) == 3:
        data = X[:,None].astype('float32')
    if len(X.shape) == 4:
        data = X[:,:,None].astype('float32')
    tf.imsave(path + ext, data = data, imagej = True)
    print('data saved to:', path + ext)

def get_subfolders(folder, extension = '*', verbose = False, make_plot_folder = False, windows = False):
    from glob import glob
    import os
    ''' extract subfolders and paths of a main directory containing extension'''
    
    folder_paths = list(filter(lambda v: os.path.isdir(v), sorted(glob(folder+ extension))))
    if windows == True:  div = '\\'
    else: div = '/'
    folders = list(map(lambda v: v.split(div)[-1], folder_paths))
    dirs = {}
    dirs['main'] = folder
    for i in range(len(folders)):
        if os.path.isdir(folder_paths[i]):
            dirs[folders[i]] = folder_paths[i] + div
    if verbose:
        [print(k) for ind, k in enumerate(dirs.keys())]
    if make_plot_folder:
        try:
            os.chdir(dirs['plots'])
            print('relocated to plot direcctory')
        except:
            print(folder + 'plots')
            try:
                os.mkdir(folder + 'plots')
            except:
                print("couldn't create folder")
    return dirs


def create_meta(meta_dict, path):
    import xmltodict
    import xml.dom.minidom as minidom
    m = {}
    m['meta'] = meta_dict
    new_xml = xmltodict.unparse(m, pretty = True)
    with open(path, 'w') as file:
        file.write(new_xml)
        print('saved meta')

def read_meta(path):
    import xmltodict
    with open(path, 'r') as file:
        tmp = file.read()
    dd = xmltodict.parse(tmp)['meta']
    return dd
    