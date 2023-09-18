import numpy as np
from glob import glob
from zfish.image import vol as vo
import matplotlib.pyplot as pl

def get_trinds(trdict, pre, post, LS_N = np.inf):
    trind = {}
    for ind, key in enumerate(trdict.keys()):
        print('trial: ' + str(ind))
        trind[key] = {}
        
        starts = trdict[key]['starts']
        ends = trdict[key]['ends']
        valid_trials = ((starts + post)<LS_N)*((starts - pre)>10)
        nonvalid = np.sum(np.array(valid_trials) == False)
        print('\n#  of non valid trials: %d' % nonvalid)
        starts = starts[valid_trials]
        ends = ends[valid_trials]
       
        trind[key]['trig_bystart'] = np.hstack([np.arange(st-pre, st+post) for st in starts])
        trind[key]['trig_byend'] = np.hstack([np.arange(st-pre, st+post) for st in ends])
    return trind


def makeTrialDict(trials):
    tr_dicts = {}
    for ind, i in enumerate(trials):
        tr_dicts[ind] = {}
        tr_dicts[ind]['starts'] = np.array([j[0] for j in i])
        tr_dicts[ind]['ends'] = np.array([j[1] for j in i])
    return tr_dicts


def applyPCA_nt(X_features_trials, num):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=num)
    pca.fit(X_features_trials.T)
    X_projected = pca.components_.dot(X_features_trials)
    return X_projected, pca.components_, pca


def applyRasterMap(data_nt, n_X = 30):
    import numpy as np
    from rastermap import Rastermap
    model = Rastermap(n_components=1, n_X=30, nPC=200, init='pca')
    model = model.fit(data_nt)
    isort_full = np.argsort(model.embedding[:,0])
    return data_nt[isort_full], isort_full


def applyNMF_tn(X_trials_features,num = 10):
    from sklearn.decomposition import NMF
    model = NMF(n_components=num, init='random', random_state=0)
    W = model.fit_transform(X_trials_features)
    H = model.components_
    return model, W, H



    
def plot_nmf_maps(weights, hact, inds, list_inds, rep = 10, title = '', path = None, fps = 1, units = 'frames'):
    vol, proj = vo.fill_vol_index(inds, list_inds, weights, rep = rep)
    fig = pl.figure(figsize = (20, 20), constrained_layout=False)
    gs1 = fig.add_gridspec(nrows=4, ncols=1, hspace = .5)
    ax1 = fig.add_subplot(gs1[1:, :])
    ax2 = fig.add_subplot(gs1[0,:])
    ax1.imshow(proj, aspect = proj.shape[1]/proj.shape[0]/1.5)
    xax = np.arange(len(hact))/fps
    ax2.plot(xax, hact)
    ax2.set_xlabel('time - ' + units)
    pl.suptitle(title)
    if path is not None:
        pl.savefig(path)
