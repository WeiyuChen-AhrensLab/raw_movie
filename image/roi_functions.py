def getStackDims(inDir):
    """parse xml file to get dimension information of experiment.
    Returns [x,y,z] dimensions as a list of ints
    """
    import struct
    f = open(inDir+"Stack dimensions.log", "rb")

    s = f.read(12)
    dims = struct.unpack("<lll", s)

    return dims


def makeDiskInd(r,dims):
    import numpy as np
    import math

    array=np.zeros((2*r+1,2*r+1))
    ii=0.;
    while ii < (2*r+1):
        jj=0.;
        while jj < (2*r+1):
            if math.sqrt((ii-r)**2 + (jj-r)**2) <= r:
                array[int(ii),int(jj)]=1
            jj=jj+1
        ii=ii+1

    ind=np.nonzero(array);
    return (ind[0]-r,ind[1]-r)

def makeDiskInd2(r,dims):
    import numpy as np
    import math

    array=np.zeros((2*int(r)+1,2*int(r)+1))
    ii=0.;
    while ii < (2*r+1):
        jj=0.;
        while jj < (2*r+1):
            if math.sqrt((ii-r)**2 + (jj-r)**2) <= r:
                array[int(ii),int(jj)]=1
            jj=jj+1
        ii=ii+1

    ind=np.nonzero(array);
    return np.int32(array), np.int32((ind[0]-r)*dims[1]+(ind[1]-r)) ,np.int32((2*r+1,2*r+1))


def imNormalize(array,q):
    import numpy as np


    vmax=np.quantile(array.reshape(-1),q)
    vmin=array.min()
    array2=(array-vmin)/(vmax-vmin)
    array2=np.clip(array2,0,1)

    return array2


def imNormalize_color(array,q):
    import numpy as np

    vmax=np.quantile(array.reshape(-1),q)
    vmin=array.min()
    array2=(array-vmin)/(vmax-vmin)
    array2=np.clip(array2,0,1)
    array2=np.transpose(np.tile(array2,[3,1,1]),axes=(1,2,0))

    return array2



def loadNestedKeys(fname):
    import numpy as np

    cell_key_array=np.load(fname)

    cell_keylist=[]

    area=len(cell_key_array[0])
    for i in range(len(cell_key_array)):
        dk=[];
        tmp=cell_key_array[i]
        for j in range(area):
            dk.append((tmp[j][0],tmp[j][1],tmp[j][2]))

        cell_keylist.append(dk)

    return cell_keylist






def createCellSelectionGray(img):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import Slider, Button, RadioButtons

    # img = np.swapaxes(img,0,2)
    dims=np.asarray(img.shape)

    cimg=np.zeros((dims[0], dims[1], 3,dims[2]))
    i=0
    while i<dims[2]:
        tmp=imNormalize(img[:,:,i],0.9999).reshape((dims[0],dims[1],1))
        cimg[:,:,:,i]=np.tile(tmp,(1,1,3));
        i=i+1;

    zp=0

    fig = plt.figure()
    f1=plt.axes([0.05, 0.1, 0.9, 0.8]).imshow(cimg[:,:,:,zp]);
    title=plt.title('')
    fig.canvas.draw()

    axs=plt.axes([0.1, 0.92, 0.8, 0.05])
    slz=Slider(axs,'Z', 0, dims[2]-1, valinit=0)
    slz.valtext.set_visible(False)

    axm=plt.axes([0.05, 0.92, 0.05, 0.05])
    slm= Button(axm, '-')

    axp=plt.axes([0.9, 0.92, 0.05, 0.05])
    slp= Button(axp, '+')

    def update_panel(value):
        t=int(round(value))
        f1.set_data(cimg[:,:,:,t])
        title.set_text('plane'+ str(t+1))
        fig.canvas.draw()

    def plusbutton(val):
        t=int(round(slz.val))
        t=t+1
        if t>dims[2]-1:
            t=dims[2]-1
        slz.set_val(float(t))

        f1.set_data(cimg[:,:,:,t])
        title.set_text('plane'+ str(t+1))
        fig.canvas.draw()

    def minusbutton(val):
        t=int(round(slz.val))
        t=t-1
        if t<0:
            t=0
        slz.set_val(float(t))

        f1.set_data(cimg[:,:,:,t])
        title.set_text('plane'+ str(t+1))
        fig.canvas.draw()

    slz.on_changed(update_panel)
    slp.on_clicked(plusbutton)
    slm.on_clicked(minusbutton)

    title.set_text('plane'+ str(int(round(slz.val))+1))

    # plt.show()
    cnum=0
    xyzlist=list()

    # try:
    import keyboard
    # if keyboard.is_pressed('q'):  # if key 'q' is pressed
    #     print('You Pressed A Key!')
    #     on = False
    # if plt.get_fignums():
    # window(s) open


    while True:
        if plt.get_fignums():
            xy=(np.asarray(fig.ginput(1,show_clicks = True))[0]).astype(int)
            print('clicked')
            if xy.min()>=1 and xy[1]<dims[0] and xy[0]<dims[1]:
                cnum=cnum+1;
                zp=int(round(slz.val))
                xyzlist.append((xy[0],xy[1],zp))


                circind=makeDiskInd(5,dims)
                cimg[xy[1]+circind[0],xy[0]+circind[1],0,zp]=1;
                cimg[xy[1]+circind[0],xy[0]+circind[1],1,zp]=0;
                cimg[xy[1]+circind[0],xy[0]+circind[1],2,zp]=0;

                f1.set_data(cimg[:,:,:,zp]);
                fig.canvas.draw()
                print((xy[0],xy[1],zp))
        else:
            print('broken')
            break

    # except KeyboardInterrupt:
        # pass

    return xyzlist


def cell_recog(ave_stack,br_threshold,cont_threshold,cell_rad,grid):

    import numpy as np
    import scipy.ndimage.morphology as mo

    import imc


    if len(ave_stack.shape)==3:
        ave_stack=np.transpose(ave_stack,axes=(2,1,0));
        dim=np.int32(ave_stack.shape);
        zlen=dim[2]
    else:
        dim=np.int32(ave_stack.shape)
        zlen=1
        dim=np.append(dim,np.int32(1));


    # preparing elements for the array

    ave_rad=int(round(cell_rad/2))+1
    r=cell_rad*2;
    dimp=(dim[0]+r*2, dim[1]+r*2)

    onedisk, _, _                   = makeDiskInd2(3,dimp)
    rankdisk, rankinds, rankdim     = makeDiskInd2(r,dimp)
    avedisk, aveinds,_              = makeDiskInd2(ave_rad,dim)
    maxdisk, maxinds,_              = makeDiskInd2(cell_rad+(cell_rad/2),dim)


    oop=np.zeros(dimp,np.uint16)
    oop[np.ix_(np.arange(r,dimp[0]-r),np.arange(r,dimp[1]-r))]=1

    nonzero_ind=np.nonzero(oop)
    one_inds=np.int32(nonzero_ind[0]*dimp[1]+nonzero_ind[1])


    rank_ones=imc.imonesrank(dim,rankdisk,rankdim)
    recog_img_total=np.zeros((dim[0],dim[1],zlen,3));

    for z in range(zlen):

        # threshold image by contrast
        if zlen==1:
            ave=ave_stack
        else:
            ave=ave_stack[:,:,z]

        thre_img1=imc.threshold_contrast(ave, grid,cont_threshold)
        thre_img1=mo.binary_dilation(mo.binary_erosion(thre_img1,onedisk),onedisk)
        thre_img1[np.where(ave<=br_threshold)]=0

        # recognizing cells in the 1st round (rank image, local average, and local maxima)

        candidates1=np.int32(np.nonzero(thre_img1));
        candidate1_inds=candidates1[0]*dim[1]+candidates1[1];

        imrank1 = imc.imrank(ave,oop,one_inds,rank_ones,rankinds,candidate1_inds);

        aveimg1 = imc.local_average(imrank1,aveinds,candidate1_inds);
        maximg1 = imc.local_max(aveimg1,maxinds,candidate1_inds);

        cellinds1=np.where((maximg1[candidates1[0],candidates1[1]]>0) & (aveimg1[candidates1[0],candidates1[1]]>0.4));
        centers1=np.zeros(dim[:2],np.int32)
        centers1[candidates1[0][cellinds1],candidates1[1][cellinds1]]=1;
        centers1_mask=mo.binary_dilation(centers1,avedisk);

        allmask=centers1;

        # recognizing cells in the 2nd round (rank image, local average, and local maxima)

        thre_img2=np.ones(dim[:2],'int32')-mo.binary_dilation(centers1_mask,maxdisk);
        thre_img2=mo.binary_dilation(mo.binary_erosion(thre_img2,onedisk),onedisk);
        candidates2=np.int32(np.nonzero(thre_img1*thre_img2));
        candidate2_inds=candidates2[0]*dim[1]+candidates2[1]

        imrank2 = imc.imrank(ave,oop,one_inds,rank_ones,rankinds,candidate2_inds)
        aveimg2 = imc.local_average(imrank2,aveinds,candidate2_inds);
        maximg2 = imc.local_max(aveimg2,aveinds,candidate2_inds);

        cellinds2=np.where((maximg2[candidates2[0],candidates2[1]]>0) & (aveimg2[candidates2[0],candidates2[1]]>0.4));
        centers2=np.zeros(dim[:2],np.int32)
        centers2[candidates2[0][cellinds2],candidates2[1][cellinds2]]=1;

        allmask=allmask+centers2;

        allmask[:ave_rad+1,:]=0;
        allmask[-ave_rad-1:,:]=0;
        allmask[:,:ave_rad+1]=0;
        allmask[:,-ave_rad-1:]=0;

    # creating cell ROIs

        print("Plane "+ str(z))
        cell_centers=np.nonzero(allmask);
        cellnum=((np.array(cell_centers)).shape)[1]

        celldisk=makeDiskInd(ave_rad,ave_rad*2+1)
        disklen=(celldisk[0]).shape[0]

        if z==0:
            cell_info=np.zeros(cellnum,dtype=[('centery',np.int32),('centerx',np.int32),('centerz',np.int32),('indy',np.int32,disklen),('indx',np.int32,disklen),('indz',np.int32,disklen)])
            tot_cellnum=cellnum;

            for i in range(cellnum):
                cell_info[i]['centery']=cell_centers[0][i]
                cell_info[i]['centerx']=cell_centers[1][i]
                cell_info[i]['centerz']=z
                cell_info[i]['indy'][:]=cell_centers[0][i]+celldisk[0]
                cell_info[i]['indx'][:]=cell_centers[1][i]+celldisk[1]
                cell_info[i]['indz'][:]=np.ones((disklen,))*z
        else:
            cell_info_add=np.zeros(cellnum,dtype=[('centery',np.int32),('centerx',np.int32),('centerz',np.int32),('indy',np.int32,disklen),('indx',np.int32,disklen),('indz',np.int32,disklen)])

            for i in range(cellnum):
                cell_info_add[i]['centery']=cell_centers[0][i]
                cell_info_add[i]['centerx']=cell_centers[1][i]
                cell_info_add[i]['centerz']=z
                cell_info_add[i]['indy'][:]=cell_centers[0][i]+celldisk[0]
                cell_info_add[i]['indx'][:]=cell_centers[1][i]+celldisk[1]
                cell_info_add[i]['indz'][:]=np.ones((disklen,))*z

            cell_info=np.append(cell_info,cell_info_add)
            tot_cellnum=tot_cellnum+cellnum;

        recog_img=imNormalize_color(np.float32(ave),0.99);
        celllabel=np.zeros(dim)
        for i in np.arange(tot_cellnum-cellnum,tot_cellnum):
            recog_img[cell_info[i]['indy'],cell_info[i]['indx'],0]=1
            # recog_img[cell_info[i]['indy'],cell_info[i]['indx'],1]=0
            # recog_img[cell_info[i]['insdy'],cell_info[i]['indx'],2]=0
        recog_img_total[:,:,z,:]=recog_img;

    celllabel=np.zeros(dim)
    cellcenters=np.zeros(dim)
    cell_keylist=[];

    x,y,z = dim
    print(x,y,z)
    lims = (15,15,15,15)
    xmin,xmax,ymin,ymax = lims

    list_inds = initialise_listinds(lims, z, y, x)
    n_removed = 0

    for i in range(tot_cellnum):
        celllabel[cell_info[i]['indy'],cell_info[i]['indx'],cell_info[i]['indz']]=i
        cellcenters[cell_info[i]['centery'],cell_info[i]['centerx'],cell_info[i]['indz']]=i
        dk=[];
        for j in range(disklen):
            dk.append((cell_info[i]['indy'][j],cell_info[i]['indx'][j],cell_info[i]['indz']))
        cell_keylist.append(dk);
        cx = cell_info[i]['centerx']
        cy = cell_info[i]['centery']
        cz = cell_info[i]['centerz']
        list_inds['cz'].append(cz)
        list_inds['cy'].append(cy)
        list_inds['cx'].append(cx)
        n = len(cell_info[i]['indx'])
        if cy<y-xmax and cy>xmin and cx>ymin and cx<x-ymax:
            list_inds['val_inds1d'].append(True)
            list_inds['val_indsnd'].extend(list(np.ones(n).astype('bool')))
        else:
            n_removed +=1
            list_inds['val_inds1d'].append(False)
            list_inds['val_indsnd'].extend(list(np.zeros(n).astype('bool')))
        list_inds['z'].extend(list(cell_info[i]['indz']))
        list_inds['y'].extend(list(cell_info[i]['indy']))
        list_inds['x'].extend(list(cell_info[i]['indx']))
        list_inds['n'].append(n)
        inds3D = np.vstack([cell_info[i]['indz'],cell_info[i]['indx'],cell_info[i]['indy']])
        linds = np.ravel_multi_index(inds3D, (z,y,x))
        list_inds['linds'].extend(list(linds))
    print('removed {} cells'.format(n_removed))
    return cell_info, celllabel, cellcenters, recog_img_total, cell_keylist, list_inds

def initialise_listinds(lims, z, y, x):
    list_inds = {}
    list_inds['z'] = []
    list_inds['y'] = []
    list_inds['x'] = []
    list_inds['n'] = []
    list_inds['cz'] = []
    list_inds['cx'] = []
    list_inds['cy'] = []
    list_inds['linds'] = []
    list_inds['val_inds1d'] = []
    list_inds['val_indsnd'] = []
    list_inds['range'] = z,x,y
    list_inds['lims'] = lims
    return list_inds



def get_linear_indices(cell_info, anat_ref, lims = (15,15,15,15)):
    import numpy as np
    z,x,y = anat_ref.shape # careful the names are flipped to be consistent with Tk's indices of cell info
    xmin,xmax,ymin,ymax = lims
    list_inds = {}
    list_inds['z'] = []
    list_inds['y'] = []
    list_inds['x'] = []
    list_inds['n'] = []
    list_inds['cz'] = []
    list_inds['cx'] = []
    list_inds['cy'] = []
    list_inds['linds'] = []
    list_inds['val_inds1d'] = []
    list_inds['val_indsnd'] = []
    list_inds['range'] = z,x,y
    list_inds['lims'] = lims
    n_removed = 0
    for ind, c in enumerate(cell_info):
        if np.mod(1000, ind):
            print(ind)
        cx = c['centerx']
        cy = c['centery']
        cz = c['centerz']
        list_inds['cz'].append(cz)
        list_inds['cy'].append(cy)
        list_inds['cx'].append(cx)
        n = len(c['indx'])
        if cy<y-xmax and cy>xmin and cx>ymin and cx<x-ymax:
            list_inds['val_inds1d'].append(True)
            list_inds['val_indsnd'].extend(list(np.ones(n).astype('bool')))
        else:
#             print(cz,cx,cy)
            n_removed +=1
            list_inds['val_inds1d'].append(False)
            list_inds['val_indsnd'].extend(list(np.zeros(n).astype('bool')))
        list_inds['z'].extend(list(c['indz']))
        list_inds['y'].extend(list(c['indy']))
        list_inds['x'].extend(list(c['indx']))
        list_inds['n'].append(n)
        inds3D = np.vstack([c['indz'],c['indx'],c['indy']])
        linds = np.ravel_multi_index(inds3D, anat_ref.shape)
        list_inds['linds'].extend(list(linds))
    print('removed {} cells'.format(n_removed))
    return list_inds




def LassoSelection(cimg, opto = False):

    from matplotlib.widgets import Slider, Button,  LassoSelector
    from matplotlib.path import Path
    import matplotlib.pyplot as plt
    import numpy as np

    # cimg[:,:,2] =0
    # cimg[:,:,0] =0

    # cimg should be y,x,3,z

    dims = cimg[:,:,0].shape
    print('dims:')
    print(dims)
    opto_ROI = 1
    print(opto_ROI)
    ROI=np.zeros((dims[0],dims[1],dims[2]));
    ROI_color=np.zeros((dims[0],dims[1],3,dims[2]),dtype=np.float32)
    points=np.fliplr(np.array(np.nonzero(np.ones((dims[0],dims[1])))).T)

    zp=0

    fig = plt.figure()
    a1=plt.axes([0.05, 0.04, 0.4, 0.8])
    # f1=plt.imshow(cimg[:,:,:,zp]);
    f1=plt.imshow(cimg[:,:,:,zp],vmin = 100, vmax=np.percentile(cimg[:].squeeze(), 99.98),cmap='gray')
    
    title1=plt.title('')
    # plt.clim([np.min(cimg), 1])
    plt.colorbar()
    print(cimg[:,:,0,zp])
    f2=plt.axes([0.55, 0.05, 0.4, 0.8]).imshow(ROI_color[:,:,:,zp]);
    title2=plt.title('')

    fig.canvas.draw()

    axs=plt.axes([0.1, 0.92, 0.8, 0.05])
    slz=Slider(axs,'Z', 0, dims[2]-1, valinit=0)
    slz.valtext.set_visible(False)

    axm=plt.axes([0.05, 0.92, 0.05, 0.05])
    slm= Button(axm, '-')

    axp=plt.axes([0.9, 0.92, 0.05, 0.05])
    slp= Button(axp, '+')
    points_poly=[]



    def update_panel(value):
        t=int(round(value))
        f1.set_data(cimg[:,:,:,t])
        title1.set_text('plane'+ str(t+1))
        f2.set_data(ROI_color[:,:,:,t])
        title2.set_text('plane'+ str(t+1))
        fig.canvas.draw()

    def plusbutton(val):
        t=int(round(slz.val))
        t=t+1
        if t>dims[2]-1:
            t=dims[2]-1
        slz.set_val(float(t))

        f1.set_data(cimg[:,:,:,t])
        title1.set_text('plane'+ str(t+1))
        f2.set_data(ROI_color[:,:,:,t])
        title2.set_text('plane'+ str(t+1))
        fig.canvas.draw()

    def minusbutton(val):
        t=int(round(slz.val))
        t=t-1
        if t<0:
            t=0
        slz.set_val(float(t))

        f1.set_data(cimg[:,:,:,t])
        title1.set_text('plane'+ str(t+1))
        f2.set_data(ROI_color[:,:,:,t])
        title2.set_text('plane'+ str(t+1))
        fig.canvas.draw()

    def onselectf(verts):
        t=int(round(slz.val))
        path=Path(verts)
        points_poly=np.where(path.contains_points(points))[0]

        tmp=np.zeros((dims[0],dims[1],1))
        n = np.max(ROI)+ 1
        if opto ==False:
            tmp[points[points_poly,1],points[points_poly,0]]=1;
        else:
            tmp[points[points_poly,1],points[points_poly,0]]=n;
            print(n)
            n = np.max(tmp)+ 1


        ROI[:,:,t]=ROI[:,:,t]+np.reshape(tmp,(dims[0],dims[1]))
        ROI_color[:,:,:,t]=ROI_color[:,:,:,t]+np.tile(tmp,(1,1,3))
        n = np.max(ROI)+ 1
        f2.set_data(ROI_color[:,:,:,t]);
        fig.canvas.draw()




    lasso=LassoSelector(a1,onselect=onselectf)
    slz.on_changed(update_panel)
    slp.on_clicked(plusbutton)
    slm.on_clicked(minusbutton)
    title1.set_text('plane'+ str(int(round(slz.val))+1))
    title2.set_text('plane'+ str(int(round(slz.val))+1))


    def accept(event):
        if event.key == "enter":
            print("ROIs have been selected")
            lasso.disconnect_events() 
            # ax.set_title("")
            # fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    plt.show()

    # ROI[np.where(ROI>1)]=1;
    ROI = np.swapaxes(np.swapaxes(ROI,0,1),0,2)
    print(ROI.max())
    print(ROI.shape)
    return ROI


def calcMovBaseline(trace,span,fraction):

    import numpy as np
    from multiprocessing import Process, Queue

    trace=trace.T

    ncell=trace.shape[0]
    totlen=trace.shape[1]
    halfspan=np.floor(span/2);

    moves=np.arange(-halfspan,halfspan+1)
    calc_inds=halfspan+np.arange(0,totlen)

    tmp=np.zeros((totlen+span+1,1))
    tmp[:totlen,0]=np.arange(1,totlen+1)
    tmp=np.tile(tmp,(len(moves),1))
    move_matrix=np.reshape(tmp[0:-span-1,0],(len(moves),totlen+span),order='C')
    move_matrix=move_matrix[:,np.int32(calc_inds)]

    dim=move_matrix.shape

    avefraction=np.float32(np.ceil(np.sum(move_matrix>0,axis=0)*fraction))

    one_matrix=np.zeros(move_matrix.shape);
    for i in range(totlen):
        one_matrix[0:avefraction[i],i]=1


    transfer_pos=np.where(move_matrix.T>0)
    transfer_inds=np.int32(move_matrix[transfer_pos[1],transfer_pos[0]]-1)
    calc_matrix=np.ones(dim,dtype=np.float32)*70000;

    baseline=np.zeros((ncell,totlen));


    def pnormalize(keys,trace_chunk,out_q):

        nrep=trace_chunk.shape[0]
        outdict={}
        for j in range(nrep):
            tmp=trace_chunk[j,:]
            calc_matrix[transfer_pos[1],transfer_pos[0]]=tmp[transfer_inds]
            outdict[keys[j]]=np.sum(one_matrix*np.sort(calc_matrix, axis=0),axis=0)/avefraction

        out_q.put(outdict)

    nprocs=12
    out_q=Queue()
    chunksize=int(np.ceil(float(ncell)/float(nprocs)))
    procs=[]

    for i in range(nprocs):
        p=Process(target=pnormalize,args=(range(chunksize*i,chunksize*(i+1)),trace[chunksize*i:chunksize*(i+1),:],out_q))
        procs.append(p)
        p.start()

    resultdict={}
    for i in range(nprocs):
        resultdict.update(out_q.get())

    for p in procs:
        p.join()

    for i in range(ncell):
        baseline[i,:]=resultdict[i].T;

    return baseline.T

def write_ablation_file(rawDir,procDir,cell_info,ex_inds2):

    import xml.etree.ElementTree as ET
    import numpy as np
    tree=ET.parse(rawDir+'ch0.xml')
    doc=tree.getroot()
    kk=doc.findall('info')
    xOffset=0;
    yOffset=0;
    zStep=0;
    for i in range(len(kk)):
        jj=kk[i].items()[0]
        if jj[0]=='camera_roi':
            xOffset=np.int32(jj[1].split('_')[0])
            yOffset=np.int32(jj[1].split('_')[2])

        elif jj[0]=='z_step':
            zStep=np.float32(jj[1])

    print (yOffset,xOffset,zStep)


    prewait=2000    # millisecond
    power=80        # % of the full pwoer
    duration=1      # millisecond
    zStart=0        # micron

    outpath = procDir+ 'EPfile.txt'

    print(outpath)
    f = open(outpath, 'w')
    for i in range(len(ex_inds2)):
        cnum=ex_inds2[i]
        f.write('ENTRY_START\n')
        f.write('ABLATION(OFF,200.0)\n')
        f.write('PRE_WAIT(' + str(prewait)+ ')\n')
        f.write('PRE_TRIGGER(OFF,5000,CONTINUE)\n')
        f.write('COORDINATES('+str(yOffset+cell_info[cnum]['centery'])+','+str(xOffset + cell_info[cnum]['centerx'])+','+str(zStart + cell_info[cnum]['slice']*zStep)+')\n')
        f.write('SCAN_TYPE(POINT)\n')
        f.write('POWER(' + str(power) + ')\n')
        f.write('DURATION(' +str(duration)+')\n')
        f.write('POST_TRIGGER(OFF,0, CONTINUE)\n')
        f.write('CAMERA_TRIGGER\n')
        f.write('ENTRY_END\n\n')
    f.close()


def calculate_anova_p(cell_resp,timelist,trials):

    import numpy as np
    from multiprocessing import Process, Queue
    from scipy.stats import f_oneway

    timelist2=np.insert(np.cumsum(timelist),0,0)

    ncell = cell_resp.shape[1]
    triallen=timelist2[-1]
    nrep=len(trials)


    anova_matrix=np.zeros((nrep,len(timelist),ncell))

    for t in range(len(trials)):
        for p in range(len(timelist)):
            baset=(trials[t]*triallen)+timelist2[p+1];
            anova_matrix[t,p,:] = np.reshape(np.mean(cell_resp[baset-5:baset,:],axis=0),(1,1,ncell))


    def anova_p(keys,anova_chunk,out_q):

        ncell2=anova_chunk.shape[2]
        outdict={}
        for j in range(ncell2):
            _,outdict[keys[j]]=f_oneway(np.squeeze(anova_chunk[:,0,j]),np.squeeze(anova_chunk[:,1,j]),np.squeeze(anova_chunk[:,2,j]),
            np.squeeze(anova_chunk[:,3,j]),np.squeeze(anova_chunk[:,4,j]))

        out_q.put(outdict)

    nprocs=12
    out_q=Queue()
    chunksize=int(np.ceil(float(ncell)/float(nprocs)))
    resp_matrix2=np.zeros((ncell,))
    procs=[]

    for i in range(nprocs):
        chunkrange=range(chunksize*i,np.minimum(chunksize*(i+1),ncell))
        p=Process(target=anova_p,args=(chunkrange,anova_matrix[:,:,chunkrange],out_q))
        procs.append(p)
        p.start()

    resultdict={}
    for i in range(nprocs):
        resultdict.update(out_q.get())

    for p in procs:
        p.join()

    for i in range(ncell):
        resp_matrix2[i]=resultdict[i];

    return resp_matrix2
