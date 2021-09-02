## The SOM code

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
#import tensorflow as tf
from matplotlib import colors
import popsom
import pandas as pd
import colorsys

import h5py as h5
import sys, os

from plot2d_var import read_var
from plot2d_var import Conf
from plot2d_var import imshow


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans

from sklearn import metrics

import argparse

parser = argparse.ArgumentParser(description='popsom code')
parser.add_argument('--xdim', type=int, dest='xdim', default=10, help='an integer for the accumulator')
parser.add_argument('--ydim', type=int, dest='ydim', default=10, help='an integer for the accumulator')
parser.add_argument('--alpha', type=float, dest='alpha', default=0.5, help='an float for the accumulator')
parser.add_argument('--train', type=int, dest='train', default=10000, help='an integer for the accumulator')

args = parser.parse_args()

def neighbors(arr,x,y,n=3):
        ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
        arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
        return arr[:n,:n]

if __name__ == "__main__":

        # set-up plotting
        #plt.fig = plt.figure(1, figsize=(4,3.5), dpi=200)
        #fig = plt.figure(1, figsize=(6,6), dpi=300)

        plt.rc('font',  family='sans-serif')
        #plt.rc('text',  usetex=True)
        plt.rc('xtick', labelsize=5)
        plt.rc('ytick', labelsize=5)
        plt.rc('axes',  labelsize=5)


        scaler = StandardScaler()

        #--------------------------------------------------
        #read data
        #--------------------------------------------------
        #build feature matrix
        feature_list = [
                        'rho',
                        'bx',
                        'by',
                        'bz',
                        'ex',
                        'ey',
                        'ez',
                        'jx',
                        'jy',
                        'jz',
                        ]

        conf = Conf()

        fdir = 's6/'

        print("plotting {}".format(fdir))

        fname_F = "flds"

        xmin = 0.0
        ymin = 0.0
        xmax = 1.0
        ymax = 1.0

        #--------------------------------------------------
        def read_h5_data_size(fdir, lap):
            conf.fields_file   = fdir + 'flds_'+str(lap)+'.h5'
            f5F = h5.File(conf.fields_file,'r')

            #read one and check size
            val0 = read_var(f5F, feature_list[0])
            nx,ny = np.shape(val0)

            return nx,ny


        #--------------------------------------------------
        def read_h5_data(fdir, lap):


            conf.fields_file   = fdir + 'flds_'+str(lap)+'.h5'
            f5F = h5.File(conf.fields_file,'r')

            #read one and check size
            val0 = read_var(f5F, feature_list[0])
            nx,ny = np.shape(val0)

            #now read all
            x = np.zeros((nx,ny,len(feature_list)))
            for i,var in enumerate(feature_list):
                val = read_var(f5F, var)
                x[:,:,i] = val

            return x


        laps = [5000]
        #re-organize into subimages

        #subimage size
        splitf = 1

        #total size of the complete image in subimages
        splitfx = splitf*len(laps)
        splitfy = splitf

        nx,ny = read_h5_data_size(fdir, laps[0])
        nxs = nx//splitf
        nys = ny//splitf
        data = np.zeros((nxs,nys,splitfx*splitfy,len(feature_list)))

        for ilap, lap in enumerate(laps):
            x = read_h5_data(fdir, lap)

            fea_means = np.zeros(len(feature_list))
            n0 = x[:,:,0]
            bx = x[:,:,1]
            by = x[:,:,2]
            bz = x[:,:,3]
            ex = x[:,:,4]
            ey = x[:,:,5]
            ez = x[:,:,6]
            jx = x[:,:,7]
            jy = x[:,:,8]
            jz = x[:,:,9]

            n0s = np.std(n0)
            bs = np.std(np.sqrt(bx*bx + by*by + bz*bz))
            es = np.std(np.sqrt(ex*ex + ey*ey + ez*ez))
            js = np.std(np.sqrt(jx*jx + jy*jy + jz*jz))

            x[:,:,0] /= n0s

            x[:,:,1] /= bs
            x[:,:,2] /= bs
            x[:,:,3] /= bs

            x[:,:,4] /= es
            x[:,:,5] /= es
            x[:,:,6] /= es

            x[:,:,7] /= js
            x[:,:,8] /= js
            x[:,:,9] /= js

            #--------------------------------------------------
            print("old array img size is {} x {}".format(nx, ny))
            print("new array subimg size is {} x {} with a split of {}".format(nxs, nys, splitf))
            for i in range(len(feature_list)):
                spf = ilap*splitf**2 #global image counter (=splitf^2)


                for spfy in range(splitf):
                        for spfx in range(splitf):

                                ix0 =  spfx*nxs
                                ix1 = (spfx+1)*nxs

                                iy0 = spfy*nys
                                iy1 = (spfy+1)*nys

                                #step ranges one subimg forward
                                data[:,:,spf,i] = x[iy0:iy1, ix0:ix1, i]
                                spf += 1

        #--------------------------------------------------
        # feature engineering
        # re-defining feature_list here
        if False:
                feature_list = [
                                'rho', #0
                                'bperp_2', #1
                                'bpar_2', #2
                                'eperp_2',#3
                                'epar', #_2',#4
                                'jperp_abs', #5
                                'jpar_abs',#6
                                'je',#7
                                'exb', #8
                                'je_par',#9
                                'je_perp', #10
                                'exb_perp',#11
                                'exb_par',#12
                                'v_drift',#13
                                'v_drift_perp',#14
                                'v_drift_par',#15

                                ]

                data2 = np.zeros((nxs,nys,splitfx*splitfy,len(feature_list)))

                #rho
                data2[:,:,:,0] = data[:,:,:,0]

                #bperp_2
                data2[:,:,:,1] = data[:,:,:,1]**2 + data[:,:,:,2]**2

                #bpar:
                data2[:,:,:,2] = (data[:,:,:,3])**2

                #eperp
                data2[:,:,:,3] = data[:,:,:,4]**2 + data[:,:,:,5]**2

                #epar
                data2[:,:,:,4] = (data[:,:,:,6])

                #jperp
                data2[:,:,:,5] = np.sqrt(data[:,:,:,7]**2 + data[:,:,:,8]**2)

                #jpar_abs
                data2[:,:,:,6] = np.abs(data[:,:,:,9])
                #data2[:,:,:,6] = data[:,:,:,9]

                #je
                #ata2[:,:,:,7] = \
                data2[:,:,:,7] = \
                         data[:,:,:,4]*data[:,:,:,7] +  \
                         data[:,:,:,5]*data[:,:,:,8] +  \
                         data[:,:,:,6]*data[:,:,:,9]

                #exb=q*v
                data2[:,:,:,8]= \
                       np.sqrt((data[:,:,:,5]*data[:,:,:,3]- data[:,:,:,6]*data[:,:,:,2])**2 + \
                       (data[:,:,:,4]*data[:,:,:,3]- data[:,:,:,6]*data[:,:,:,1])**2)
                        #(data[:,:,:,4]*data[:,:,:,2]- data[:,:,:,5]*data[:,:,:,1])**2)

                #je_par
                #in parallel xy direction
                #ata2[:,:,:,7] = \
                data2[:,:,:,9] = \
                        data[:,:,:,6]*data[:,:,:,9]

                #je_perp
                # in perpendicular xy direction
                #ata2[:,:,:,7] = \
                data2[:,:,:,10] = \
                        data[:,:,:,4]*data[:,:,:,7] + data[:,:,:,5]*data[:,:,:,8]

                # exb_perp
                data2[:,:,:,11]= \
                       np.sqrt( \
                        (data[:,:,:,5]*data[:,:,:,3] - data[:,:,:,6]*data[:,:,:,2])**2  \
                       +(data[:,:,:,4]*data[:,:,:,3] - data[:,:,:,6]*data[:,:,:,1])**2
                       )

               # exb_par
               # in parallel z direction
                data2[:,:,:,12]=np.sqrt( (data[:,:,:,4]*data[:,:,:,2]- data[:,:,:,5]*data[:,:,:,1])**2 )


                #: velocity drift
                b2 = (data[:,:,:,1]**2 + data[:,:,:,2]**2+(data[:,:,:,3])**2)
                data2[:,:,:,13] = np.divide(np.sqrt(
                    (data[:,:,:,5]*data[:,:,:,3] - data[:,:,:,6]*data[:,:,:,2])**2 + \
                    (data[:,:,:,4]*data[:,:,:,3] - data[:,:,:,6]*data[:,:,:,1])**2 + \
                    (data[:,:,:,4]*data[:,:,:,2] - data[:,:,:,5]*data[:,:,:,1])**2
                ) , b2)

                #v_drift_perp
                data2[:,:,:,14] = np.divide(np.sqrt(
                 (data[:,:,:,5]*data[:,:,:,3] - data[:,:,:,6]*data[:,:,:,2])**2 + \
                 (data[:,:,:,4]*data[:,:,:,3] - data[:,:,:,6]*data[:,:,:,1])**2 ) , b2)

                #v_drift_par
                data2[:,:,:,15] = np.divide(np.sqrt(
                 (data[:,:,:,4]*data[:,:,:,2] - data[:,:,:,5]*data[:,:,:,1])**2) , b2)

        if True:
                feature_list = [
                    'bperp_2', #1 Picked as best for sheets
                    'j_par_abs',#7
                    'je_par'
                    ]
                data2 = np.zeros((nxs,nys,splitfx*splitfy,len(feature_list)))

                #B_PERP**2
                data2[:,:,:,0] = data[:,:,:,1]**2 + data[:,:,:,2]**2

                #J_PAR**abs
                data2[:,:,:,1] = np.abs(data[:,:,:,9])

                #JE_PAR
                data2[:,:,:,2] = \
                        data[:,:,:,6]*data[:,:,:,9]

 ###############################

        print("Min and max for exb")

        #Before the data hack:
        data = data2

        #--------------------------------------------------

        #plot original images
        if True:
            fig = plt.figure(1, figsize=(6,6*len(laps)), dpi=300)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
            gs = plt.GridSpec(splitfx, splitfy)
            gs.update(hspace = 0.05)
            gs.update(wspace = 0.05)

            axs = []
            for ix in range(splitfx):
                for iy in range(splitfy):
                    axs.append( plt.subplot(gs[ix,iy]) )


            plot_confs = {
            'rho':     {'cmap':'viridis', 'vmin':0.0, 'vmax':15.0}, #0
            'bperp_2': {'cmap':'viridis', 'vmin':0.0, 'vmax':30.0}, #1
            'bpar_2':  {'cmap':'viridis', 'vmin':0.0, 'vmax':50.0}, #2
            'eperp_2': {'cmap':'viridis', 'vmin':0.0, 'vmax':40.0}, #3
            'epar_2':  {'cmap':'viridis', 'vmin':0.0, 'vmax':5.0}, #4
            'jperp_abs': {'cmap':'viridis', 'vmin':0.0, 'vmax':5.0}, #5
            'jpar_abs':  {'cmap':'viridis',    'vmin':0.0, 'vmax':5.0}, #6
            'je':      {'cmap':'PRGn',    'vmin':-2.0, 'vmax':2.0}, #7
            'exb':     {'cmap':'viridis', 'vmin':0.0, 'vmax':0.8}, #8 out atm: edasi -1 for location
            'j':      {'cmap':'PRGn',    'vmin':-2.0, 'vmax':2.0}, #9
            'je_par':  {'cmap':'PRGn',    'vmin':-2.0, 'vmax':2.0}, #9
            'je_perp': {'cmap':'PRGn',    'vmin':-2.0, 'vmax':2.0}, #10
            'exb_perp':{'cmap':'Purples', 'vmin':0.0, 'vmax':69.0}, #11
            'exb_par': {'cmap':'Purples', 'vmin':0.0, 'vmax':28.0}, #12
            'v_drift': {'cmap': 'viridis', 'vmin': 0.0, 'vmax':1.0}, #13
            'v_drift_perp': {'cmap': 'viridis', 'vmin': 0.0, 'vmax':1.0}, #14
            'v_drift_par': {'cmap': 'viridis', 'vmin': 0.0, 'vmax':1.0}, #15
            'epar':  {'cmap':'RdBu', 'vmin':-2.0, 'vmax':2.0}, #4
            'jpar':  {'cmap':'RdBu',    'vmin':-2.0, 'vmax':2.0}, #6

            }
        if True:
                print("plotting original images")
                i = 2 #only pick one and plot first feature

                for spf in range(splitfx*splitfy):
                    ax = axs[spf]

                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)


                    fea = feature_list[i]
                    ops = plot_confs[fea]
                    print("feature {} min {} max {}".format(feature_list[i], np.min(data[:,:,spf,i]), np.max(data[:,:,spf,i])))

                    imshow(ax, data[:,:,spf,i],xmin, xmax, ymin, ymax,
                            cmap=ops['cmap'], vmin=ops['vmin'], vmax=ops['vmax'],
                            clip=None,
                            )

                fig.savefig('som_orig_6img_{}.pdf'.format(feature_list[i]))

        #--------------------------------------------------

        #plot "labels"/target
        if False:
                print("plotting target images")
                for spf in range(splitfx*splitfy):
                        ax = axs[spf]

                        ax.axes.get_xaxis().set_visible(False)
                        ax.axes.get_yaxis().set_visible(False)

                        imshow(ax, flat_target_data[:,:,spf],
                                   xmin, xmax, ymin, ymax,
                                   cmap='inferno', vmin=0.0, vmax=0.1)


                fig.savefig('kmeans_labels_8x8.pdf')


        #128,128,64,10 image here
        # subimg len x subimage width x number of subimages x feature_list
        print("orig shape before:", np.shape(data))

        print("creating feature data")
        #correct reshaping!
        #shape it into 128*128*64 (pixel-vector combining and flattening all sub-images) x 9 feature_list

        #if true, create the data; else read it from premade file

        #make features into a one long vector of pixels:
        if True:

                x = np.zeros((nxs*nys*splitfx*splitfy, len(feature_list))) #data matrix
                y = np.zeros((nxs*nys*splitfx*splitfy)) #target matrix

                j = 0
                for ix in range(nxs):
                        for iy in range(nys):
                                for isubim in range(splitfx*splitfy):
                                        for ifea in range(len(feature_list)):
                                                x[j, ifea] = data[ix, iy, isubim, ifea]

                                        j += 1


                #add also similarity measure for all features based on nearest neighbors

                #save feature list to file
                f5 = h5.File('data_{}.h5'.format(lap), 'w')
                dsetx = f5.create_dataset("features",  data=x)
                dsety = f5.create_dataset("target",  data=y)

                asciilist = [n.encode("ascii", "ignore") for n in feature_list]
                dsetf = f5.create_dataset("names",  data=asciilist)
                f5.close()

        else:
                f5 = h5.File('data_{}.h5'.format(lap), 'r')
                x = f5['features'][()]
                y = f5['target'][()]
                feature_list = f5['names'][()]

                feature_list = [n.decode('utf-8') for n in feature_list]
                f5.close()

        print(feature_list)

        print("shape after x:", np.shape(x))

        #--------------------------------------------------
        # analyze
        #1. standardize:
        scaler = StandardScaler()
        scaler = MinMaxScaler()

        scaler.fit(x)
        x = scaler.transform(x)


        ##### Using the SOM:

        #       POPSOM SOM:
        attr=pd.DataFrame(x)
        attr.columns=feature_list
        #parser.parse_args
        print("set dimensions", parser.parse_args())
        m=popsom.map(args.xdim, args.ydim, args.alpha, args.train)

        labels = [str(xxx) for xxx in range(len(x))]
        m.fit(attr,labels)
        m.starburst()

        m.significance()


        #Data matrix with neuron positions:
        data_matrix=m.projection()
        data_Xneuron=data_matrix['x']
        data_Yneuron=data_matrix['y']

        print("Printing Xneuron info")
        print(data_Xneuron)
        print("Printing Xneuron info position 5")
        print(data_Xneuron[4])
        print("Printing Yneuron info")
        print(data_Yneuron)

        #Neuron matrix with centroids:
        umat = m.compute_umat(smoothing=2)
        centrs = m.compute_combined_clusters(umat, False, 0.15) #0.15
        centr_x = centrs['centroid_x']
        centr_y = centrs['centroid_y']

        #create list of centroid _locations
        nx, ny = np.shape(centr_x)

        centr_locs = []
        for i in range(nx):
                for j in range(ny):
                        cx = centr_x[i,j]
                        cy = centr_y[i,j]

                        centr_locs.append((cx,cy))

        unique_ids = list(set(centr_locs))
        print(unique_ids)
        n_clusters = len(unique_ids)
        print("Number of clusters")
        print(n_clusters)

        mapping = {}
        for I, key in enumerate(unique_ids):
                print(key, I)
                mapping[key] = I

        #print("testing labeling for 18,0", mapping[(18,0)])

        clusters = np.zeros((nx,ny))
        for i in range(nx):
                for j in range(ny):
                        key = (centr_x[i,j], centr_y[i,j])
                        I = mapping[key]

                        clusters[i,j] = I


        print(centr_x)
        print(centr_y)

        print("clusters")
        print(clusters)
        print(np.shape(clusters))
        #TRANSFER RESULT BACK INTO ORIGINAL DATA PLOT
        if True:
                print("plotting SOM cluster images")
                fig = plt.figure(1, figsize=(6,6*len(laps)), dpi=300) #need to hack the number of columns according to image count
                fig.clf()

                fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
                gs = plt.GridSpec(splitfx, splitfy)
                gs.update(hspace = 0.05)
                gs.update(wspace = 0.05)


                #Create N number of colors:
                def get_N_HexCol(N=n_clusters):
                        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
                        hex_out = []
                        for rgb in HSV_tuples:
                                rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
                                hex_out.append('#%02x%02x%02x' % tuple(rgb))
                        return hex_out
                cols=get_N_HexCol(N=n_clusters)
                print("The count of colors, should be same as number of clusters")
                print(len(cols))
                print("Colors", cols)

                cmap = colors.ListedColormap(cols) #should here be len(cols)?
                bounds=np.arange(cmap.N)
                norm = colors.BoundaryNorm(bounds, cmap.N)
                data_back = np.zeros((nxs,nys,splitfx*splitfy))

                xinds = np.zeros(len(data_Xneuron))
                print("shape of xinds:", np.shape(xinds))
                j = 0
                for ix in range(nxs):
                        for iy in range(nys):
                                for isubim in range(splitfx*splitfy):
                                        data_back[ix,iy,isubim] = clusters[data_Xneuron[j], data_Yneuron[j]]
                                        xinds[j] = clusters[data_Xneuron[j], data_Yneuron[j]]
                                        j += 1
                f5 = h5.File('clusterID_{}.h5'.format(lap), 'w')
                dsetx = f5.create_dataset("databack",  data=data_back)
                f5.close()
                print("Done writing the cluster ID file")

                for spf in range(splitfx*splitfy):
                        ax = fig.add_subplot(splitfx,splitfy, spf+1, xticks=[], yticks=[])

                        ax.axes.get_xaxis().set_visible(False)
                        ax.axes.get_yaxis().set_visible(False)

                        ax.clear()
                        ax.minorticks_on()
                        ax.set_xlim(xmin, xmax)
                        ax.set_ylim(ymin, ymax)
                        extent = [ xmin, xmax, ymin, ymax ]
                        mgrid = data_back[:,:,spf].T

                        im = ax.imshow(mgrid,
                        fig.savefig('SOM_transform.pdf')


        print("done plotting")

       # #PLOTTING:
        #visualize clusters

        x = np.array(x)
        data_back = np.array(data_back)
        if True:
                print("visualizing SOM data")
                fig2 = plt.figure(2, figsize=(splitfx,splitfy), dpi=400)
                fig2.clf()

                ic = 0
                nfea = len(feature_list)
                for jcomp in range(nfea):
                        for icomp in range(nfea):
                                ic += 1
                                print("i {} j {}".format(icomp, jcomp))

                                #skip covariance with itself
                                if icomp == jcomp:
                                        continue

                                #skip upper triangle
                                if icomp > jcomp:
                                        continue

                                ax = fig2.add_subplot(nfea, nfea, ic)

                                for ki in range(n_clusters):
                                        indxs = np.where(xinds == ki)
                                        #print("len of found pixels:", len(indxs), indxs)
                                        xx = x[indxs, icomp]
                                        yy = x[indxs, jcomp]

                                        xxt = xx[::500]
                                        yyt = yy[::500]

                                        ax.scatter(
                                                xxt,
                                                yyt,
                                                c=cols[ki],
                                                marker='.',
                                                s=0.1,
                                                rasterized=True,
                                                )

                                if False:
                                        #visualize most dissipative points
                                        xx = x[:,icomp]
                                        yy = x[:,jcomp]
                                        zz = y[:]

                                        xxd = xx[np.where(np.abs(zz) > 0.020)]
                                        yyd = yy[np.where(np.abs(zz) > 0.020)]
                                        zzd = zz[np.where(np.abs(zz) > 0.020)]

                                        xxd = xxd[::100]
                                        yyd = yyd[::100]
                                        zzd = zzd[::100]

                                        print("found {} points above threshold".format(len(xxd)))

                                        ax.scatter(xxd,yyd,c=zzd,
                                                                cmap='inferno',
                                                                vmin=-0.015,
                                                                vmax= 0.015,
                                                                marker='.',
                                                                s=0.05,
                                                                alpha=0.1,
                                                                )

                                if jcomp == nfea-1:
                                        ax.set_xlabel('{}'.format(feature_list[icomp]))
                                else:
                                        ax.set_xticks([])

                                if icomp == 0:
                                        ax.set_ylabel('{}'.format(feature_list[jcomp]))
                                else:
                                        ax.set_yticks([])

                fig2.savefig('som_clusters.pdf')
