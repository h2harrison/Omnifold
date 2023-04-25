import numpy as np
import matplotlib.pyplot as plt
import energyflow as ef
import energyflow.archs
import tensorflow as tf
from scipy.optimize import curve_fit

import omnifold
import modplot
import ibu
import sys
import getopt
import random

    
#increase Keras/Python precision
tf.keras.backend.set_floatx('float64')

# for plotting
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'serif'


# load datasets
#datasets={'Embed': np.load('JetPt_embed.npz'),'Data': np.load('JetPt_data.npz')}
#datasets={'Embed': np.load('JetPt_embed_50runsTESTLUMI_4.21.23.npz'),'Data': np.load('JetPt_data.npz')}
datasets={'Embed': np.load('JetPt_embed_151runs_2.23.23.npz'),'Data': np.load('JetPt_data.npz')}


#full embedding data set available
fullembed= datasets['Embed']
fullembed_size= int((fullembed['sim_JetPt']).size) # <-- # of jets

print("**********")
print("Embed NEvts:",fullembed_size)
print("**********")

# gen, sim
embedding, nature = datasets['Embed'], datasets['Embed'] # <-- for closure test


#used these to calculate 1/lumi in npz file
xsec=np.array([9.006,1.462,3.544e-1,1.514e-1,2.489e-2,5.846e-3,2.305e-3,3.427e-4,4.563e-5,9.738e-6,5.020e-7])
fudgefact=np.array([(1/1.228), (1/1.051), (1/1.014)])


# load weights
# (inverse luminoisity  weights)
#lumis=np.load('weights_embed_allruns.npz')
lumis=np.load('weights_embed_151runs_2.23.23.npz')
#lumis=np.load('weights_embed_50runsTESTLUMI_4.21.23.npz')


#add seed to minimize fluctuations in fit
seed=43
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


#how many iterations of unfolding process
itnum=4
#itnum=0

# for unifold
obs_multifold = ['JetpT']


# a dictionary to hold information about the observables
obs = {}
# DICTIONARY ENTRIES
# jet pT and histogram style information
obs.setdefault('JetpT', {}).update({
    'func': lambda dset, ptype: dset[ptype + '_JetPt'],
    #'nbins_det': 50, 'nbins_mc': 50,
    'nbins_det': 100, 'nbins_mc': 100,
    'xlim': (0,55),
    #'xlim': (0,30),
    #OG settings
    'xlabel': r'Jet pT [GeV]'
})
    

# additional histogram and plot style information
hist_style = {'histtype':'step','density': True, 'zorder': 2}
#gen_style = {'histtype':'step','linestyle': '--', 'lw': 1.0,'density':True
gen_style = {'linestyle': '--', 'color': 'black', 'lw': 1.15}
#omnifold_style = {'histtype':'step','zorder':2,'ls': '-','lw':1.0}
omnifold_style = {'color':'tab:red','zorder':2,'ls': '-','lw':1.0}
ibu_style = {'ls': '-', 'marker': 'o', 'ms': 2.5, 'color': 'gray', 'zorder': 1}
#truth_style={'edgecolor':'green','lw':1.15, 'zorder': 0,'facecolor':(0.75,0.875,0.75)}
truth_style = {'step': 'mid', 'edgecolor': 'green', 'facecolor': (0.75, 0.875, 0.75),
               'lw': 1.25, 'zorder': 0, 'label': '``Truth\"'}

# calculate quantities to be stored in obs
for obkey,ob in obs.items():
    # calculate observable for (GEN), SIM, DATA, and TRUE

    # SIM
    ## 1/2 closure test
    ob['genobs']=ob['func'](embedding,'gen')[::2]
    ob['simobs']=ob['func'](embedding,'sim')[::2]

    ##### CLOSURE TEST TRUTH ######
    ob['truthobs'] = (ob['func'](nature, 'gen')[1::2])
    
    # DATA
    ## 1/2 closure test
    ob['dataobs'] = (ob['func'](nature,'sim')[1::2])

    ob['bins_mc'] = np.linspace(ob['xlim'][0], ob['xlim'][1], ob['nbins_mc']+1)
    
    #ensure data/sim have same number of event
    #NEvts= min(np.shape(ob['simobs']))
    NEvts= min(np.shape(ob['dataobs']))

    ob['dataobs']= (ob['dataobs'])[:NEvts]
    ob['genobs']= ob['genobs'][:NEvts]
    ob['simobs']= ob['simobs'][:NEvts]
    ob['truthobs']= ob['truthobs'][:NEvts]
    
    
    # setup bins
    ob['bins_det'] = np.linspace(ob['xlim'][0], ob['xlim'][1], ob['nbins_det']+1)
    ob['bins_mc'] = np.linspace(ob['xlim'][0], ob['xlim'][1], ob['nbins_mc']+1)
    ob['midbins_det'] = (ob['bins_det'][:-1] + ob['bins_det'][1:])/2
    ob['midbins_mc'] = (ob['bins_mc'][:-1] + ob['bins_mc'][1:])/2
    ob['binwidth_det'] = ob['bins_det'][1] - ob['bins_det'][0]
    ob['binwidth_mc'] = ob['bins_mc'][1] - ob['bins_mc'][0]

    inverseL=lumis['sim_weights'][1::2]
    wdata=(inverseL)
    wdata=wdata[:NEvts]


    print("wdata:",wdata)
    #print("dataobs:",len(ob['dataobs']))
    
    
    # get the histograms of GEN, DATA
    ob['genobs_hist'] = np.histogram(ob['genobs'], bins=ob['bins_mc'], density=True)[0]
    ob['data_hist'] = np.histogram(ob['dataobs'], bins=ob['bins_det'],weights=wdata,density=True)[0]

    ob['truth_hist'],ob ['truth_hist_unc'] = modplot.calc_hist(ob['genobs'],bins=ob['bins_mc'],weights=wdata)[:2]
    #ob['truth_hist'],ob ['truth_hist_unc'] = modplot.calc_hist(ob['genobs'],bins=ob['bins_mc'],weights=wdata)[:2]
    
    ###### FOR IBU ######
    # compute (and normalize) the response matrix between GEN and SIM                                     
    ob['response'] = np.histogram2d(ob['simobs'], ob['genobs'], bins=(ob['bins_det'], ob['bins_mc']))[0]
    ob['response'] /= (ob['response'].sum(axis=0) + 10**-50)
    
    # perform iterative Bayesian unfolding                                                                
    ob['ibu_phis'] = ibu.ibu(ob['data_hist'], ob['response'], ob['genobs_hist'],ob['binwidth_det'], ob['binwidth_mc'], it=itnum)
    ob['ibu_phi_unc'] = ibu.ibu_unc(ob, it=itnum, nresamples=25)
    ######################
    print('Done with', obkey)

    
    # OMNIFOLD
    #model_layer_sizes = [100, 100, 100] # "BEST training"
    #model_layer_sizes = [100, 100] # "better training"
    model_layer_sizes = [100] #<--- "easier" according to Dima
    
    # set up the array of data/simulation detector-level observables
    X_det = np.asarray([np.concatenate((obs[obkey]['dataobs'], obs[obkey]['simobs'])) for obkey in obs_multifold]).T
    Y_det = ef.utils.to_categorical(np.concatenate((np.ones(len(obs['JetpT']['dataobs'])),np.zeros(len(obs['JetpT']['simobs'])))))
    
    # set up the array of generation particle-level observables
    X_gen = np.asarray([np.concatenate((obs[obkey]['genobs'], obs[obkey]['genobs'])) for obkey in obs_multifold]).T
    Y_gen = ef.utils.to_categorical(np.concatenate((np.ones(len(obs['JetpT']['genobs'])),np.zeros(len(obs['JetpT']['genobs'])))))
    
    # standardize the inputs
    X_det = (X_det - np.mean(X_det, axis=0))/np.std(X_det, axis=0)
    X_gen = (X_gen - np.mean(X_gen, axis=0))/np.std(X_gen, axis=0)

    
    # Specify the training parameters
    # model parameters for the Step 1 network
    det_args = {'input_dim': len(obs_multifold), 'dense_sizes': model_layer_sizes,
                'patience': 50, 'filepath': 'Step1_{}', 'save_weights_only': False,'modelcheck_opts': {'save_best_only': True,'verbose': 1}}
    
    # model parameters for the Step 2 network
    mc_args = {'input_dim': len(obs_multifold), 'dense_sizes': model_layer_sizes,
               'patience': 50, 'filepath': 'Step2_{}', 'save_weights_only': False,
               'modelcheck_opts': {'save_best_only': True, 'verbose': 1}}
    
    #fitargs = {'batch_size': 500, 'epochs': 100, 'verbose': 1}
    fitargs = {'batch_size': 5000, 'epochs': 100, 'verbose': 1}
   
    # reweight the sim and data to have the same total weight to begin with
    ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])


    ######## GET WINIT FROM BEST FIT DATA/SIM #########

    exp=0
    def TestFunct(x):
        #return (10**exp)*(x**2)+0.05*x+1 #<-- "Dima's test funct"
        return 1+(0.05 * x)

    #ob['winit']= TestFunct(ob['truthobs'])
    winit= inverseL*TestFunct(ob['truthobs'])
    #ob['winit']=ndata/nsim*(lumis['sim_weights']/(max(lumis['sim_weights']))) #<---lumi
    #winit=ob['winit'][:NEvts]

    #winit = ndata/nsim*np.ones(nsim)   #<--- omnifold example default; "1's"
    #winit = ndata/nsim*inverseL
    #winit = ndata/nsim*(lumis['sim_weights'])/(max(lumis['sim_weights'])) #<---norm lumi
    winit=winit[:NEvts]


    ##################################################
    
    # apply the OmniFold procedure to get weights for the generation
    multifold_ws = omnifold.omnifold(X_gen, Y_gen, X_det, Y_det, wdata, winit,(ef.archs.DNN, det_args),(ef.archs.DNN, mc_args),fitargs, val=0.2, it=itnum, trw_ind=-2,weights_filename='1sweights_3iter_1observable')

                
    ########## PLOT RESULTS OF UNFOLDING ***************
    for i,(obkey,ob) in enumerate(obs.items()):
        
        # get the styled axes on which to plot
        fig, [ax0, ax1] = modplot.axes(**ob)
        if ob.get('yscale') is not None:
            ax0.set_yscale(ob['yscale'])
        
        ax0.set_yscale('log')
        ax0.set_ylim(10e-10,10e0)
        #ax0.set_ylim(10e-10,10e1)            

        #ax0.set_title(f"Closure: 50 runs \n Winit=$0.05pT_t+1$ \n Wdata=$(1/Lumi)*(1+0.05pT_t)$, iter={itnum}")
        #ax0.set_title(f"Closure: 150 runs \n Winit=$10^{exp}pT_t^2+0.05pT_t+1$ \n $Wdata=pT_t$, iter={itnum}")
        ax0.set_title("Closure: 50 runs \n Winit=(1/$\mathscr{L}$) \n Wdata=(1/$ \mathscr{L})(1+0.05pT_t) $, iter=4")
        #ax0.set_title("Closure: 150 runs \n Winit=1+(0.05 * $\mathscr{L}$) \n Wdata=(1/$ \mathscr{L}) $, iter=4")
        #ax0.set_title("Closure: 50 runs, ALL part. pTbin")
        ax0.set(xlabel= ob['xlabel'])
        ax0.set(ylabel='Normalized Num. Evts')
        
        # "closure test", embed weighted with inverse lumi
        #data
        ax0.hist(ob['dataobs'], bins=ob['bins_det'], color='blue', label='"Data" (Embed Sample #2, Wdata)',weights=wdata,**hist_style)
            
        #embed
        ax0.hist(ob['simobs'], bins=ob['bins_det'], color='orange', label='"Sim"(Embed Sample #1)',ls='--',**hist_style)
        
        #gen
        #ax0.hist(ob['genobs'], bins=ob['bins_det'], color='black', label='"Gen"(Embed Sample #1)',ls='--',**gen_style)
        ax0.plot(ob['midbins_mc'], ob['genobs_hist'], **gen_style)
        
        #truth
        #ax0.hist(ob['truthobs'], bins=ob['bins_det'], weights=wdata,color='green', alpha=0.5,label='Truth',ls='--')
        ax0.fill_between(ob['midbins_mc'], ob['truth_hist'], **truth_style)
        
        # plot the OmniFold distribution
        of_histgen, of_histgen_unc = modplot.calc_hist(ob['genobs'], weights=multifold_ws[2*itnum],bins=ob['bins_mc'],density=True)[:2]
        #of_histgen, of_histgen_unc = modplot.calc_hist(ob['genobs'], weights=multifold_ws[2*itnum],bins=ob['bins_mc'])[:2]
        #ax0.hist(ob['genobs'], bins=ob['bins_mc'], weights=multifold_ws[2*itnum],label='Multifold',color='red',**omnifold_style)
        ax0.plot(ob['midbins_mc'], of_histgen, **omnifold_style, label='MultiFold')
        
        # plot the IBU distribution
        ax0.plot(ob['midbins_mc'], ob['ibu_phis'][itnum], **ibu_style, label='IBU ')


        # Plot the Ratios of the OmniFold and IBU distributions to truth (with statistical uncertainties)
        ibu_ratio = ob['ibu_phis'][itnum]/(ob['truth_hist'] + 10**-50)
        of_ratio = of_histgen/(ob['truth_hist'] + 10**-50)
        ax1.plot([np.min(ob['midbins_mc']), np.max(ob['midbins_mc'])], [1, 1], '-', color='green', lw=0.75)

        # ratio uncertainties
        truth_unc_ratio = ob['truth_hist_unc']/(ob['truth_hist'] + 10**-50)

        ibu_unc_ratio = ob['ibu_phi_unc']/(ob['truth_hist'] + 10**-50)
        of_unc_ratio = of_histgen_unc/(ob['truth_hist'] + 10**-50)



        #DEBUG
        #of_histgen_nodensity, of_histgen_unc_nodensity = modplot.calc_hist(ob['genobs'], weights=multifold_ws[2*itnum],bins=ob['bins_mc'])[:2]
        #of_unc_ratio_nodensity = of_histgen_unc_nodensity/(ob['truth_hist'] + 10**-50)

        #of_ratio_nodensity = of_histgen_nodensity/(ob['truth_hist'] + 10**-50)
        
        #print('of_hist:',of_histgen)
        #print('of_hist NO DENSITY:',of_histgen_nodensity)
        
        
        # ratio plot formatting
        ax1.fill_between(ob['midbins_mc'], 1 - truth_unc_ratio, 1 + truth_unc_ratio, 
                         facecolor=truth_style['facecolor'], zorder=-2)
        ax1.errorbar(ob['midbins_mc'], ibu_ratio, xerr=ob['binwidth_mc']/2, yerr=ibu_unc_ratio, 
                     color=ibu_style['color'], **modplot.style('errorbar'))
        ax1.errorbar(ob['midbins_mc'], of_ratio, xerr=ob['binwidth_mc']/2, yerr=of_unc_ratio, 
                     color='red', **modplot.style('errorbar'))
        
        ax1.set_ylim([0.6,1.4])

        
        # legend style and ordering
        loc, ncol = ob.get('legend_loc', 'upper right'), ob.get('legend_ncol', 2)
        modplot.legend(ax=ax0,frameon=False, loc=loc, ncol=ncol)

        plt.show()
        
    

