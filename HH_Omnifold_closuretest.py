import numpy as np
import matplotlib.pyplot as plt
import energyflow as ef
import energyflow.archs
import tensorflow as tf

import omnifold
import modplot
import ibu

    
#increase Keras/Python precision
tf.keras.backend.set_floatx('float64')

# for plotting
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'serif'


# load datasets
datasets={'Embed': np.load('JetPt_Kevinsfullembed.npz'),'Data': np.load('JetPt_data.npz')}
# ^above uses full embedding sample (601 runs)


# gen, sim
#embedding, nature = datasets['Embed'], datasets['Data'] # <-- for regular Omnifold
embedding, nature = datasets['Embed'], datasets['Embed'] # <-- for closure test


# load weights
# (inverse luminoisity  weights)
lumis=np.load('sim_weights.npz')


#how many iterations of unfolding process
# (I have been setting this to 3)
itnum=0


# for unifold
obs_multifold = ['JetpT']

# a dictionary to hold information about the observables
obs = {}
# DICTIONARY ENTRIES
# jet pT and histogram style information
obs.setdefault('JetpT', {}).update({
    'func': lambda dset, ptype: dset[ptype + '_JetPt'],
    'nbins_det': 80, 'nbins_mc': 80,
    'xlim': (0,50),
    'xlabel': r'Jet pT [GeV]',
    'ylabel': r''
})
    

# additional histogram and plot style information
hist_style = {'histtype':'step','density': True, 'lw': 1, 'zorder': 2}
gen_style = {'histtype':'step','linestyle': '--', 'color': 'tab:blue', 'lw': 1.0,'density':True}
omnifold_style = {'ls': '-','ms': 1, 'color': 'tab:red', 'zorder':3,'lw':1.0}
ibu_style = {'ls': '-', 'marker': 'o', 'ms': 2.5, 'color': 'gray', 'zorder': 1}

#truth (for closure test)
truth_style={'markerfacecolor':'green','lw':1.15, 'zorder': 0,'facecolor':(0.75,0.875,0.75)}


# calculate quantities to be stored in obs
for obkey,ob in obs.items():
    # calculate observable for (GEN), SIM, DATA, and TRUE
    
    # SIM
    ob['genobs'],ob['simobs']=ob['func'](embedding, 'gen'),ob['func'](embedding,'sim')
    
    # DATA
    ob['dataobs'] = ob['func'](nature,'sim')
    ob['bins_mc'] = np.linspace(ob['xlim'][0], ob['xlim'][1], ob['nbins_mc']+1)

    
    #ensure data/sim have same number of events
    NEvts= min(np.shape(ob['simobs']) )
    ob['dataobs']= (ob['dataobs'][:NEvts])
    ob['genobs']= ob['genobs'][:NEvts]
    ob['simobs']= ob['simobs'][:NEvts]
    
    # setup bins
    ob['bins_det'] = np.linspace(ob['xlim'][0], ob['xlim'][1], ob['nbins_det']+1)
    ob['bins_mc'] = np.linspace(ob['xlim'][0], ob['xlim'][1], ob['nbins_mc']+1)
    ob['midbins_det'] = (ob['bins_det'][:-1] + ob['bins_det'][1:])/2
    ob['midbins_mc'] = (ob['bins_mc'][:-1] + ob['bins_mc'][1:])/2
    ob['binwidth_det'] = ob['bins_det'][1] - ob['bins_det'][0]
    ob['binwidth_mc'] = ob['bins_mc'][1] - ob['bins_mc'][0]
    
    # get the histograms of GEN, DATA
    ob['genobs_hist'] = np.histogram(ob['genobs'], bins=ob['bins_mc'], density=True)[0]
    ob['data_hist'] = np.histogram(ob['dataobs'], bins=ob['bins_det'], density=True)[0]

    
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
    ##these model size/training params result in quicker training(from demo)
    model_layer_sizes = [100, 100]
    
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
                'patience': 10, 'filepath': 'Step1_{}', 'save_weights_only': False,
                'modelcheck_opts': {'save_best_only': True, 'monitor':'acc','verbose': 1}}
    
    # model parameters for the Step 2 network
    mc_args = {'input_dim': len(obs_multifold), 'dense_sizes': model_layer_sizes,
               'patience': 10, 'filepath': 'Step2_{}', 'save_weights_only': False,
               'modelcheck_opts': {'save_best_only': True, 'verbose': 1,'monitor':'acc'}}
    
    
    # general training parameters
    fitargs = {'batch_size': 500, 'epochs': 200, 'verbose': 1}
    #fitargs = {'batch_size': 500, 'epochs': 100, 'verbose': 1}
    # ^ use this for a full training(from demo)
    
    # reweight the sim and data to have the same total weight to begin with
    ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])
    wdata = np.ones(ndata)

    winit = ndata/nsim*np.ones(nsim)*10 #<--- "10's" (currently using for Closure Test diagnosis)
    #winit = ndata/nsim*np.ones(nsim)   #<--- omnifold example default; "1's" (currently using)
    #winit= ndata/nsim * lumis['sim_weights'] #<--- inverse lumis (Youqi is using these)

    
    #weights used on DATA to perform closure test
    wclosure=ndata/nsim*np.ones(nsim)
    
    # apply the OmniFold procedure to get weights for the generation
    multifold_ws = omnifold.omnifold(X_gen, Y_gen, X_det, Y_det, wclosure, winit,
                                     (ef.archs.DNN, det_args), (ef.archs.DNN, mc_args),fitargs, val=0.2, it=itnum, trw_ind=-2, weights_filename='1sweights_1iter_1observable_wIBU_fulltrain_closure')


    
    ##### CLOSURE TEST TRUTH ######
    ob['truth_hist'],ob ['truth_hist_unc'] = modplot.calc_hist(ob['genobs'], weights=wclosure[:NEvts],bins=ob['bins_mc'], density=True)[:2]
    ###############################

    
        
    ########## PLOT RESULTS OF UNFOLDING ***************
    for i,(obkey,ob) in enumerate(obs.items()):
            
        # get the styled axes on which to plot
        fig, [ax0, ax1] = modplot.axes(**ob)
        if ob.get('yscale') is not None:
            ax0.set_yscale(ob['yscale'])

        ax0.set_yscale('log')
        ax0.set_ylim(10e-7,10e0)            
        ax0.set_title("Closure Test: All partonicpT bins\n Winit=10, Wclosure=1 \n iter=3, nbins=80") 

        
        # "closure test", embed weighted with inverse lumi
        #data
        ax0.hist(ob['dataobs'], bins=ob['bins_det'], color='blue',weights=wclosure, label='"Data"(Det-Lev embed w/ Wclosure)', **hist_style)
            
        #embed
        ax0.hist(ob['simobs'], bins=ob['bins_det'], color='orange', label='"Sim"(Det-Lev embed)', **hist_style)
            
        #truth
        # plot the "gen" histogram of the observable
        ax0.plot(ob['midbins_mc'], ob['genobs_hist'],color='black',label='"Gen"(Part-Lev embed)')
            
        # plot the "truth" histogram of the observable
        ax0.fill_between(ob['midbins_mc'], ob['truth_hist'],color='green',label='"Truth"(Gen w/ Wclosure)',alpha=0.5)
    
        # plot the OmniFold distribution
        of_histgen, of_histgen_unc = modplot.calc_hist(ob['genobs'], weights=multifold_ws[2*itnum], 
                                                       bins=ob['bins_mc'], density=True)[:2]
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


        # ratio plot formatting
        ax1.fill_between(ob['midbins_mc'], 1 - truth_unc_ratio, 1 + truth_unc_ratio, 
                         facecolor=truth_style['facecolor'], zorder=-2)
        ax1.errorbar(ob['midbins_mc'], ibu_ratio, xerr=ob['binwidth_mc']/2, yerr=ibu_unc_ratio, 
                     color=ibu_style['color'], **modplot.style('errorbar'))
        ax1.errorbar(ob['midbins_mc'], of_ratio, xerr=ob['binwidth_mc']/2, yerr=of_unc_ratio, 
                     color=omnifold_style['color'], **modplot.style('errorbar'))

        ax1.set_ylim([0.6,1.4])

        
        # legend style and ordering
        loc, ncol = ob.get('legend_loc', 'upper right'), ob.get('legend_ncol', 2)
        modplot.legend(ax=ax0, frameon=False, loc=loc, ncol=ncol)

        plt.show()
        
    

