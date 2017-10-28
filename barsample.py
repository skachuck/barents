"""
giamcice.py

Author: Samuel B. Kachuck

This module uses the emcee module to perform Markov Chain Monte Carlo sampling
to invert data for earth parameters.
"""
import numpy as np
import emcee
import giapy as gp

from giapy.giamc import sampleOut, gen_metadatastr,\
                        get_metadatadict

from giapy.numTools.stats import OnlineSamplingCovariance

import cPickle as pickle

rootdata = np.loadtxt(gp.MODPATH+'/data/obs/root_etal_2015_2a.txt').T
auriacdata = np.load(gp.MODPATH+'/data/obs/auriac_etal_2016_table2.p')
gps_locs = np.array(auriacdata.values())

def generativeModelIce(props, areaNames, ice, data, earth, topo, grid):
    """Return interpolated calculation to locations and times of data given an
    earth viscosity profile given by [paramzarray,  params] and convolved with
    ice.

    Parameters
    ----------
    params : 1D array
        An array [log(\eta_i), log(D)] of log viscosities followed by
        the log flexural rigidity. The log viscosities are those for layers
        whose thicknesses are given by paramzarray, in descending order.
    paramzarray : 1D array
        Thickness of the viscosity layers (starting from the surface).
    ice : <giapy.ice_tools.ice_history>
    data : <giapy.data_tools.emerge_data>
    topo : 2D array
        Today's topography - must match size with grid and ice
    grid : <giapy.map_tools.GridObject>
    preds : 
    """
    assert ice.shape == topo.shape == grid.shape, 'Array shapes inconsistent.'
    
    # Copy and alter the ice.
    icenew = ice.copy()
    icenew.updateAlterationAreas(dict(zip(areaNames, props)))
    icenew = icenew.applyAlteration()

    sim = gp.sim.GiaSimGlobal(earth, icenew, grid=grid) 
    sim.out_times = np.union1d(icenew.times, [-0.1, 0.1])
    result = sim.performConvolution(topo=topo, eliter=5)

    emergecalc = gp.data_tools.emergedata.calcEmergence(result, data, smooth=False)

    # Calculate the predictions here, where the convolution is performed.
    pred = predictions(result, emergecalc)

    return emergecalc, pred

def lnlike(propstdvs, areaNames, ice, data, earth, topo, grid, resCov=False,
                fitgrav=False):
    """

    """
    props = propstdvs[:len(areaNames)]
    stdev = propstdvs[len(areaNames):]

    if len(stdev) == 1:
        stdev = np.repeat(stdev[0], len(data.long_data))
    elif len(stdev) == len(data.locs):
        stdev = np.concatenate([np.repeat(sigloc, len(loc.ts)) for sigloc, loc
                                in zip(stdev, data)])
    else:
        raise ValueError('stdvs not right shape')

    emergecalc, pred = generativeModelIce(props, areaNames, 
                        ice, data, earth, topo, grid)

    residuals = (emergecalc.long_data - data.long_data)
    
    prob = -0.5*np.sum((residuals/stdev)**2) - np.sum(np.log(np.sqrt(2*np.pi)*stdev))

    if fitgrav:
        gravrates = pred[:33].reshape(11, 3)
        prob -= 0.5*np.sum(((gravrates[1:, 2] - rootdata[1]) / rootdata[2])**2)
        #prob += -0.5*((pred[0]*1000 - 0.2)/0.05)**2

    if not resCov:
        return float(prob), pred
    else:
        return float(prob), np.r_[pred, residuals]

def lnprior(params):
    """For boxcar (uniform) prior in range, return 0 in range, -np.inf outside.
    """
    if np.all(np.logical_and(0.01 < params, params < 4.)): 
        return 0.0
    return -np.inf

def lnprob(props, areaNames, ice, data, earth, topo, grid, nblobs,
                resCov=False, fitgrav=False):
    lp = lnprior(props[:len(areaNames)])

    if not np.isfinite(lp):
        if resCov:
            blobs = np.repeat(np.nan, nblobs+len(data.long_data))
        else:
            blobs = np.repeat(np.nan, nblobs)

        return -np.inf, blobs 
    
    prob, pred = lnlike(props, areaNames, ice, data, earth, topo, grid,
                        resCov, fitgrav)

    prob += lp

    return float(prob), pred

def get_uplrate(result, t0=-0.1, t1=0.1, specout=False):
    n0 = result['upl'].locateByTime(t0)
    n1 = result['upl'].locateByTime(t1)
            
    # x5 makes it in mm/yr
    uplrate = (result['upl'][n0] - result['upl'][n1])*5
                        
    if not specout:
        uplrate = result.inputs.harmTrans.spectogrd(uplrate)
    return uplrate

class CalcMultiGravrates(object):
    def __init__(self, grid, lpsigs, area):
        nlat = grid.shape[0]
        ms, self.ns = gp.sim.spharm.getspecindx(nlat-1)
        self.Lon, self.Lat = grid.Lon, grid.Lat
        self.area = area
        self.bps = []
        for lpsig in lpsigs:
            self.bps.append(gp.data_tools.gravdata.sphgauss_bp(nlat, lpsig, 600))

        self.nblobs = 3*(len(lpsigs) + 1) + 18

    def __call__(self, result, emergecalc):
        gravrate = gp.data_tools.gravdata.get_gravrate(result, specout=True)


        tmprate = result.inputs.harmTrans.spectogrd(gravrate)
        maxs = gp.map_tools.lonlatmax_area(self.Lon, self.Lat,  tmprate, self.area)
       
        gravs = [maxs]

        for bp in self.bps:
            tmprate = result.inputs.harmTrans.spectogrd(gravrate*bp[self.ns])
            maxs = gp.map_tools.lonlatmax_area(self.Lon, self.Lat,  tmprate, self.area)
            gravs.append(maxs)

        gravs = np.array(gravs)
        gravs[:,2] = -gravs[:,2]

        uplrates = result.inputs.grid.interp(get_uplrate(result).T, 
                            gps_locs[:,0], gps_locs[:,1], latlon=True)

        return np.r_[gravs.ravel(), uplrates]
        
if __name__ == '__main__':
    import sys, os, argparse

    parser = argparse.ArgumentParser(description='Perform MCMC in the Barents')
    parser.add_argument('nsteps', type=int)
    parser.add_argument('covkey')
    parser.add_argument('corrskip', type=int)
    parser.add_argument('fitgrav')
    parser.add_argument('--o', default='.')
    args = parser.parse_args()

    cname = '_'+args.c if args.c else ''

    nsteps, corrskip = args.nsteps, args.corrskip


    covkey = True if args.covkey=='True' else False  
    fitgrav = True if args.fitgrav == 'True' else False

    if fitgrav:
        dstyle = 'emer_grav'
    else:
        dstyle = 'emer'

    writefile='barentsMCMC_{}_{}{}_{}.txt'.format(estyle,istyle,cname,dstyle)
    writefile=args.o+'/'+writefile
    print('Writefile: {}'.format(writefile))

    try:
        f = open(writefile, 'r')
        f.close()
        sys.stdout.write('writefile exists. Continue this sample? [y/n] ')
        choice = raw_input().lower()
        if choice == 'y':
            contkey = 1
        elif choice == 'n':
            contkey = 0
    except:
        contkey = 0 


    ename = '75km0p04Asth_4e23Lith'
    iname = 'AA2_Tail_nochange5_hightres_Pers288_square'
    tname = 'sstopo288'


    configdict = {'earth': ename,
                  'ice'  : iname,
                  'topo' : tname}
    sim = gp.giasim.configure_giasim(configdict)

    earth, ice, topo = sim.earth, sim.ice, sim.topo
    grid = gp.maps.GridObject(mapparam={}, shape=topo.shape)
    iceNames = [str(i) for i in range(1, len(ice.areaProps.keys())+1)]
 

    # import the data, and select only the barents locations.
    emergedata = gp.data_tools.emergedata.importEmergeDataFromFile()
    datarange = range(409, 416) + range(418, 442) + range(444, 447) 
    emergedata = emergedata.filter(emergedata.by_recnbr, {'nbrs': datarange})

    barcent = np.s_[258:270, 165:187]
    predictions = CalcMultiGravrates(grid, np.arange(210, 310, 10), barcent)

    sys.stdout.write('loaded, starting the mcmc\r')
    sys.stdout.flush()

    ndim, nwalkers, nblobs = len(iceNames)+1, 2*(len(iceNames)+1), predictions.nblobs
    args = (iceNames, ice, emergedata, earth, topo, grid, nblobs)

    if contkey == 1:
        # Consistency check
        metadata = get_metadatadict(writefile)
        concheck = np.all([ndim == metadata['ndim'], 
                            nwalkers == metadata['nwalkers'],
                            nblobs == metadata['nblobs']])
        assert concheck, "emcee parameters don't match from %s" % writefile

        written = np.loadtxt(writefile, skiprows=metadata['linecount'])
        written = written.reshape((-1, nwalkers, ndim+2+predictions.nblobs))
        written = written.transpose([1,0,2])
        pos = written[:,-1,2:2+ndim]



        lnprob0 = written[:,-1,1]
        lnprob0 = None
        blobs0 = written[:,-1,2+ndim:2+ndim+nblobs]
        blobs0 = None

        # If we're to record the residual covariances,
        if covkey:
            # see if such covariances already exist and load it or,
            try:
                covMat = np.load(os.path.splitext(writefile)[0]+'_resCov.p', 'r')
            # if not, create it.
            except:
                covMat = OnlineSamplingCovariance(m=len(emergedata.long_data))

    else:
        np.random.seed(7539639)
        # Generate metadata string
        metadata = {'earth'   : estyle,
                    'data'    : datarange,
                    'datafile': 'Emergence_Data_seqnr_2014.txt',
                    'ice'     : istyle,
                    'areas'   : args[0],
                    'ndim'    : ndim,
                    'nwalkers': nwalkers,
                    'nblobs'  : nblobs,
                    'seed'    : 7539639,
                    'corrksip': corrskip,
                    'minned'  : minned,
                    'fitgrav' : fitgrav}
          
        pos = np.random.rand(nwalkers, ndim) * 4
        pos[:,-1] += 8

        # Add starting positions to metadata.
        metadata['pos'] = pos
        # Create metadata string.
        metadatastr = gen_metadatastr(metadata)
            
        # Create (or overwrite) writefile and append metadata
        with open(writefile, 'w') as f:
            f.write(metadatastr)

        lnprob0 = None
        blobs0 = None

        # If we are to save residual covariances, create it here.
        if covkey:
            covMat = OnlineSamplingCovariance(m=len(emergedata.long_data))

    if covkey:
        args += (True,)
    else:
        args += (False,)
        covMat = None

    if fitgrav:
        args += (True,)
    else:
        args += (False,)
 
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args,
                                        threads=16)
    
    try:
        sampleOut(sampler, pos, lnprob0, blobs0, writefile, nsteps,
                    verbose=True, resCov=covMat, resCovDump=1,
                    corrSkip=corrskip)
    except KeyboardInterrupt:
        pass

    if covkey:
        pickle.dump(covMat, open(os.path.splitext(writefile)[0]+'_resCov.p', 'w'), -1)
