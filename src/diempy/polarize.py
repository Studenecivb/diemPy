import numpy as np
import multiprocessing
from multiprocessing import shared_memory
import time
#import numba
import pandas as pd
import hashlib



#my original code, faster than what copilot suggested
### move this to 'input' submodule or 'diemtype' submodule?
def hapStrings_to_hapIndices(x):
    """
    Converts a list of haplotype strings to a NumPy array of haplotype indices.

    Each input string is expected to contain numeric characters representing haplotypes,
    with underscores ('_') indicating missing data. Underscores are
    replaced with '3', and all characters are converted to integers, incremented by 1.
    The resulting array is post-processed so that any value equal to 4 is set to 0. I.e. 0 = missing data, 1 = hom1,  2= het, 3 = homAlt

    Parameters
    ----------
    x : list of str
        List of haplotype strings, each of equal length.

    Returns
    -------
    numpy.ndarray
        2D array of shape (len(x), len(x[0])) with dtype np.int8, containing haplotype indices.
        Values are in the range [0, 4], where 0 represents processed underscores.
    """

    # re-write this code so that it takes 0,1,2 as the input types 
    # and 'anything else' like '_' or 'U' as missing data

    res = np.ones((len(x),len(x[0])), dtype=np.int8)
    for idx1, s in enumerate(x):
        t = s.replace('_','3')
        tt = np.array(list(t),dtype = np.int8)
        tt = (tt+1)
        res[idx1]=tt
    res[res==4]=0
    return res


# updated version, faster, from copilot
### again, move to 'input' submodule or to 'diemtype' submodule?
def hapStateIndices_to_hapMatrix(hapIndices):
    """
    Converts haplotype state indices to a 3D matrix representation.

    Args:
        hapIndices (np.ndarray): 2D array of shape (nMarkers, nInds) with state indices.

    Returns:
        np.ndarray: 3D array of shape (nMarkers, nInds, 4) with counts for each state.
    """
    nMarkers, nInds = hapIndices.shape
    Marr = np.zeros((nMarkers, nInds, 4), dtype=np.int8)
    # Vectorized assignment
    rows, cols = np.nonzero(hapIndices >= 0)
    states = hapIndices[rows, cols]
    Marr[rows, cols, states] += 1
    return Marr

def initialize_given_polarity(M_, initPolarity):
    '''
    Initialize the polarity of the M array to the given initPolarity.
    Flips the states for indices where initPolarity is 1.
    '''
    M = M_.copy()
    flip_idx = np.where(initPolarity == 1)[0]
    for idx in flip_idx:
        M[idx][:] = M[idx][:,[0,3,2,1]]
    return M

def initialize_test_polarity(M_):
    '''
    Initialize the polarity of the M array to alternating 0s and 1s.
    Flips the states for odd indices.
    '''
    initPolarity = np.arange(len(M_)) % 2
    initPolarity = initPolarity.astype(np.int8)
    # Only copy if you need to preserve the original
    M = M_.copy()
    flip_idx = np.where(initPolarity == 1)[0]
    for idx in flip_idx:
        M[idx][:] = M[idx][:,[0,3,2,1]]
    return initPolarity, M


def initialize_random_polarity(M_):
    '''
    Randomly polarize the M array and flip the states in the M array accordingly.
    '''
    initRandomPolarity = np.random.choice([0,1],len(M_))
    initRandomPolarity = initRandomPolarity.astype(np.int8)
    M = M_.copy()
    flip_idx = np.where(initRandomPolarity == 1)[0]
    for idx in flip_idx:
        M[idx][:] = M[idx][:,[0,3,2,1]]
    return initRandomPolarity, M


def get_hybrid_index(A4):
    #IA44 is the matrix of (nHaps x statecounts)
    nHaps = len(A4)
    HIarr = np.zeros(nHaps)
    for idx,counts in enumerate(A4):
        hiNum = counts[1]*0 + counts[2]*1 + counts[3]*2
        hiDenom = 2*(counts[1] + counts[2] + counts[3])
        hi = hiNum/hiDenom
        HIarr[idx] = hi
    return HIarr

def rescale_hybrid_index(hiArr):
    rescaled = np.zeros(len(hiArr))
    hmin = min(hiArr)
    hmax = max(hiArr)
    if hmax == hmin:
        rescaled[:] = 0.5
    else:
        for idx,hi in enumerate(hiArr):
            rescaled[idx] = (hi-hmin)/(hmax-hmin)
    return rescaled

# def get_error_rate(I4):
#     nHaps = len(I4)
#     ErrorArr = np.zeros(nHaps)
#     for idx, counts in enumerate(I4):
#         eNum = counts[0]
#         eDenom = sum(counts)
#         e = eNum/eDenom
#         ErrorArr[idx] = e
#     return ErrorArr

def get_center(HIs):

    # HIs is an array of rescaled hybrid indices
    hiSorted = np.sort(HIs);
    hiDiffs = hiSorted[1:] - hiSorted[0:-1]
    hiDiffVals = [[float(a),float(b)] for a,b in list(zip(hiSorted[0:-1],hiSorted[1:]))]
    idxMaxDiff = np.argmax(hiDiffs)
    center = (hiDiffVals[idxMaxDiff][0] + hiDiffVals[idxMaxDiff][1])/2
    #center = (hiDiffVals[idxMaxDiff][0] - hiDiffVals[idxMaxDiff][1])/2
    return center

def get_hi_beta_weights(HIs):
    c = get_center(HIs)
    betaArr = np.zeros(len(HIs))
    for idx, hi in enumerate(HIs):
        if (hi-c) > (c- hi):
            betaArr[idx] = (hi-c)/(1-c)
        else:
            betaArr[idx] = (c-hi)/c
    return c, betaArr

def get_I4_Ideal(I4,rescaledHybridIndices,center):
    # Here, the I4 passed already has the Zeta counts adjusted
    # so the nMarkersListScaled accounts for the (n+4*Zeta) bit.
    idealMarker = np.zeros((len(I4),4),dtype = int)
    nMarkersListScaled = [sum(x) for x in I4] #here, this could maybe differ for each individual because of ploidy?
    for idx,hi in enumerate(rescaledHybridIndices):
        if hi <= center:
            idealMarker[idx][1] = 1
        else:
            idealMarker[idx][3] = 1
    I4Ideal = idealMarker
    for idx in range(len(I4)):
        I4Ideal[idx,:] = I4Ideal[idx,:]*nMarkersListScaled[idx]
    return I4Ideal

def get_V_matrix(I4Data,I4Ideal,epsilon,betaArr):
    dataTerm = ((1 -epsilon*betaArr)*I4Data.transpose()).transpose()
    #print(dataTerm)
    idealTerm = (epsilon*betaArr*I4Ideal.transpose()).transpose()
    #print(idealTerm)
    V = dataTerm + idealTerm
    #print(V)
    return V


# my faster version
def MArray_to_stateMatrix(M):
    nMarkers = M.shape[0]
    nInds = M.shape[1]
    x= np.nonzero(M)[2]
    sm = x.reshape(nMarkers,nInds).transpose().astype(M.dtype).copy()
    return sm

# my original version
# def MArray_to_stateMatrix(M):
#     stateMatrix = np.zeros([len(M),len(M[0])],dtype=np.int8).transpose()
#     for markerIdx,thisMarker in enumerate(M):
#         states = np.where(thisMarker==1)[1]
#         stateMatrix[:,markerIdx] = states
#         # for indIdx,state in np.argwhere(thisMarker==1):
#         #     stateMatrix[indIdx][markerIdx] = state
#     return stateMatrix


def stateMatrix_to_MArray_old(SM):
    nMarkers = SM.shape[1]
    nInds = SM.shape[0]
    Marr = np.zeros((nMarkers,nInds,4),dtype=np.int8)
    for markerIdx in range(nMarkers):
        for indIdx in range(nInds):
            state = SM[indIdx][markerIdx]
            Marr[markerIdx][indIdx][state] = 1
    return Marr

def stateMatrix_to_MArray(SM): #eye method suggested by AI
    """
    Ultra-fast version using identity matrix trick.
    This is likely the fastest approach.
    """
    nInds, nMarkers = SM.shape
    
    # Use identity matrix to create one-hot encoding
    eye = np.eye(4, dtype=np.int8)
    
    # Direct indexing with broadcasting
    Marr = eye[SM].transpose(1, 0, 2)
    
    return Marr



def em_worker(lims, MBC_cat_shape, polBC_cat_shape, DIBC_cat_shape, Support_cat_shape, shm_MBC_cat_name, shm_polBC_cat_name, shm_DIBC_cat_name, shm_Support_cat_name, LP4):
    '''
    Worker function for the EM algorithm to polarize the genome in parallel.

    Args:
        lims (tuple): Tuple containing the start and end indices for the chunk of data to process.
        MBC_cat_shape (tuple): Shape of the MBC_cat array.
        polBC_cat_shape (tuple): Shape of the polBC_cat array.
        DIBC_cat_shape (tuple): Shape of the DIBC_cat array.
        shm_MBC_cat_name (str): Name of the shared memory block for MBC_cat.
        shm_polBC_cat_name (str): Name of the shared memory block for polBC_cat.
        shm_DIBC_cat_name (str): Name of the shared memory block for DIBC_cat.
        LP4 (np.ndarray): Array used for calculating the likelihood.

    Returns:
        None: The function modifies the shared memory arrays in place.      
    '''

    # Attach to shared memory
    shm_MBC_cat = shared_memory.SharedMemory(name=shm_MBC_cat_name)
    shm_polBC_cat = shared_memory.SharedMemory(name=shm_polBC_cat_name)
    shm_DIBC_cat = shared_memory.SharedMemory(name=shm_DIBC_cat_name)
    shm_Support_cat = shared_memory.SharedMemory(name=shm_Support_cat_name)

     # Create NumPy arrays backed by shared memory
    MBC_cat = np.ndarray(MBC_cat_shape, dtype=np.int8, buffer=shm_MBC_cat.buf)
    polBC_cat = np.ndarray(polBC_cat_shape, dtype=np.int8, buffer=shm_polBC_cat.buf)
    DIBC_cat = np.ndarray(DIBC_cat_shape, dtype=np.float64, buffer=shm_DIBC_cat.buf)
    Support_cat = np.ndarray(Support_cat_shape, dtype=np.float64, buffer=shm_Support_cat.buf)

    # flipDict = {}
    # l = lims[0]
    # r = lims[1]

    # for idx in range(l,r,1):
    #     p = polBC_cat[idx]
    #     m = MBC_cat[idx]
    #     key = (m.tobytes(), p)
    #     if key in flipDict:
    #         newM, newP, newDI, newS = flipDict[key]
    #         MBC_cat[idx] = newM
    #         polBC_cat[idx] = newP
    #         DIBC_cat[idx] = newDI
    #         Support_cat[idx] = newS
    #     else:
    #         mf = m[:, [0, 3, 2, 1]]
    #         mLike = np.sum(m * LP4)
    #         mfLike = np.sum(mf * LP4)
    #         if mfLike > mLike:
    #             MBC_cat[idx] = mf
    #             polBC_cat[idx] = 1 - p
    #         diagnosticIndex = max(mLike, mfLike)
    #         DIBC_cat[idx] = diagnosticIndex
    #         Support_cat[idx] = diagnosticIndex - min(mLike, mfLike)
    #         flipDict[key] = [MBC_cat[idx], polBC_cat[idx], DIBC_cat[idx], Support_cat[idx]]

    l = lims[0]
    r = lims[1]
    for idx in range(l,r,1):
        p = polBC_cat[idx]
        m = MBC_cat[idx]
        
        mf = m[:, [0, 3, 2, 1]]
        mLike = np.sum(m * LP4)
        mfLike = np.sum(mf * LP4)
        if mfLike > mLike:
            MBC_cat[idx] = mf
            polBC_cat[idx] = 1 - p
        diagnosticIndex = max(mLike, mfLike)
        DIBC_cat[idx] = diagnosticIndex
        Support_cat[idx] = diagnosticIndex - min(mLike, mfLike)


    shm_MBC_cat.close()
    shm_polBC_cat.close()
    shm_DIBC_cat.close()
    shm_Support_cat.close()


# this is the parallel version.  On the (maybe too small) test data, it is about: 
# 1.6x faster than the single-core version with n=2 cores
# 2.0x faster with with n=3 cores
# 2.3x faster with n=4 cores
# 2.0x faster with with n=5
# 2.0x faster with with n= 8 
# this could be an issue of the parallel overhead outweighing the benefits of more cores
# on larger data, the speedup should be more pronounced
def run_em_parallel(
        initMBC, initPolBC, ploidyBC,sitesExcludedByChr = None,individualsExcluded = None, maxItt =500, epsilon = 0.99999, nCPUs = None):
    # should sites be specified as 'included' or 'excluded'
    # per-scaffoled stuff, inversions, 
    ''' 
    Run the EM algorithm to Polarize the genome.

    Args:
        initMBC (List[np.ndarray]): Initial Markov array of shape (nMarkers, nIndividuals, 4).
        initPolBC (List[np.ndarray]): Initial polarity array of shape (nMarkers,).
        ploidyBC (List[np.ndarray]): List of ploidy arrays for each chromosome.
        sitesExcludedByChr (List[np.ndarray], optional): List of site indices to exclude for each chromosome. Defaults to None. If not None, it must be of length # chromosomes and if a chromosome has no exclusions,should be None.
        individualsExcluded (np.ndarray, optional): Array of individual indices to exclude. Defaults to None
        maxItt (int, optional): Maximum number of iterations. Defaults to 500.
        epsilon (float, optional): Convergence threshold. Defaults to 0.99.
        nCPUs (int, optional): Number of CPUs to use. Defaults to None.

    Returns:
        finalMBC (List[np.ndarray]): Final Marker M array after polarization.
        finalPolBC (List[np.ndarray]): Final polarity array after polarization.
        finalDIBC (List[np.ndarray]): Final diagnostic index array after polarization.
    '''

    print("starting")
    # although diemType polarize calls this function, and diemType.polarize has nCPUs as an argument, I am keeping this here for now.
    if nCPUs is None:
        nCPUs = multiprocessing.cpu_count()
    print("using ", nCPUs, " CPUs")
    # this subsets individuals for the polarizing
    # subsetting requires a final flipping of the initMBC using the new polarity
    
    individualsIncluded = np.arange(len(initMBC[0][0]))
    if individualsExcluded is not None:
        individualsIncluded = np.setdiff1d(individualsIncluded,individualsExcluded)

    sitesIncludedBC = []
    for idx in range(len(initMBC)):
        included = np.arange(len(initMBC[idx]))
        if sitesExcludedByChr is not None:
            if sitesExcludedByChr[idx] is not None:
                included = np.setdiff1d(included,sitesExcludedByChr[idx])
        sitesIncludedBC.append(included)
    
    MBC = [x.copy()[:,individualsIncluded,:] for x in initMBC ]
    ploidyBC = [x[individualsIncluded] for x in ploidyBC]

    polBC = [x.copy() for x in initPolBC]
    nChroms = len(MBC)
    nMarkersBC = [len(x) for x in MBC]
    nIndividuals = MBC[0].shape[1]  
    totalMarkers = sum(nMarkersBC)
    nStates = MBC[0].shape[2] #this is 4

    # setting up the DIBC list of arrays, may not need any more
    DIBC = []
    for v in nMarkersBC:
        DIBC.append(np.zeros(v,dtype=np.double))

    SupportBC = []
    for v in nMarkersBC:
        SupportBC.append(np.zeros(v,dtype=np.double))
    
    #note: got this  shared memory approach from AI chat... 
    MBC_cat_shape = (totalMarkers,nIndividuals,nStates)
    Pol_cat_shape = (totalMarkers,)
    DIBC_cat_shape = (totalMarkers,)
    Support_cat_shape = (totalMarkers,)

    shm_MBC_cat = shared_memory.SharedMemory(create=True, size=np.prod(MBC_cat_shape)*np.dtype(MBC[0].dtype).itemsize)
    shm_Pol_cat = shared_memory.SharedMemory(create=True, size=np.prod(Pol_cat_shape)*np.dtype(polBC[0].dtype).itemsize)
    shm_DIBC_cat = shared_memory.SharedMemory(create=True, size=np.prod(DIBC_cat_shape)*np.dtype(DIBC[0].dtype).itemsize)
    shm_Support_cat = shared_memory.SharedMemory(create=True, size=np.prod(Support_cat_shape)*np.dtype(SupportBC[0].dtype).itemsize)

    MBC_cat_shared = np.ndarray(MBC_cat_shape, dtype=MBC[0].dtype, buffer=shm_MBC_cat.buf)
    Pol_cat_shared = np.ndarray(Pol_cat_shape, dtype=polBC[0].dtype, buffer=shm_Pol_cat.buf)
    DIBC_cat_shared = np.ndarray(DIBC_cat_shape, dtype=DIBC[0].dtype, buffer=shm_DIBC_cat.buf)
    Support_cat_shared = np.ndarray(Support_cat_shape, dtype=SupportBC[0].dtype, buffer=shm_Support_cat.buf)

    #initializing the values in the shared memory arrays
    chrLims = []
    startIdx = 0
    for idx in range(nChroms):
        nM = nMarkersBC[idx]
        chrLims.append((startIdx,startIdx+nM))
        MBC_cat_shared[startIdx:startIdx+nM,:,:] = MBC[idx]
        Pol_cat_shared[startIdx:startIdx+nM] = polBC[idx]
        DIBC_cat_shared[startIdx:startIdx+nM] = DIBC[idx]
        Support_cat_shared[startIdx:startIdx+nM] = SupportBC[idx]
        startIdx += nM


    lims = []
    for chunk in np.array_split(np.arange(totalMarkers),nCPUs):
        l = chunk[0]
        r = chunk[-1]+1 
        lims.append([int(l),int(r)])
    jobList = lims

    history = {}
    polarityFound = False
    historyIndex = None

    for i in range(maxItt+1):

        if i == maxItt:
            print()
            print("CAUTION! No polarity found after ", maxItt, " iterations")
            print("returning the state after the final iteration")
            print("consider increasing maxItt or checking the input data")
            print()
            
            break


        # More efficient polarity counting using shared memory directly
        np0 = np.sum(Pol_cat_shared == 0)
        np1 = np.sum(Pol_cat_shared == 1)
        print('after ',i,' EM steps ','  num 0 polarity = ', np0, '  num 1 polarity = ', np1)

        start = time.time()
        if polarityFound:
            print("polarity found")
            print('the state after ', i, ' iterations is the same as the state after ', historyIndex, ' iterations')
            #print("state after match found to step ", historyIndex)
            break

        # Memory optimized: use views instead of copies for iteration calculations
        # Only copy when actually needed for final result
        MBC = [MBC_cat_shared[lim[0]:lim[1],:,:] for lim in chrLims]
        polBC = [Pol_cat_shared[lim[0]:lim[1]] for lim in chrLims]   

        pol_bytes = Pol_cat_shared.tobytes()
        pol_hash = hashlib.sha256(pol_bytes).hexdigest()
        history[pol_hash] = i
        
    


        # getting I4 and A4 over all chromosomes - memory efficient single pass
        # taking care to subset sites we consider for barrier
        I4All = None
        A4All = None
        for idx, a in enumerate(MBC):
            I4_chr = np.sum(a[sitesIncludedBC[idx],:,:], axis=0)
            A4_chr = np.dot(np.diag(ploidyBC[idx]), I4_chr)
            
            if I4All is None:
                I4All = I4_chr.copy()
                A4All = A4_chr.copy()
            else:
                I4All += I4_chr
                A4All += A4_chr

        m = np.min(I4All)
        if m > 1:
            zeta = -1
        elif m==1:
            zeta = 0
        else:
            zeta = 1
        I4AllZeta = I4All + zeta
        nMarkers = sum(I4All[0])
        #zPerMarker = zeta/nMarkers
        A4AllZeta = A4All+zeta
        A4AllZeta
        # getting HIs, finding barrier center, and the L4 matrix
        hybridIndices = get_hybrid_index(A4AllZeta)
        rescaledHybridIndices = rescale_hybrid_index(hybridIndices)

        center, betaArr = get_hi_beta_weights(rescaledHybridIndices)
        I4Ideal = get_I4_Ideal(I4AllZeta,rescaledHybridIndices,center)

        V4 = get_V_matrix(I4AllZeta, I4Ideal,epsilon,betaArr)
        P4 = V4/(nMarkers+3*zeta)
        LP4 = np.log(P4)

        # EM step: getting the new polarity
        startFlipTime = time.time() 

        with multiprocessing.Pool(nCPUs) as pool:
            pool.starmap(
                em_worker,
                [(lim, MBC_cat_shape, Pol_cat_shape, DIBC_cat_shape, Support_cat_shape, shm_MBC_cat.name, shm_Pol_cat.name, shm_DIBC_cat.name, shm_Support_cat.name, LP4)
                 for lim in jobList]
            )
        # uncomment this to see time for flipping step
        if i == 0:
            print('time to flip: ', time.time() - startFlipTime)

        k = Pol_cat_shared.tobytes()
        k = hashlib.sha256(k).hexdigest()

        kAlt = (~Pol_cat_shared+2).tobytes()
        kAlt = hashlib.sha256(kAlt).hexdigest()

        if k in history:
            polarityFound = True
            historyIndex = history[k]
            # print("polarity found")
            # print("history index is ", history[k])
        if kAlt in history:
            polarityFound = True
            historyIndex = history[kAlt]
            # print("polarity found (alt)")
            # print("history index is ", history[kAlt])
        
        # for testing:
        
        # print("time for iteration in seconds: ", time.time() - start) 
        # return MBC,polBC,DIBC # for testing only !!! uncomment to only do one round of EM
    
    # have to re-copy these once more (particularly DIBC) from shared memory to return the correct stuff
    # again, see note above about copying vs pointing to the shared memory objects.
    # key here is that I DO think this needs to be copied at the last step in order to close the shared mem afterward
    MBC = [MBC_cat_shared[lim[0]:lim[1],:,:].copy() for lim in chrLims]
    polBC = [Pol_cat_shared[lim[0]:lim[1]].copy() for lim in chrLims]
    DIBC = [DIBC_cat_shared[lim[0]:lim[1]].copy() for lim in chrLims]
    SupportBC = [Support_cat_shared[lim[0]:lim[1]].copy() for lim in chrLims]

    shm_MBC_cat.close()
    shm_Pol_cat.close()
    shm_DIBC_cat.close()
    shm_MBC_cat.unlink()
    shm_Pol_cat.unlink()
    shm_DIBC_cat.unlink()
    shm_Support_cat.close()
    shm_Support_cat.unlink()


    if individualsExcluded is None:
        return MBC,polBC,DIBC,SupportBC
    else:
        # need to flip the init polarity of all individuals according to the new polarity
        FinalMBC = [x.copy() for x in initMBC]
        for idxChr in range(nChroms):
            for idxMarker in range(nMarkersBC[idxChr]):
                pinit = initPolBC[idxChr][idxMarker]
                p = polBC[idxChr][idxMarker]
                if pinit != p:
                    FinalMBC[idxChr][idxMarker][:] = FinalMBC[idxChr][idxMarker][:,[0,3,2,1]]
        return FinalMBC,polBC,DIBC,SupportBC
    



# this is the original single-core version of the EM algorithm
# it may be more memory efficient, as it does not require the shared memory overhead... but need to check this
# if user requests only one core from the diplotype.polarize function (the default value), this function is used 
def run_em_linear(
        initMBC, initPolBC, ploidyBC,sitesExcludedByChr = None,individualsExcluded = None, maxItt =500, epsilon = 0.99999):
    
    ''' Run the EM algorithm in serial to Polarize the genome.

    Args:
        initMBC (List[np.ndarray]): Initial Markov array of shape (nMarkers, nIndividuals, 4).
        initPolBC (List[np.ndarray]): Initial polarity array of shape (nMarkers,).
        ploidyBC (List[np.ndarray]): List of ploidy arrays for each chromosome.
        sitesExcludedByChr (List[np.ndarray], optional): List of site indices to exclude for each chromosome. Defaults to None. If not None, it must be of length # chromosomes and if a chromosome has no exclusions,should be None.
        individualsExcluded (List[int], optional): List of individual indices to exclude. Defaults to None
        sitesBC (List[np.ndarray], optional): List of site positions for each chromosome. Defaults to None.
        maxItt (int, optional): Maximum number of iterations. Defaults to 500.
        epsilon (float, optional): Convergence threshold. Defaults to 0.99.
        nCPUs (int, optional): Number of CPUs to use. Defaults to None.
    
    Returns:
        (tuple): tuple containing:
            finalMBC (List[np.ndarray]): Final Marker M array after polarization.
            finalPolBC (List[np.ndarray]): Final polarity array after polarization.
            finalDIBC (List[np.ndarray]): Final diagnostic index array after polarization.  
    '''
    print("starting")

    individualsIncluded = np.arange(len(initMBC[0][0]))
    if individualsExcluded is not None:
        individualsIncluded = np.setdiff1d(individualsIncluded,individualsExcluded)

    sitesIncludedBC = []
    for idx in range(len(initMBC)):
        included = np.arange(len(initMBC[idx]))
        if sitesExcludedByChr is not None:
            if sitesExcludedByChr[idx] is not None:
                included = np.setdiff1d(included,sitesExcludedByChr[idx])
        sitesIncludedBC.append(included)
     
    MBC = [x.copy()[:,individualsIncluded,:] for x in initMBC ]
    ploidyBC = [x[individualsIncluded] for x in ploidyBC]
    print(' num individuals in MBC is ', MBC[0].shape[1])   

    polBC = [x.copy() for x in initPolBC]

    nChroms = len(MBC)
    nMarkersBC = [len(x) for x in MBC]
    #print(nMarkersBC)

    DIBC = []
    for v in nMarkersBC:
        DIBC.append(np.zeros(v,dtype=np.double))

    SupportBC = []
    for v in nMarkersBC:
        SupportBC.append(np.zeros(v,dtype=np.double))



    history = {}
    polarityFound = False

    for i in range(maxItt):
        start = time.time()
        if polarityFound:
            print("polarity found")
            break

        
        # Create hash without unnecessary copies
        pol_bytes = np.hstack(polBC).tobytes()
        pol_hash = hashlib.sha256(pol_bytes).hexdigest()
        history[pol_hash] = i
        
        

        np0 = sum([sum(x==0) for x in polBC])
        np1 = sum([sum(x==1) for x in polBC])
        print('iteration ',i,'  num 0 polarity = ', np0, '  num 1 polarity = ', np1)

        # getting I4 and I4z over all chromosomes
        # taking care to subset sites we consider for barrier
        I4BC = []
        for idx,a in enumerate(MBC):
            I4BC.append(np.sum(a[sitesIncludedBC[idx],:,:],axis=0))
        I4All = np.sum(I4BC,axis=0)


    
        m = np.min(I4All)
        if m > 1:
            zeta = -1
        elif m==1:
            zeta = 0
        else:
            zeta = 1
        I4AllZeta = I4All + zeta
  

    
        #getting A4 and A4z over all chromosomes - memory efficient version
        A4All = np.zeros_like(I4All)
        for I, ploidy in zip(I4BC, ploidyBC):
            A4All += np.dot(np.diag(ploidy), I)
        nMarkers = sum(I4All[0])
        #zPerMarker = zeta/nMarkers
        A4AllZeta = A4All+zeta
        A4AllZeta
        # getting HIs, finding barrier center, and the L4 matrix
        hybridIndices = get_hybrid_index(A4AllZeta)
        rescaledHybridIndices = rescale_hybrid_index(hybridIndices)

        center, betaArr = get_hi_beta_weights(rescaledHybridIndices)
        I4Ideal = get_I4_Ideal(I4AllZeta,rescaledHybridIndices,center)

        V4 = get_V_matrix(I4AllZeta, I4Ideal,epsilon,betaArr)
        P4 = V4/(nMarkers+3*zeta)
        LP4 = np.log(P4)

        # EM step: getting the new polarity

        
        for idxChr in range(nChroms):
            for idxMarker in range(nMarkersBC[idxChr]):
                p = polBC[idxChr][idxMarker]
                m = MBC[idxChr][idxMarker]

   
                mf = m[:,[0,3,2,1]]
                mLike = np.sum(m*LP4)
                mfLike = np.sum(mf*LP4)
                if mfLike > mLike:
                    MBC[idxChr][idxMarker] = mf
                    if p ==0:
                        polBC[idxChr][idxMarker] = 1
                    else:
                        polBC[idxChr][idxMarker] = 0
                diagnosticIndex = max(mLike,mfLike)
                support = diagnosticIndex - min(mLike,mfLike)
                SupportBC[idxChr][idxMarker] = support
                DIBC[idxChr][idxMarker] = diagnosticIndex

        # flipDict = {}
        # for idxChr in range(nChroms):
        #     for idxMarker in range(nMarkersBC[idxChr]):
        #         p = polBC[idxChr][idxMarker]
        #         m = MBC[idxChr][idxMarker]

        #         # More memory efficient key using hash instead of full bytes
        #         key = (hash(m.tobytes()),p)
        #         if key in flipDict:
        #             newM, newP, newDI, newS = flipDict[key]
        #             MBC[idxChr][idxMarker] = newM
        #             polBC[idxChr][idxMarker] = newP
        #             DIBC[idxChr][idxMarker] = newDI
        #             SupportBC[idxChr][idxMarker] = newS
        #         else:
        #             mf = m[:,[0,3,2,1]]
        #             mLike = np.sum(m*LP4)
        #             mfLike = np.sum(mf*LP4)
        #             if mfLike > mLike:
        #                 MBC[idxChr][idxMarker] = mf
        #                 if p ==0:
        #                     polBC[idxChr][idxMarker] = 1
        #                 else:
        #                     polBC[idxChr][idxMarker] = 0
        #             diagnosticIndex = max(mLike,mfLike)
        #             support = diagnosticIndex - min(mLike,mfLike)
        #             SupportBC[idxChr][idxMarker] = support
        #             DIBC[idxChr][idxMarker] = diagnosticIndex
        #             flipDict[key] = [
        #                 MBC[idxChr][idxMarker],
        #                 polBC[idxChr][idxMarker],
        #                 DIBC[idxChr][idxMarker],
        #                 SupportBC[idxChr][idxMarker]
        #             ]
        
        # check if polarity has been seen before, and if so, stop
        polStack = np.hstack(polBC)
        k = polStack.tobytes()
        k = hashlib.sha256(k).hexdigest()

        kAlt = (~polStack+2).tobytes()
        kAlt = hashlib.sha256(kAlt).hexdigest()


        if k in history:
            polarityFound = True
            print("polarity found")
        if kAlt in history:
            polarityFound = True
            print("polarity found (alt)")
        
        print("time for iteration in seconds: ", time.time() - start)
        # for testing:
        
        #print("time for iteration in seconds: ", time.time() - start) !!! uncoment to check run time
        # return MBC,polBC,DIBC # for testing only !!! uncomment to only do one round of EM


    if individualsExcluded is None:
        return MBC,polBC,DIBC,SupportBC
    else:
        # need to flip the init polarity of all individuals according to the new polarity
        FinalMBC = [x.copy() for x in initMBC]
        for idxChr in range(nChroms):
            for idxMarker in range(nMarkersBC[idxChr]):
                pinit = initPolBC[idxChr][idxMarker]
                p = polBC[idxChr][idxMarker]
                if pinit != p:
                    FinalMBC[idxChr][idxMarker][:] = FinalMBC[idxChr][idxMarker][:,[0,3,2,1]]
        return FinalMBC,polBC,DIBC,SupportBC