#likely need to adjust what needs being imported, 
#for local imports, check which functions are being used and whether there are better functions defined elsewhere

import numpy as np
import os
import pandas as pd
# np.set_printoptions(legacy = '1.25')

#import pandas as pd
#import csv
#import time
#import os
#import multiprocessing
# from collections import Counter
# import pickle

# from . import polarize as pol
# from . import smooth as ks

# left and right indices are inclusive. So interval is [l,r] inclusive of both ends.
def get_intervals(chrName, indName, statesList, posList, mapPosList=None, includeSingle = True):
    '''
    Function to get intervals for a given contig from states and positions. 
    single site intervals can be included or excluded. By default they are included.

    Args:

        chrName (str): Chromosome name.
        indName (str): Individual name.
        statesList (list): List of states at each site for this individual and chromosome.
        posList (list): List of positions.
        mapPosList (list, optional): List of map positions in the same order as posList.

    Returns:
        list: List of intervals.
    '''

    lidx = 0
    ridx = 0
    ivls = []

    if len(statesList) == 0:
        return ivls
    
    while ridx <= len(statesList)-1:
        currentState = statesList[lidx]

        if ridx == len(statesList)-1:
            l = posList[lidx]
            r = posList[ridx]
            mapl = mapPosList[lidx] if mapPosList is not None else None
            mapr = mapPosList[ridx] if mapPosList is not None else None
            iv = Interval(chrName, indName, lidx, ridx, l, r, currentState, mapl=mapl, mapr=mapr)

            if includeSingle == True:
                ivls.append(iv)
            else:
                if r-l>0: 
                    ivls.append(iv)
            break

        if statesList[ridx+1] == currentState:
            ridx += 1
        else:
            l = posList[lidx]
            r = posList[ridx]
            mapl = mapPosList[lidx] if mapPosList is not None else None
            mapr = mapPosList[ridx] if mapPosList is not None else None

            iv = Interval(chrName, indName, lidx, ridx, l, r, currentState, mapl=mapl, mapr=mapr)

            if includeSingle == True:
                ivls.append(iv)
            else:
                if r-l>0: 
                    ivls.append(iv)

            ridx +=1
            lidx=ridx
            
    return ivls




# interval should also add info about (any) sites supporting the interval in addition to the left and right sites defining it
# this should be done in a way to look for gene conversion events later on
class Interval:
    '''
    Represents a genomic interval for a specific individual and chromosome.

    Args:

        chrName (str): Chromosome name.
        indName (str): Individual name.
        idxl (int): Left index (inclusive).
        idxr (int): Right index (inclusive). So slice of state matrix would be [idxl:idxr+1]
        l (float): Left position (physical).
        r (float): Right position (physical).
        state (int): State of the interval.
        mapl (float, optional): Left position on map scale.
        mapr (float, optional): Right position on map scale.

    :ivar str chrName: Chromosome name.
    :ivar str indName: Individual name.
    :ivar int idxl: Left index (inclusive).
    :ivar int idxr: Right index (inclusive). So slice of state matrix would be [idxl:idxr+1]
    :ivar float l: Left position (physical).
    :ivar float r: Right position (physical).
    :ivar float mapl: Left position on map scale.
    :ivar float mapr: Right position on map scale.
    :ivar float span: Physical span of the interval (r-l).
    :ivar float mapspan: Map span of the interval (mapr-mapl).
    :ivar int state: State of the interval.

    '''

    def __init__(self,chrName,indName,idxl,idxr,l,r,state,mapl=None,mapr=None):

        self.chrName = chrName
        self.indName = indName
        self.idxl = idxl
        self.idxr = idxr
        self.l = l
        self.r = r
        self.mapl = mapl
        self.mapr = mapr
        self.span = None if (self.l is None or self.r is None) else (self.r - self.l)
        self.mapspan = None if (self.mapl is None or self.mapr is None) else (self.mapr - self.mapl)
        self.state = state



    def info(self):
        print(f"chr = {self.chrName}, ind = {self.indName}, idxl = {self.idxl}, idxr = {self.idxr}, l = {self.l}, r = {self.r}, mapl = {self.mapl}, mapr = {self.mapr}, span = {self.span}, mapspan = {self.mapspan}, state = {self.state}")
    

    
#the chromosome class contains the haplotype structure of a single chromosome.
# the individual and chromosome are indexedIt contains 
# for a single individual and single chromosome. Maybe 'individualChromsome' would be a better name?
# the individual is simply a list of the intervals (as defined above)
class Contig:
    
    '''
    Represents a contiguous sequence of genomic intervals for a specific individual and chromosome.
    
    Args:
        chrName (str): Chromosome name.
        indName (str): Individual name.
        intervalList (list): List of Interval objects.

    :ivar str chr: Chromosome name.
    :ivar str ind: Individual name.
    :ivar int num_intervals: Number of intervals.
    :ivar list intervals: List of Interval objects.

    '''

    
    
    # individual is a list of intervals pertaining to a single chromosome

    def __init__(self,chrName=None,indName=None,intervalList=None):
  
        self.intervals = intervalList
        if self.intervals is None:
            self.num_intervals = 0
        else:
            self.num_intervals = len(self.intervals)
        self.indName = indName
        self.chrName = chrName

        # self.getZeroIntervals()
        # self.getOneIntervals()
        # self.getTwoIntervals()
        # self.getThreeIntervals()
        
    

    def printIntervals(self,lim=10):
        print("formatting is as follows [leftPosition,rightPosition,state]")
        if lim is None:
            print([[x.l,x.r,x.state] for x in self.intervals]) 
        else:
            print([[x.l,x.r,x.state] for x in self.intervals[0:min(self.num_intervals,lim)]])


    def get_my_intervals_of_state(self,state):
        ivs = []
        for x in self.intervals:
            if x.state == state:
                ivs.append(x)
        return ivs


def pack_interval(interval):
    '''
    Packs an Interval object into a dictionary.
    Args:
        interval (Interval): Interval object to be packed.
    Returns:
        dict: Dictionary representation of the Interval object.
    '''
    return interval.__dict__

def unpack_interval(d):
    '''
    Unpacks a dictionary into an Interval object.
    Args:
        d (dict): Dictionary representation of an Interval object.
    Returns:
        Interval: Unpacked Interval object.
    '''
    blankArgs = [None for _ in range(7)]
    i = Interval(*blankArgs)
    i.__dict__.update(d)
    if not hasattr(i, 'span'):
        i.span = None if (i.l is None or i.r is None) else (i.r - i.l)
    if not hasattr(i, 'mapspan'):
        mapl = getattr(i, 'mapl', None)
        mapr = getattr(i, 'mapr', None)
        i.mapspan = None if (mapl is None or mapr is None) else (mapr - mapl)
    return i

def pack_intervalList(ivl):
    '''
    Packs a list of Interval objects into a list of dictionaries.
    Args:
        ivl (list): List of Interval objects to be packed.
    Returns:
        list: List of dictionary representations of the Interval objects.
    '''
    return [pack_interval(iv) for iv in ivl]

def unpack_intervalList(dlist):
    '''
    Unpacks a list of dictionaries into a list of Interval objects.
    Args:
        dlist (list): List of dictionary representations of Interval objects.
    Returns:
        list: List of unpacked Interval objects.
    '''
    return [unpack_interval(d) for d in dlist]

def pack_contig(contig):
    '''
    Packs a Contig object into a dictionary. This requires 'packing' the list of Interval objects as well.
    Args:
        contig (Contig): Contig object to be packed.
    Returns:
        dict: Dictionary representation of the Contig object.
    '''
    d = contig.__dict__.copy()
    d['intervals'] = pack_intervalList(contig.intervals)
    return d

def unpack_contig(d):
    contig = Contig()

    contig.__dict__.update(d)
    contig.intervals = unpack_intervalList(d['intervals'])
    return contig

def pack_contig_matrix(cArr):
    '''
    Packs a Matrix of Contig objects into a matrix of dictionaries.
    Matrix is (num_chromosomes, num_individuals) in sort order of the diemtype parent object.

    Args:
        cArr (np.array dtype=object): Matrix of Contig objects to be packed.
    Returns:
        list: List of dictionary representations of the Contig objects.
    '''
    
    return [[pack_contig(c) for c in row] for row in cArr]

def unpack_contig_matrix(dArr):
    '''
    Unpacks a Matrix of dictionaries into a matrix of Contig objects.
    Matrix is (num_chromosomes, num_individuals) in sort order of the diemtype parent object.

    Args:
        dArr (list): List of dictionary representations of Contig objects.
    Returns:
        np.array dtype=object: Matrix of unpacked Contig objects.
    '''
    return np.array([[unpack_contig(d) for d in row] for row in dArr], dtype=object)






def build_contig_matrix(diemType,includeSingle = True):
    '''
    Creates a matrix of Contig objects from a DiemType object.
    Matrix is (num_chromosomes, num_individuals) in sort order of the diemtype parent object.

    Args:
        diemType (DiemType): DiemType object from which to create the Contig matrix.
    Returns:
        np.array dtype=object: Matrix of Contig objects.
    '''
    nChrs = len(diemType.chrNames)
    nInds = len(diemType.indNames)

    cArr = np.empty((nChrs, nInds), dtype=object)

    for cIdx in range(nChrs):
        chrName = diemType.chrNames[cIdx]
        mapPosList = diemType.MapBC[cIdx] if diemType.MapBC is not None else None
        for indIdx in range(nInds):
            indName = diemType.indNames[indIdx]
            statesList = diemType.DMBC[cIdx][indIdx]
            posList = diemType.posByChr[cIdx]

            ivl = get_intervals(chrName, indName, statesList, posList, mapPosList=mapPosList, includeSingle=includeSingle)
            contig = Contig(chrName, indName, ivl)
            cArr[cIdx, indIdx] = contig

    return cArr


def export_contigs_to_ind_bed_files(diemType, outputDir):
    '''
    Exports contig intervals to BED files for each individual.

    Args:
        diemType (DiemType): DiemType object containing contig data.
        outputDir (str): Directory where BED files will be saved.
    '''

    nChrs = len(diemType.chrNames)
    nInds = len(diemType.indNames)

    # Ensure output directory exists
    os.makedirs(outputDir, exist_ok=True)

    for indIdx in range(nInds):

        indName = diemType.indNames[indIdx]
        bedFilePath = os.path.join(outputDir, f"{indName}_contigs.bed")
        data = []

        with open(bedFilePath, 'w') as bedFile:
            for cIdx in range(nChrs):
                thisContig = diemType.contigMatrix[cIdx][indIdx]
            
                for ivl in thisContig.intervals:
                    data.append([ivl.chrName, ivl.l-1, ivl.r-1, ivl.state])  # BED format is 0-based start, 1-based end
        dfInd = pd.DataFrame(data, columns=["chrom", "start", "end", "state"])
        dfInd.to_csv(bedFilePath, sep="\t", header=True, index=False)


# This version does 'U' for state 0, and '0','1','2' for states 1,2,3
# def export_contigs_to_ind_bed_files(diemType, outputDir):
#     '''
#     Exports contig intervals to BED files for each individual.

#     Args:
#         diemType (DiemType): DiemType object containing contig data.
#         outputDir (str): Directory where BED files will be saved.
#     '''

#     nChrs = len(diemType.chrNames)
#     nInds = len(diemType.indNames)

#     # Ensure output directory exists
#     os.makedirs(outputDir, exist_ok=True)

#     for indIdx in range(nInds):

#         indName = diemType.indNames[indIdx]
#         bedFilePath = os.path.join(outputDir, f"{indName}_contigs.bed")
#         data = []

#         with open(bedFilePath, 'w') as bedFile:
#             for cIdx in range(nChrs):
#                 thisContig = diemType.contigMatrix[cIdx][indIdx]
#                 # change diemtype state vaules of 0,1,2,3 to 'U', '0', '1', '2'
#                 for ivl in thisContig.intervals:
#                     state = ivl.state
#                     if state == 0:
#                         state = 'U'
#                     else:
#                         state = str(state - 1)  # Convert 1,2,3 to '0','1','2'
#                     data.append([ivl.chrName, ivl.l-1, ivl.r-1, state])  # BED format is 0-based start, 1-based end
#         dfInd = pd.DataFrame(data, columns=["chrom", "start", "end", "state"])
#         dfInd.to_csv(bedFilePath, sep="\t", header=True, index=False)