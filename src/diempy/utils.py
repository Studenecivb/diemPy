import numpy as np
#np.set_printoptions(legacy = '1.25')

import pandas as pd
import csv
import time
import os
import multiprocessing
from collections import Counter
import pickle


from . import polarize as pol 
from . import smooth as ks


import matplotlib.pyplot as plt
from matplotlib import colors


# copied from https://github.com/Studenecivb/CarpePy/blob/main/carpepy/core_functions.py#L374
# to match the colors across implementations
diemColors4 = [
    'white',
    colors.to_hex((128/255, 0, 128/255)),  # RGBColor[128/255, 0, 128/255] - Purple
    colors.to_hex((255/255, 229/255, 0)),  # RGBColor[255/255, 229/255, 0] - Yellow
    colors.to_hex((0, 128/255, 128/255))   # RGBColor[0, 128/255, 128/255] - Teal
]
diemColors5 = [
    'LightGray',
    'white',
    colors.to_hex((128/255, 0, 128/255)),  # RGBColor[128/255, 0, 128/255] - Purple
    colors.to_hex((255/255, 229/255, 0)),  # RGBColor[255/255, 229/255, 0] - Yellow
    colors.to_hex((0, 128/255, 128/255))   # RGBColor[0, 128/255, 128/255] - Teal
]
bounds5 = [-1.5,-0.5,.5,1.5,2.5,3.5]

AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 11
AXIS_FONTWEIGHT = 'bold'


'''
Some utility functions for working with the data and results
'''


def plot_painting(diemMatrix,names=None,figSize = None,outputPath=None):
    '''
     Plot a painting of a diemMatrix. that is, an element of diemtype.DMBC, i.e., a diem matrix for a single chromosome.
     
    Args:
        diemMatrix: A diem matrix for a single chromosome (element of diemtype.DMBC)
        names: Optional list of names for each individual (to label y-axis)
        figSize: Optional tuple specifying figure size (width, height)
        outputPath: Optional path to save the figure. Extension determines format (e.g., .png, .pdf)
        names: Optional list of names for each individual (to label y-axis)
     '''
    # for a single chromosome in the 'stateMatrixByChromsome'

    if figSize is None:
        figSize = (12,4)
    mycmap = colors.ListedColormap( diemColors4)
    bounds = [-.5,.5,1.5,2.5,3.5]
    norm = colors.BoundaryNorm(bounds,mycmap.N)
    
    fig, ax = plt.subplots(figsize=figSize)
    ax.pcolormesh(diemMatrix, cmap = mycmap, norm=norm)
    
    if names is not None:
        # Set y-tick positions at the center of each row
        ax.set_yticks(np.arange(len(names)) + 0.5)
        ax.set_yticklabels(names)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    _style_axis_text(ax)
    
    if outputPath is not None:
        plt.savefig(outputPath,bbox_inches='tight')


    plt.show()


def plot_painting_with_positions(diemMatrix, positions,markerWidth=25,figSize = None,outputPath=None,names=None):
    '''
    Plot a painting of a diemMatrix with proper spacing based on physical positions,
    showing blank spaces between markers.
    
    Args:
        diemMatrix: A diem matrix for a single chromosome (element of diemtype.DMBC)
        positions: Array of physical positions for each marker (element of diemtype.posByChr)
        markerWidth: Half-width of each marker's visual representation (in base pairs)
        figSize: Optional tuple specifying figure size (width, height)
        outputPath: Optional path to save the figure. Extension determines format (e.g., .png, .pdf)
        names: Optional list of names for each individual (to label y-axis
    '''
    # for a single chromosome in the 'stateMatrixByChromsome'

    if figSize is None:
        figSize = (12,4)
    mycmap = colors.ListedColormap(diemColors4)
    bounds = [-.5,.5,1.5,2.5,3.5]
    norm = colors.BoundaryNorm(bounds,mycmap.N)
    
    fig, ax = plt.subplots(figsize=figSize)
    
    # Each marker gets a narrow bin (2 bp wide: 1 bp left and right of position)
    # Gaps between markers will show as blank space
    x_edges = np.zeros(len(positions) + 1)
    
    if len(positions) == 1:
        # Handle single position case
        x_edges[0] = positions[0] - 1
        x_edges[1] = positions[0] + 1
    else:
        # Create edges for each marker spanning exactly 1 bp on each side
        # This will leave gaps where markers are far apart
        for i in range(len(positions)):
            x_edges[i] = positions[i] - 1
        x_edges[-1] = positions[-1] + 1
    
    # Individual boundaries (one more than number of individuals)
    y_edges = np.arange(diemMatrix.shape[0] + 1)
    
    # Plot each marker individually to show gaps
    for i in range(len(positions)):
        # Create a small mesh for this single marker
        x_marker = [positions[i] - markerWidth, positions[i] + markerWidth]
        X_marker, Y_marker = np.meshgrid(x_marker, y_edges)
        
        # Extract data for this marker (column i of the matrix)
        marker_data = diemMatrix[:, i:i+1]  # Keep as 2D array
        
        # Plot this marker
        ax.pcolormesh(X_marker, Y_marker, marker_data, cmap=mycmap, norm=norm)
    

    # Add y-axis labels if names are provided
    if names is not None:
        ax.set_yticks(np.arange(len(names)) + 0.5)
        ax.set_yticklabels(names)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    ax.set_xlabel('Position', fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_FONTWEIGHT)
    
    # Set x-axis limits to show the full range
    ax.set_xlim(positions[0] - (positions[-1] - positions[0]) * 0.05, 
                positions[-1] + (positions[-1] - positions[0]) * 0.05)
    
    # Format x-axis to show positions in a readable way
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    _style_axis_text(ax)
    
    if outputPath is not None:
        plt.savefig(outputPath,bbox_inches='tight')

    #plt.tight_layout()
    plt.show()


def _style_axis_text(ax):
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.setp(ax.get_xticklabels(), fontweight=AXIS_FONTWEIGHT)
    plt.setp(ax.get_yticklabels(), fontweight=AXIS_FONTWEIGHT)

    # Keep the scientific notation offset text readable too.
    x_offset_text = ax.xaxis.get_offset_text()
    x_offset_text.set_fontsize(TICK_LABEL_FONTSIZE)
    x_offset_text.set_fontweight(AXIS_FONTWEIGHT)

def characterize_markers(dt):
    '''
    Given a diemtype instance, dt, return a dataframe with information about the marker configurations
    including their counts, DI values, number of individuals differing from ideal marker, and number of individuals with missing data
    '''
    #dt is a diemtype instance
    dmat_cat = np.hstack(dt.DMBC)
    di_cat = np.hstack(dt.DIByChr)

    markerMatrix = dmat_cat.transpose()

    dictDI = {}
    dictCount = {}
    
    for idx,marker in enumerate(markerMatrix):
        config = tuple(marker)
        
        if not config in dictDI:
            dictDI[config] = di_cat[idx]

        if config in dictCount:
            dictCount[config]+=1
        else:
            dictCount[config] = 1

    configList = list( dictDI.keys())
    diList = [dictDI[k] for k in configList]
    countList = [dictCount[k] for k in configList]

    df = pd.DataFrame({'configuration':configList,'count':countList,'DI':diList})
    df.sort_values(by = 'DI',ascending=False,inplace=True,ignore_index=True)

    # adding in additional info...
    idealMarker = df.iloc[0]['configuration']
    diffsFromIdealList = []
    sitesMissingList = []
    for marker in df['configuration']:
        kdiffs = np.sum(np.array(idealMarker)!=np.array(marker))
        diffsFromIdealList.append(kdiffs)
        kmissing = np.sum(np.array(marker)==0)
        sitesMissingList.append(kmissing)
    df['diffs_from_ideal'] = diffsFromIdealList
    df['individuals_missing'] = sitesMissingList
    return df
    

def check_coverage_after_thresholding(dt,threshold):
    '''
    Given a diemtype instance, dt, and a threshold value, return the proportion of sites retained after thresholding
    '''
    #dt is a diemtype instance
    propRetainedByChr = []
    for idx,di in enumerate(dt.DIByChr):
        propRetainedByChr.append( sum(di>=threshold)/len(di))

    return propRetainedByChr

def count_site_differences(dmbc1,dmbc2):
    '''
    Given two lists of diem matrices (that is, by chromosome), count the number of sites that differ between them
    '''
    if len(dmbc1) != len(dmbc2):
        raise ValueError("count_site_differences: diem matrices must have the same number of chromosomes")

    kdiffs = 0
    for chr1, chr2 in zip(dmbc1, dmbc2):
        if chr1.shape != chr2.shape:
            raise ValueError("count_site_differences: diem matrices must have the same shape")
        kdiffs += np.sum(chr1 != chr2)

    return kdiffs

# some functions below used to make dataframes for investigating the effect of thresholding and stuff like that
# these are not currently incorporated into the main workflow but were used, e.g., in making the eseb poster
# and they should be incorporated properly in the data analysis procedure. 

# DS: We used this for Zia's project for exploratory analysis, but I don't think it is needed or wanted for the main workflow
# def remove_missing(stateMatrixByChr, positionsByChr, polarityByChr, DIByChr, mapPosByChr):

#     smbc = []
#     posbc = []
#     polbc = []
#     dibc = []
#     mpbc = []

#     for idx in range(len(stateMatrixByChr)):
#         sm = stateMatrixByChr[idx]
#         pos = positionsByChr[idx]
#         pol = polarityByChr[idx]
#         DI = DIByChr[idx]
#         mapPos = mapPosByChr[idx]
        
#         trans = sm.transpose()
#         thisFilter = trans==0
#         thisFilter = np.array([sum(x)==0 for x in thisFilter])

#         trans = trans[thisFilter]
#         sm = trans.transpose()
#         pos = pos[thisFilter]
#         pol = pol[thisFilter]
#         DI = DI[thisFilter]
#         mapPos = mapPos[thisFilter]

#         smbc.append(sm)
#         posbc.append(pos)
#         polbc.append(pol)
#         dibc.append(DI)
#         mpbc.append(mapPos)

#     return smbc,posbc,polbc,dibc,mpbc
    



# below are some functions for doing basic plotting, but they are to be superceded by more advanced methods in other packages, at lesat for the advanced plotting for publication-level figures
# some functions are for statematrix in diplotype class (where to put this code?)
# some functions are for intervals in the haplolotype class (also, where to put this code?)

def plot_all_spans(d_):
    # d_ is an instance of the Diplotype class
    print("spans for all individuals. blue = hom left allele, magenta = het, red = hom right")
    spans = []
    for thisChr in d_.chromosomes:
        spans = spans + [x.mapSpan for x in thisChr.getOneIntervals()]
    n,myBins,patches = plt.hist(spans,bins=1000,histtype='step',color='b')
    
    for thisChr in d_.chromosomes:
        spans = spans + [x.mapSpan for x in thisChr.getTwoIntervals()]
    plt.hist(spans,bins=1000,histtype='step',color='m')
    
    for thisChr in d_.chromosomes:
        spans = spans + [x.mapSpan for x in thisChr.getThreeIntervals()]
    plt.hist(spans,bins=1000,histtype='step',color='r')
    
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
    
    
def plot_left_spans(d_,markerBreakIdx_):
    print("spans for individuals left of barrier. blue = hom left allele, magenta = het, red = hom right")
    spans = []
    for thisChr in d_.chromosomes:
        if thisChr.ind < markerBreakIdx_:
            spans = spans + [x.mapSpan for x in thisChr.getOneIntervals()]
    n,myBins,patches = plt.hist(spans,bins=1000,histtype='step',color='b')
    
    spans = []
    for thisChr in d_.chromosomes:
        if thisChr.ind < markerBreakIdx_:
            spans = spans + [x.mapSpan for x in thisChr.getTwoIntervals()]
    plt.hist(spans,bins=1000,histtype='step',color='m')
    
    spans = []
    for thisChr in d_.chromosomes:
        if thisChr.ind < markerBreakIdx_:
            spans = spans + [x.mapSpan for x in thisChr.getThreeIntervals()]
    plt.hist(spans,bins=1000,histtype='step',color='r')
    
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
    
    
def plot_right_spans(d_,markerBreakIdx_):
    print("spans for individuals right of barrier. blue = hom left allele, magenta = het, red = hom right")
    spans = []
    for thisChr in d_.chromosomes:
        if thisChr.ind >= markerBreakIdx_:
            spans = spans + [x.mapSpan for x in thisChr.getOneIntervals()]
    n,myBins,patches = plt.hist(spans,bins=1000,histtype='step',color='b')
    
    spans = []
    for thisChr in d_.chromosomes:
        if thisChr.ind >= markerBreakIdx_:
            spans = spans + [x.mapSpan for x in thisChr.getTwoIntervals()]
    plt.hist(spans,bins=1000,histtype='step',color='m')
    
    spans = []
    for thisChr in d_.chromosomes:
        if thisChr.ind >= markerBreakIdx_:
            spans = spans + [x.mapSpan for x in thisChr.getThreeIntervals()]
    plt.hist(spans,bins=1000,histtype='step',color='r')
    
    plt.yscale("log")
    plt.xscale("log")
    plt.show()



# these functions are for exploring the data a bit, used in creating my eseb poster. 
# we should consider if they are useful/ informative and whether they should be included in the final release

def get_intro_amt_and_bias(lv,rv,brakeIdx,conf):
    conf = np.array(conf)
    lconf,rconf = [conf[0:brakeIdx], conf[brakeIdx:]]
    nl = 2*sum(lconf != 0)
    nr = 2*sum(rconf != 0 )
    if nl == 0 or nr==0:
        return(np.nan,np.nan)
    introAmount = (1*sum(lconf == 2) + 2*sum(lconf == rv) + 1*sum(rconf==2) + 2*sum(rconf==lv))/(nl+nr)
    lb = (1*sum(lconf==2) + 2*sum(lconf == rv))/nl
    rb = (1*sum(rconf == 2) + 2*sum(rconf == lv))/nr
    totb = -(lb - rb)
    return introAmount,totb

# def get_thresholds(SMBC_,DIBC_):
#     sm = np.hstack(SMBC_)
#     smt = sm.transpose()
#     dis = np.concat(DIBC_)
#     bestIdx = np.argmax(dis)
#     idealMarker = smt[bestIdx]
#     kmax = len(idealMarker)
#     DIbyDiffs = np.zeros(kmax+1)
    
#     for idx,markerConfig in enumerate(smt):
        
#         kDiffs = sum(markerConfig != idealMarker)
#         thisDI = dis[idx]
#         if thisDI < DIbyDiffs[kDiffs]:
#             DIbyDiffs[kDiffs] = thisDI
#     return idealMarker, [[k,d] for k, d in enumerate(DIbyDiffs)]

#this function possibly made for Zia's project specifically, I don't recal. 
def calc_contig_his(SMBC,sexPloidy,autoPloidy,isSexChr):
    #sexPloidy is a list of the ploidy of the individuals as ordered in SMBC
    #autoPloidy is a list of the ploidy (here, always twos) as for sexPloidy, ordered by SMBC!
    #isSexChr is a list with '1' indicating sex chr and '0' indicated autosome
    HIBC = []
    for idx,sm in enumerate(SMBC):

        am = pol.hapStateIndices_to_hapMatrix(sm.transpose())
        if isSexChr[idx]==1:
            ploidyArr = sexPloidy[:,np.newaxis]
        else:
            ploidyArr = autoPloidy[:,np.newaxis]
        am = am*ploidyArr
        A4 = np.sum(am,axis=0)
        his = pol.get_hybrid_index(A4)
        HIBC.append(his)

    return np.array(HIBC)