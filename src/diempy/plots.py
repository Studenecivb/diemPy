from __future__ import annotations


# ---- stdlib ----
from collections import defaultdict
from itertools import groupby, chain
import multiprocessing as mp


# ---- numpy / pandas ----
import numpy as np
import pandas as pd


# ---- matplotlib ----
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Polygon, Rectangle
import matplotlib.colors as mcolors
from matplotlib.widgets import Button, Slider
from matplotlib.colors import to_rgb, LinearSegmentedColormap

import time
from dataclasses import dataclass
from typing import Iterable, Callable, Dict, Tuple, List, Optional, Sequence

# ---- diem internals ----
from . import smooth


# explicitly used smoothing entry point
from .smooth import laplace_smooth_multiple_haplotypes

# more explicit imports
from fractions import Fraction
#from bisect import bisect_left
import bisect
from joblib import Parallel, delayed # for parallel computation of 'pwdmatrix' pairwise distance matrix rows

"""********************************************************
Keyword 'DMBCtoucher' is used to label the 
points where plots.py touches diemtypes.DMBC states

This is for future-proofing against the day 
where DMBC is allowed to store states other than {0,1,2,3}->{U,0,1,2} 
(c.f. vcf2diem output)

plots.py _currently_ has potential access to extra vcf2diem states through
    read_diem_bed_4_plots
    -> diemIrisFromPlotPrep
    -> diemLongFromPlotPrep

TODO:
    Chr_Nickname
    Ind_Nickname   can be linked with meta data to allow eg coloured name plotting (eg showing a-priori clusters)

    Stop using files bypass to get at extra states.
        Collab with Derek to store extra state internally in DMBC

    Rename the summaries functions more transparently - and move them out into an analytics section
********************************************************"""


""" _____________________ START Mathematica2Python _____________________"""
##############################
#### Mathematica2Python
### Author: Stuart J.E. Baird
###############################


def Split(seq, same_test=lambda a, b: a == b):
    '''
    Function to split a sequence into sublists based on a test function.
    ChatGPT 5.2 provided Mathematica Split equivalent in Python.
    
    Args:
        seq (list): The input sequence to be split.
        same_test (function): A function that takes two arguments and returns True if they are considered the same.
    Returns:
        list: A list of sublists, where each sublist contains consecutive elements that are considered the same.
    '''
    if not seq:
        return []
    out = [[seq[0]]]
    for x in seq[1:]:
        (out[-1] if same_test(out[-1][-1], x) else out.append([x])) and out[-1].append(x)
    return out


def RichRLE(lst):
    """
    Function generating a Rich Run Length Encoding of a list.
    
    Args:
        lst (list): The input list to be encoded.
    Returns:
        lists: A list containing four lists: states, lengths, starts, and ends of each run.
    """
    slst = Split(lst, lambda x, y: y == x)
    cumstart = 0
    states = []
    lengths = []
    starts = []
    ends = []
    for i in slst:
        leni = len(i)
        states.append(i[0])
        lengths.append(leni)
        starts.append(cumstart)
        cumstart += leni
        ends.append(cumstart - 1)
    return [states, lengths, starts, ends]


def Map(f, lst): 
    """
    equivalent to Mathematica Map 
    """
    return list(map(f, lst))



def ParallelMap(f, lst):
    """ 
    equivalent to Mathematica ParallelMap 
    """
    pool = mp.Pool()
    return list(pool.map(f, lst))


def Flatten(lstOlists): 
    """ 
    equivalent to Mathematica Flatten
    """
    return list(chain.from_iterable(lstOlists)) #itertools



def StringJoin(slst):
    """ 
    equivalent to Mathematica StringJoin 
    """
    separator = ''
    return separator.join(slst)


def Transpose(mat): 
    """ 
    equivalent to Mathematica Transpose
    """
    return list(np.array(mat).T)  # care here - hidden type casting on heterogeneous 'mat'rices


def StringTranspose(slst): 
    """ 
    equivalent to Mathematica StringTranspose 
    """
    return Map(StringJoin, Transpose(Map(Characters, slst)))


def Tally(lst):  # single pass so in principle fast ( O(n) ) but answers unsorted
    """ 
    equivalent to Mathematica Tally 
    """
    states = []
    tally = []
    for x in lst:
        p = FirstPosition(states, x)
        if p == []:
            states.append(x)
            tally.append([x, 1])
        else:
            tally[p[0]][1] += 1
    return tally

def Second(lst): 
    """ equivalent to Mathematica Second """
    return lst[1]


def Total(lst): 
    """"""
    return sum(lst)


def Join(lst1, lst2): 
    """ equivalent to Mathematica Join """
    return lst1 + lst2


def Take(lst, n):
    """ 
    equivalent to Mathematica Take 
    """
    if n > 0:
        ans = lst[:n]
    elif n == 0:
        ans = lst
    else:
        ans = lst[n:]
    return ans


def Drop(lst, n):
    """ equivalent to Mathematica Drop """
    if n > 0:
        ans = lst[n:]
    elif n == 0:
        ans = lst
    else:
        ans = lst[:n]
    return ans

def FirstPosition(lst, elem):
    """ equivalent to Mathematica FirstPosition """
    i = -1
    pos = []
    for l in lst:
        i += 1
        if l == elem:
            pos.append(i)
            break
    return pos

def Characters(s): 
    """ equivalent to Mathematica Characters """
    return [*s]

def StringTakeList(string, lengths):
    """ equivalent to Mathematica StringTakeList """
    substrings = []
    current_index = 0
    for length in lengths:
        substrings.append(string[current_index:current_index + length])
        current_index += length
    return substrings

""" _____________________ END Mathematica2Python _____________________"""


""" _____________________ START DIEMPy 2023 snippets _____________________"""

##############################
#### From DIEMPy
### Author: Stuart J.E. Baird
###############################


StringReplace20_dict = str.maketrans('02', '20')
""" simultaneous 2<->0 replacement dictionary """

def StringReplace20(text):
    """will _!simultaneously!_ replace 2->0 and 0->2"""
    return text.translate(StringReplace20_dict)


def sStateCount(s):
    """ 
    counts diem States as chars
    Args:   astring
    Returns: list of counts [nU,n0,n1,n2]
    """
    counts = Map(Second, Tally(Join(["0", "1", "2"], Characters(s))))
    nU = Total(
        Drop(counts, 3)
    )  # only the three 'call' chars above are not U encodings!
    counts = list(np.array(Take(counts, 3)) - 1)
    return Join([nU], counts)


def pHetErrOnString(s):
    """
    Calculates state frequency,heterozygosity and error rate from a string of diem states.
    Args:   astring
    Returns: tuple of (pHetErr, pHet, pErr)
    """
    sCount = sStateCount(s)
    callTotal = Total(Drop(sCount, 1))
    if callTotal > 0:
        ans = (
            Total(np.array(sCount) * [0, 0, 1, 2]) / (2 * callTotal),
            sCount[2] / callTotal,
            sCount[0] / Total(sCount),
        )
    else:  # no calls... are there any Us?
        if sCount[0] > 0:
            pErr = 1
        else:
            pErr = "NA"
        ans = ("NA", "NA", pErr)
    return ans


""" _____________________ START DIEMPy 2023 snippets _____________________"""



""" ______________new support by SJEB STARTING_______________________________"""

def Chr_Nickname(chr_name):
    """
    Shorten chromosome names for plotting.
    E.g., 'chromosome_1' -> 'Chr 1'
    Args:
        chr_name (str): Full chromosome name.
    Returns:
        str: Shortened chromosome name.
    """
    if 'scaffold_' in chr_name:
        return chr_name.replace('scaffold_', 'Scaf ')[-10:]
    elif 'scaffold' in chr_name:
        return chr_name.replace('scaffold', 'Scaf ')[-10:]
    elif 'chromosome_' in chr_name:
        return chr_name.replace('chromosome_', 'Chr ')[-9:]
    elif 'chromosome' in chr_name:
        return chr_name.replace('chromosome', 'Chr ')[-9:]
    else:
        return chr_name[-15:]
    
def Ind_Nickname(ind_name):
    """
    Shorten Ind names for plotting.
    E.g., 'chromosome_1' -> 'Chr 1'
    Args:
        ind_name (str): Full individual name.
    Returns:
        str: Shortened individual name.
    """
    shortlength = 11

    if '_NA' in ind_name:
        return ind_name.replace('_NA', '')[-shortlength:]
    else:
        return ind_name[-shortlength:]
    

def read_diem_bed_4_plots(bed_file_path, meta_file_path):
    """
    Reads a diem BED file and meta file for use in plots.py
    code copy from Derek Setter's read_diem_bed with additions by SJEB
    
    Derek comments:
    Fast version of read_diem_bed with significant performance improvements.
    
    Args:
    bed_file_path (str): Path to the diem BED file.
    meta_file_path (str): Path to the diem metadata file.

    Returns:
    A tuple of a DiemType object containing the diem BED data 
    (POLARISED (if hasPolarity)) and an IndsName file.
    """
    
    # Read metadata - no changes needed here as it's already fast
    df_meta = pd.read_csv(meta_file_path, sep='\t')
    chrNames = np.array(df_meta['#Chrom'].values)
    #chrRelativeRecRates = np.array(df_meta['relativeRecRates'].values)
    if "relativeRecRates" in df_meta.columns:
        chrRelativeRecRates = np.asarray(df_meta["relativeRecRates"].values)
    else:
        # Default: no recombination adjustment
        # Use chromosome count (24 in your case)
        chrRelativeRecRates = np.ones(len(chrNames), dtype=float)
    chrLengths = np.array(df_meta['RefEnd0'].values) - np.array(df_meta['RefStart0'].values)
    sampleNames = np.array(df_meta.columns[7:]) # Skip everything in title line up to relativeRecRates
    
    ploidyByChr = []
    for chr in chrNames:
        row = df_meta[df_meta['#Chrom'] == chr]
        ploidy = np.array(row.iloc[0,6:].values, dtype=int)
        ploidyByChr.append(ploidy)
    
    # Fast preamble reading - same as before
    preamble = []
    nSkipLines = 0
    individualsMasked = None
    with open(bed_file_path, 'r') as f:
        for line in f:
            if line.startswith('##'):
                preamble.append(line.strip())
                if line.startswith('##IndividualsMasked='):
                    clean_line = line.strip().removeprefix('##IndividualsMasked=')
                    if clean_line == 'None':
                        individualsMasked = None
                    else:
                        individualsMasked = clean_line.split(',')
                nSkipLines += 1
            else:
                break
    
    # Determine column names
    if len(preamble) > 0:
        hasPolarity = True
        column_names = [
            'chrom', 'start', 'end', 'qual', 'ref', 
            'SeqAlleles', 'SNV', 'nVNTs', 
            'exclusion_criterion', 'diem_genotype','nullPolarity','polarity',
            'DI','Support','masked'
        ]
    else:
        hasPolarity = False
        column_names = [
            'chrom', 'start', 'end', 'qual', 'ref', 
            'SeqAlleles', 'SNV', 'nVNTs', 
            'exclusion_criterion', 'diem_genotype'
        ]
    
    # Read the entire BED file at once
    df_bed = pd.read_csv(bed_file_path, sep='\t', names=column_names, skiprows=nSkipLines+1)

    # Polarise - these last lines SJEB
                
    if hasPolarity:
        #print('updating genotype polarities')
        mask = df_bed['polarity'] == 1
        df_bed.loc[mask, 'diem_genotype'] = df_bed.loc[mask, 'diem_genotype'].apply(StringReplace20)
        
    return df_bed,sampleNames,chrLengths,chrRelativeRecRates


def get_DI_span(aDT):
    """
    Get the min and max DI values across all chromosomes.

    Args:   aDT : DiemType
    """
    minDI=float('inf')
    maxDI=float('-inf')
    for idx, chr  in enumerate(aDT.DIByChr):
        minDI=min(minDI,min(aDT.DIByChr[idx]))
        maxDI=max(maxDI,max(aDT.DIByChr[idx]))
    return [minDI,maxDI]




"""______________________START statewise_genomes_summary_given_DI______________________________"""



def _statewise_summary_one_chr(SM, DI, ploidies, DIthreshold):
    """
    Compute statewise summary for ONE chromosome.
    Returns (counts_dict, (RetainedNumer, RetainedDenom))

    DMBCtoucher

    This is a helper for PAR_statewise_genomes_summary_given_DI.
    """

    nInds = SM.shape[0]

    DIfilter = DI >= DIthreshold
    RetainedNumer = int(np.count_nonzero(DIfilter))
    RetainedDenom = int(DIfilter.size)

    if RetainedNumer == 0:
        return (
            {
                "counts0": np.zeros(nInds, dtype=float),
                "counts1": np.zeros(nInds, dtype=float),
                "counts2": np.zeros(nInds, dtype=float),
                "counts3": np.zeros(nInds, dtype=float),
            },
            (RetainedNumer, RetainedDenom),
        )

    SMf = SM[:, DIfilter]

    # Vectorised masks
    is0 = (SMf == 0)
    is1 = (SMf == 1)
    is2 = (SMf == 2)
    is3 = ~(is0 | is1 | is2)

    counts0 = is0.sum(axis=1)
    counts1 = is1.sum(axis=1)
    counts2 = is2.sum(axis=1)
    counts3 = is3.sum(axis=1)

    w = ploidies.astype(float)

    return (
        {
            "counts0": w * counts0,
            "counts1": w * counts1,
            "counts2": w * counts2,
            "counts3": w * counts3,
        },
        (RetainedNumer, RetainedDenom),
    )


def PAR_statewise_genomes_summary_given_DI(
    aDT,
    DIthreshold: float,
    n_jobs=-1,
    backend="loky",
):
    """
    Parallel version of statewise_genomes_summary_given_DI.

    DMBCtoucher

    Parallelised over chromosomes.
    """

    nChr = len(aDT.DMBC)

    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        prefer="processes",
        batch_size=1,
    )(
        delayed(_statewise_summary_one_chr)(
            aDT.DMBC[i],
            aDT.DIByChr[i],
            aDT.chrPloidies[i],
            DIthreshold,
        )
        for i in range(nChr)
    )

    chrom_counts = [r[0] for r in results]
    chrom_retained = [r[1] for r in results]

    return chrom_counts, chrom_retained



def SER_statewise_genomes_summary_given_DI(aDT, DIthreshold: float):
    """
    Statewise summary of genomes under a DI threshold.

    DMBCtoucher

    Refinements over genomes_summary_given_DI:
      1) counts3 = count of states NOT in {0,1,2}
      2) returns per-chromosome per-individual state counts
      3) returns per-chromosome retained counts

    Args
    ----
    aDT : DiemType
        Must provide:
          - DMBC : list of arrays, each shape (nInds, nSites_chr)
          - DIByChr : list of 1D arrays, per chromosome
          - chrPloidies : list of per-individual ploidies
    DIthreshold : float
        DI filter threshold

    Returns
    -------
    chrom_counts : list of dicts, length nChr
        Each dict has keys:
          'counts0', 'counts1', 'counts2', 'counts3'
        Each value is a float array of shape (nInds,)
        Counts are ploidy-weighted.
    chrom_retained : list of tuples, length nChr
        Each element is (RetainedNumer_chr, RetainedDenom_chr)
    """

    nChr = len(aDT.DMBC)
    nInds = aDT.DMBC[0].shape[0]

    chrom_counts = []
    chrom_retained = []

    for chr_idx in range(nChr):
        SM = aDT.DMBC[chr_idx]               # (nInds, nSites)
        DI = aDT.DIByChr[chr_idx]            # (nSites,)
        ploidies = aDT.chrPloidies[chr_idx]  # (nInds,)

        DIfilter = DI >= DIthreshold

        RetainedNumer = int(np.count_nonzero(DIfilter))
        RetainedDenom = int(DIfilter.size)
        chrom_retained.append((RetainedNumer, RetainedDenom))

        if RetainedNumer == 0:
            chrom_counts.append({
                "counts0": np.zeros(nInds, dtype=float),
                "counts1": np.zeros(nInds, dtype=float),
                "counts2": np.zeros(nInds, dtype=float),
                "counts3": np.zeros(nInds, dtype=float),
            })
            continue

        SMf = SM[:, DIfilter]  # (nInds, nRetained)

        # Vectorised state masks
        is0 = (SMf == 0)
        is1 = (SMf == 1)
        is2 = (SMf == 2)

        # counts3 = NOT in {0,1,2}
        is3 = ~(is0 | is1 | is2)

        # Per-individual counts
        counts0 = is0.sum(axis=1)
        counts1 = is1.sum(axis=1)
        counts2 = is2.sum(axis=1)
        counts3 = is3.sum(axis=1)

        # Apply ploidy weights once
        w = ploidies.astype(float)

        chrom_counts.append({
            "counts0": w * counts0,
            "counts1": w * counts1,
            "counts2": w * counts2,
            "counts3": w * counts3,
        })

    return chrom_counts, chrom_retained


def statewise_genomes_summary_given_DI(aDT, DIthreshold: float):
    if len(aDT.DMBC) >= 6:
        return PAR_statewise_genomes_summary_given_DI(aDT, DIthreshold)
    else:
        return SER_statewise_genomes_summary_given_DI(aDT, DIthreshold)
"""______________________END statewise_genomes_summary_given_DI______________________________"""


def summaries_from_statewise_counts(chrom_counts):
    """
    Compute [HI, HOM1, HET, HOM2, U] from statewise chrom_counts.

    Args: chrom_counts: iterable of dicts with keys
        'counts0', 'counts1', 'counts2', 'counts3'
        (arrays of shape nInds)

    Returns:
        summaries = [HI, HOM1, HET, HOM2, U]
    """

    A0 = sum(c["counts0"] for c in chrom_counts)
    A1 = sum(c["counts1"] for c in chrom_counts)
    A2 = sum(c["counts2"] for c in chrom_counts)
    A3 = sum(c["counts3"] for c in chrom_counts)

    dipDenom   = A1 + A2 + A3
    hapDenom   = 2 * dipDenom
    stateDenom = dipDenom + A0

    HI = np.divide((A2 + 2*A3), hapDenom, out=np.zeros_like(A2, dtype=float), where=(hapDenom != 0))
    HOM1 = np.divide(A1, dipDenom, out=np.zeros_like(A1, dtype=float), where=(dipDenom != 0))
    HET  = np.divide(A2, dipDenom, out=np.zeros_like(A2, dtype=float), where=(dipDenom != 0))
    HOM2 = np.divide(A3, dipDenom, out=np.zeros_like(A3, dtype=float), where=(dipDenom != 0))
    U    = np.divide(A0, stateDenom, out=np.zeros_like(A0, dtype=float), where=(stateDenom != 0))

    return [HI, HOM1, HET, HOM2, U]



def genomes_summary_given_DI(aDT, DIthreshold: float):
    """
    Summarises a diemType.DMBC across chromosomes, applying a DI threshold filter.
    Uses statewise_genomes_summary_given_DI for efficiency and single-point-of-entry to DMBC.

    Args:
    aDT: a DiemType
    DIthreshold: DI threshold filter
    Returns:
    summaries: list of per-individual summary arrays [HI, HOM1, HET, HOM2, U]
    RetainedNumer: number of retained sites after DI filtering
    RetainedDenom: total number of sites before DI filtering
    """
    chrom_counts, chrom_retained = \
        statewise_genomes_summary_given_DI(aDT, DIthreshold)

    summaries = summaries_from_statewise_counts(chrom_counts)

    DInumer = sum(n for n, _ in chrom_retained)
    DIdenom = sum(d for _, d in chrom_retained)

    return summaries, DInumer, DIdenom



def fractional_positions_of_multiples(A, delta):
    """
    Calculate fractional positions of multiples of delta in a sorted array A.

    Used as an inverse linear interpolation from reference (physical) positions to site indices;
    for example, to place ticks at regular physical intervals on a plot of site indices.

    Args:
        A (array-like): Sorted array of values (e.g. physical positions of DI filtered SNVs).
        delta (float): The interval for multiples (e.g. 1000 for kb ticks).
    Returns:
        np.ndarray: Array of (value (tick label), position (tick placement on SNV metric)) pairs.
    """

    A = np.asarray(A)
    n = len(A)

    values = []
    positions = []

    max_k = A[-1] // delta

    for k in range(1, max_k + 1):
        x = k * delta
        i = bisect.bisect_left(A, x)

        # skip values below the first element
        if i == 0:
            continue

        # exact match
        if i < n and A[i] == x:
            pos = float(i)
        else:
            left, right = A[i - 1], A[i]
            pos = (i - 1) + (x - left) / (right - left)

        values.append(x)
        positions.append(pos)
    tick_values = np.array(values)/delta
    tick_positions = np.array(positions)+1
    return np.column_stack((tick_values, tick_positions))
""" new support by SJEB ENDING"""


diemColours = [
    'white',
    mcolors.to_hex((128/255, 0, 128/255)),  # RGBColor[128/255, 0, 128/255] - Purple
    mcolors.to_hex((255/255, 229/255, 0)),  # RGBColor[255/255, 229/255, 0] - Yellow
    mcolors.to_hex((0, 128/255, 128/255))   # RGBColor[0, 128/255, 128/255] - Teal
]


"""________________________________________ cache-ing helpers ___________________"""


# Types matching existing API
ChromCounts = List[dict]                 # length nChr, dict with counts0..counts3 arrays (nInds,)
ChromRetained = List[Tuple[int, int]]    # length nChr, (kept, total)


@dataclass
class _ChrIncrementalState:
    """Internal mutable state for one chromosome during cache prefill."""
    order_asc: np.ndarray          # (nSites,) indices sorting DI ascending
    di_sorted: np.ndarray          # (nSites,) DI in ascending order
    k: int                         # current start index of retained suffix in di_sorted
    c0: np.ndarray                 # (nInds,) int counts (unweighted)
    c1: np.ndarray                 # (nInds,)
    c2: np.ndarray                 # (nInds,)
    c3: np.ndarray                 # (nInds,)
    ploidy_w: np.ndarray           # (nInds,) float weights


class StatewiseDIIncrementalCache:
    """
    Prefills snapshots for statewise_genomes_summary_given_DI for a fixed DI grid.

    Key property:
      - Across the whole prefill, each site is incorporated at most once (incremental).
      - No giant per-site prefix arrays are stored.
      - Snapshots are stored as ploidy-weighted float arrays (same as your API).
    """

    def __init__(
        self,
        aDT,
        di_grid: np.ndarray,
        progress: Optional[str] = None,   # None | "text"
        label: str = "Statewise cache prefill",
    ):
        self.aDT = aDT
        self.di_grid = np.asarray(di_grid, dtype=float)
        self.progress = progress
        self.label = label

        # Snapshots keyed by exact grid value (float)
        self._snapshots: Dict[float, Tuple[ChromCounts, ChromRetained]] = {}

        # Precompute per-chromosome incremental state
        self._chr_states: List[_ChrIncrementalState] = self._init_chr_states()

    def _init_chr_states(self) -> List[_ChrIncrementalState]:
        nChr = len(self.aDT.DMBC)
        nInds = self.aDT.DMBC[0].shape[0]
        states: List[_ChrIncrementalState] = []

        for chr_idx in range(nChr):
            DI = np.asarray(self.aDT.DIByChr[chr_idx], dtype=float)
            order = np.argsort(DI, kind="mergesort")  # stable, asc
            di_sorted = DI[order]

            ploidy_w = np.asarray(self.aDT.chrPloidies[chr_idx], dtype=float)

            states.append(
                _ChrIncrementalState(
                    order_asc=order,
                    di_sorted=di_sorted,
                    k=di_sorted.size,  # start with none retained (threshold > max)
                    c0=np.zeros(nInds, dtype=np.int64),
                    c1=np.zeros(nInds, dtype=np.int64),
                    c2=np.zeros(nInds, dtype=np.int64),
                    c3=np.zeros(nInds, dtype=np.int64),
                    ploidy_w=ploidy_w,
                )
            )
        return states

    def prefill(self):
        """
        Fill snapshots for all DI values in self.di_grid.

        We iterate thresholds from HIGH → LOW so the retained set only grows,
        letting us add newly-included sites once.
        """
        t0 = time.time()

        # We prefill in descending DI so retained grows monotonically.
        grid_desc = np.array(sorted(self.di_grid, reverse=True), dtype=float)
        n_steps = grid_desc.size

        for step_i, thr in enumerate(grid_desc, start=1):
            chrom_counts: ChromCounts = []
            chrom_retained: ChromRetained = []

            for chr_idx, st in enumerate(self._chr_states):
                nSites = st.di_sorted.size
                nInds = st.c0.size

                # New retained suffix start index for DI >= thr
                k_new = int(np.searchsorted(st.di_sorted, thr, side="left"))

                # Add sites that become newly retained: [k_new : st.k)
                if k_new < st.k:
                    # indices in original site order
                    idxs = st.order_asc[k_new:st.k]
                    SM = self.aDT.DMBC[chr_idx]  # (nInds, nSites_chr)

                    # Take the block (this allocates a temporary block array)
                    block = np.take(SM, idxs, axis=1)

                    # Count states in the block
                    b0 = (block == 0).sum(axis=1)
                    b1 = (block == 1).sum(axis=1)
                    b2 = (block == 2).sum(axis=1)
                    # everything else -> state3
                    b3 = block.shape[1] - (b0 + b1 + b2)

                    st.c0 += b0
                    st.c1 += b1
                    st.c2 += b2
                    st.c3 += b3

                    st.k = k_new  # retained suffix starts earlier now

                kept = nSites - st.k
                chrom_retained.append((int(kept), int(nSites)))

                w = st.ploidy_w
                chrom_counts.append({
                    "counts0": w * st.c0.astype(float),
                    "counts1": w * st.c1.astype(float),
                    "counts2": w * st.c2.astype(float),
                    "counts3": w * st.c3.astype(float),
                })

            # Store snapshot under exact threshold value
            self._snapshots[float(thr)] = (chrom_counts, chrom_retained)

            if self.progress == "text":
                elapsed = time.time() - t0
                pct = int(round(100.0 * step_i / n_steps))
                print(f"{self.label}: {step_i}/{n_steps} ({pct}%)  elapsed {elapsed:.1f}s")

        return self

    def get(self, DIthreshold: float) -> Tuple[ChromCounts, ChromRetained]:
        """
        Retrieve nearest snapshot for arbitrary DIthreshold.
        """
        thr = float(DIthreshold)

        # Find nearest grid value
        grid = self.di_grid
        j = int(np.argmin(np.abs(grid - thr)))
        key = float(grid[j])

        # NOTE: snapshots stored by exact float of grid values; should exist if prefilling done
        return self._snapshots[key]


@dataclass
class _ChrPWState:
    order_asc: np.ndarray   # indices sorting DI ascending
    di_sorted: np.ndarray   # DI ascending
    k: int                  # current suffix start index (retained = [k:])
    SM: np.ndarray          # (n_ind, n_sites_chr) numeric codes, clamped to 0..3


def _pw_weight_matrix():
    W = np.zeros((4, 4), dtype=np.float32)
    W[1, 2] = W[2, 1] = 1
    W[1, 3] = W[3, 1] = 2
    W[2, 2] = 1
    W[2, 3] = W[3, 2] = 1
    return W


class PairwiseDIIncrementalCache:
    """
    Incremental prefill cache for PARApwmatrixFromDiemType-like distance matrices
    over a fixed DI grid.

    - Prefill iterates DI thresholds HIGH → LOW so retained set grows.
    - Each site is incorporated at most once across the entire prefill.
    - Snapshots store M (num/den) for each DI grid point.
    """

    def __init__(
        self,
        aDT,
        di_grid: np.ndarray,
        *,
        chrom_indices: Optional[Sequence[int]] = None,
        progress: Optional[str] = None,      # None | "text"
        label: str = "Pairwise cache prefill",
        snapshot_dtype=np.float32,           # store snapshots as float32 to reduce memory
    ):
        self.aDT = aDT
        self.di_grid = np.asarray(di_grid, dtype=float)
        self.progress = progress
        self.label = label
        self.snapshot_dtype = snapshot_dtype

        self.n_ind = int(aDT.DMBC[0].shape[0])
        self.W = _pw_weight_matrix()

        if chrom_indices is None:
            chrom_indices = range(len(aDT.DMBC))
        self.chrom_indices = [int(i) for i in chrom_indices]

        self._chr_states = self._init_chr_states()
        self._snapshots: Dict[float, np.ndarray] = {}

        # Global incremental accumulators across ALL chosen chromosomes
        self._num = np.zeros((self.n_ind, self.n_ind), dtype=np.float64)
        self._den = np.zeros((self.n_ind, self.n_ind), dtype=np.int64)
        self._diag_num = np.zeros(self.n_ind, dtype=np.float64)
        self._diag_den = np.zeros(self.n_ind, dtype=np.int64)

    def _init_chr_states(self):
        states = []
        for chr_idx in self.chrom_indices:
            DI = np.asarray(self.aDT.DIByChr[chr_idx], dtype=float)
            order = np.argsort(DI, kind="mergesort")   # asc
            di_sorted = DI[order]

            # Clamp states: >2 -> 3 (future-proofing) ; keep 0 as missing
            #SM = np.minimum(self.aDT.DMBC[chr_idx], 3).astype(np.int8)
            # the clamp is wrong. Fix it: we want 0 to stay 0, and everything else >2 to become 3
            SM0 = self.aDT.DMBC[chr_idx]
            SM = np.where((SM0 >= 0) & (SM0 <= 3), SM0, 0).astype(np.int8)

            states.append(_ChrPWState(
                order_asc=order,
                di_sorted=di_sorted,
                k=di_sorted.size,  # start with none retained (thr > max)
                SM=SM,
            ))
        return states

    def _add_site_block(self, SM, idxs):
        W = self.W

        for s in idxs:
            col = SM[:, s]          # (n_ind,)
            valid = (col != 0)
            idx = np.nonzero(valid)[0]
            m = idx.size
            if m == 0:
                continue

            vals = col[idx].astype(np.int64)

            # ---- diagonal accumulators (within-individual) ----
            # valid site contributes to denominator
            self._diag_den[idx] += 1
            # HET contributes 1 unit (later /2 -> 0.5)
            self._diag_num[idx] += (vals == 2)

            # ---- off-diagonal accumulators ----
            if m < 2:
                continue

            ww = W[vals[:, None], vals[None, :]].astype(np.float64)

            ones = np.ones((m, m), dtype=np.int64)
            np.fill_diagonal(ones, 0)

            ix = np.ix_(idx, idx)
            self._num[ix] += ww
            self._den[ix] += ones


    def prefill(self):
        t0 = time.time()
        grid_desc = np.array(sorted(self.di_grid, reverse=True), dtype=float)
        n_steps = grid_desc.size

        for step_i, thr in enumerate(grid_desc, start=1):
            # Update each chromosome incremental state, and add newly-retained sites
            for st in self._chr_states:
                # retained set is DI >= thr => suffix start k_new
                k_new = int(np.searchsorted(st.di_sorted, thr, side="left"))

                if k_new < st.k:
                    # newly included sites are [k_new : st.k) in DI-sorted order
                    idxs = st.order_asc[k_new:st.k]
                    self._add_site_block(st.SM, idxs)
                    st.k = k_new

            # Snapshot matrix for this threshold
            M = np.full((self.n_ind, self.n_ind), np.nan, dtype=np.float64)

            mask = self._den > 0
            M[mask] = self._num[mask] / (2.0 * self._den[mask])

            # fill diagonal with within-individual heterozygosity-like distance
            dmask = self._diag_den > 0
            diag = np.full(self.n_ind, np.nan, dtype=np.float64)
            diag[dmask] = self._diag_num[dmask] / (2.0 * self._diag_den[dmask])
            np.fill_diagonal(M, diag)

            self._snapshots[float(thr)] = M.astype(self.snapshot_dtype, copy=False)


            if self.progress == "text":
                elapsed = time.time() - t0
                pct = int(round(100.0 * step_i / n_steps))
                print(f"{self.label}: {step_i}/{n_steps} ({pct}%)  elapsed {elapsed:.1f}s")

        return self

    def get(self, DIthreshold: float) -> np.ndarray:
        """
        Return nearest snapshot matrix to DIthreshold.
        """
        thr = float(DIthreshold)
        grid = self.di_grid
        j = int(np.argmin(np.abs(grid - thr)))
        key = float(grid[j])
        return self._snapshots[key]

"""________________________________________ START GenomeSummariesPlot ___________________"""


class GenomeSummaryPlot:
    """
    Plots genome summaries with DI filtering and interactive widgets.

    These summaries include HI, HOM1, HET, HOM2, and U proportions per individual.
    Cursor hover displays individual IDs.
    Reorder button sorts individuals by HI given current DI filter.
    
    Args:
        dPol: DiemType object containing genomic data.
    
    Drop-in extension:
      - optional cache prefill over a DI grid (text progress)
      - DI slider uses cached results (nearest match within tol)
      - PREFILL uses StatewiseDIIncrementalCache (incremental, sorted-DI sweep)
    """
    def __init__(
        self,
        dPol,
        *,
        prefill_cache=False,      # NEW
        prefill_step=None,        # NEW
        cache_tol=None,           # NEW
        progress=None,          # NEW: "text" | "none"
    ):
        self.dPol = dPol

        # ---- initial state ----
        self.IndNickNames = [Ind_Nickname(name) for name in dPol.indNames]
        self.indNameFont = 6
        self.indHIorder = np.arange(len(dPol.indNames))

        # ---- cache (per instance) ----
        # cache maps DI_value(float) -> (summaries, chrom_retained)
        self._cache = {}
        self._cache_keys_sorted = []

        def _cache_set(di, value):
            di = float(di)
            if di not in self._cache:
                bisect.insort(self._cache_keys_sorted, di)
            self._cache[di] = value

        def _cache_get_nearest(di, tol):
            """Return cached payload for nearest DI within tol, else None."""
            if not self._cache_keys_sorted:
                return None
            di = float(di)

            keys = self._cache_keys_sorted
            j = bisect.bisect_left(keys, di)

            candidates = []
            if 0 <= j < len(keys):
                candidates.append(keys[j])
            if 0 <= j - 1 < len(keys):
                candidates.append(keys[j - 1])

            best = None
            best_dist = None
            for k in candidates:
                dist = abs(k - di)
                if best_dist is None or dist < best_dist:
                    best = k
                    best_dist = dist

            if best is None or best_dist is None or best_dist > tol:
                return None
            return self._cache[best]

        self._cache_set = _cache_set
        self._cache_get_nearest = _cache_get_nearest

        # ---- initial summaries (no filter) ----
        self.chrom_counts, self.chrom_retained = statewise_genomes_summary_given_DI(
            self.dPol, float("-inf")
        )
        self.summaries = summaries_from_statewise_counts(self.chrom_counts)

        self.DInumer = sum(n for n, _ in self.chrom_retained)
        self.DIdenom = sum(d for _, d in self.chrom_retained)
        self.prop = self.DInumer / self.DIdenom if self.DIdenom else 0.0

        # ---- figure & axes ----
        self.fig, self.ax = plt.subplots(figsize=(11, 4))

        colours = Flatten([['red'], diemColours[1:], ['gray']])

        self.lines = []
        for summary, colour in zip(self.summaries, colours):
            line, = self.ax.plot(summary, color=colour, marker='.')
            self.lines.append(line)

        self.ax.legend(['HI', 'HOM1', 'HET', 'HOM2', 'U'])
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Genomes summaries; no DI filter')
        self.ax.tick_params(axis='x', rotation=55)

        self._update_xticks()

        # ---- widgets ----
        self._init_widgets()

        # ---- OPTIONAL: prefill cache grid (INCREMENTAL helper) ----
        if prefill_cache:
            DI_span = get_DI_span(self.dPol)
            di_min, di_max = float(DI_span[0]), float(DI_span[1])

            if prefill_step is None:
                span = di_max - di_min
                prefill_step = span / 200.0 if span > 0 else 1.0

            if cache_tol is None:
                cache_tol = float(prefill_step) / 2.0

            # Build DI grid including endpoints
            if di_max > di_min:
                n_steps = int(np.floor((di_max - di_min) / prefill_step))
                di_values = [di_min + k * prefill_step for k in range(n_steps + 1)]
                if not di_values or di_values[-1] < di_max:
                    di_values.append(di_max)
            else:
                di_values = [di_min]

            di_grid = np.asarray(di_values, dtype=float)

            # --- incremental cache prefill ---
            inc = StatewiseDIIncrementalCache(
                self.dPol,
                di_grid=di_grid,
                progress=("text" if progress == "text" else None),
                label="GenomeSummary cache prefill",
            ).prefill()

            # Convert chrom_counts snapshots into summaries once, and store
            # NOTE: inc._snapshots keys are floats from the grid (descending fill, but dict unordered)
            for di_key, (chrom_counts, chrom_retained) in inc._snapshots.items():
                summaries = summaries_from_statewise_counts(chrom_counts)
                self._cache_set(di_key, (summaries, chrom_retained))

            self._cache_tol = float(cache_tol)
        else:
            self._cache_tol = 0.0

        # ---- coordinate display ----
        self._install_format_coord()

        plt.show()

    # ---------------- helpers ----------------

    def _update_xticks(self):
        self.ax.set_xticks(
            np.arange(len(self.IndNickNames)),
            np.array(self.IndNickNames)[self.indHIorder],
            rotation=55,
            fontsize=self.indNameFont,
            horizontalalignment='right'
        )

    def _install_format_coord(self):
        n = len(self.dPol.indNames)
        tolerance = 0.03  # vertical proximity in y-units

        def format_coord(x, y):
            fallback = "\u2007" * 30
            i = int(round(x))
            if i < 0 or i >= n:
                return fallback

            for summary in self.summaries:
                y0 = summary[self.indHIorder][i]
                if abs(y - y0) < tolerance:
                    return f"IndID: {self.dPol.indNames[self.indHIorder[i]]}"
            return fallback

        self.ax.format_coord = format_coord

    # ---------------- widgets ----------------

    def _init_widgets(self):
        DI_span = get_DI_span(self.dPol)

        self.fig.subplots_adjust(bottom=0.3)

        # DI slider
        DI_box = self.fig.add_axes([0.2, 0.1, 0.65, 0.03])
        self.DI_slider = Slider(
            ax=DI_box,
            label='DI',
            valmin=DI_span[0],
            valmax=DI_span[1],
            valinit=DI_span[0],
        )
        self.DI_slider.on_changed(self.DIupdate)

        # Font slider
        FONT_box = self.fig.add_axes([0.25, 0.025, 0.1, 0.04])
        self.FONT_slider = Slider(
            ax=FONT_box,
            label='IndLabels font',
            valmin=1,
            valmax=16,
            valinit=self.indNameFont,
        )
        self.FONT_slider.on_changed(self.FONTupdate)

        # Reorder button
        reorderBox = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        self.reo_button = Button(
            reorderBox,
            'Reorder by HI',
            hovercolor='0.975',
            color='red'
        )
        self.reo_button.on_clicked(self.reorder)

    # ---------------- callbacks ----------------

    def DIupdate(self, val):
        payload = None
        if self._cache_tol > 0:
            payload = self._cache_get_nearest(val, self._cache_tol)

        if payload is None:
            # lazy compute + store
            chrom_counts, chrom_retained = statewise_genomes_summary_given_DI(self.dPol, val)
            summaries = summaries_from_statewise_counts(chrom_counts)
            payload = (summaries, chrom_retained)
            self._cache_set(val, payload)

        self.summaries, self.chrom_retained = payload

        self.DInumer = sum(n for n, _ in self.chrom_retained)
        self.DIdenom = sum(d for _, d in self.chrom_retained)
        self.prop = self.DInumer / self.DIdenom if self.DIdenom else 0.0

        self.ax.set_title(
            "Genomes summaries DI ≥ {:.2f}  {} SNVs  ({:.1f}% divergent across barrier)"
            .format(val, self.DInumer, 100 * self.prop)
        )

        for line, summary in zip(self.lines, self.summaries):
            line.set_ydata(summary[self.indHIorder])

        self.fig.canvas.draw_idle()

    def reorder(self, event):
        self.indHIorder = np.argsort(self.summaries[0])  # HI
        self._update_xticks()

        for line, summary in zip(self.lines, self.summaries):
            line.set_ydata(summary[self.indHIorder])

        self.fig.canvas.draw_idle()

    def FONTupdate(self, val):
        self.indNameFont = int(val)
        self._update_xticks()
        self.fig.canvas.draw_idle()

#________________________________________ END GenomeSummariesPlot ___________________



#________________________________________ START GenomeMultiSummaryPlot ___________________



class GenomeMultiSummaryPlot:
    """
    Plots genome summaries per chromosome with DI filtering and interactive widgets.
    These summaries include HI, HOM1, HET, HOM2, and U proportions per individual.
    Cursor hover displays individual IDs.
    Reorder button sorts individuals by global HI given current DI filter.

    Drop-in extension:
      - optional incremental cache prefill using StatewiseDIIncrementalCache
      - DI slider uses cached results (nearest match within tol)
      - keeps existing hover behaviour and plot style

    Args:
        dPol: DiemType object containing genomic data.
        chrom_indices: List of chromosome indices to plot.
        max_cols: max subplot columns.
        prefill_cache: precompute incremental cache over DI grid.
        prefill_step: DI step for grid (defaults to span/200).
        cache_tol: nearest-cache tolerance (defaults to prefill_step/2).
        progress: "text" | "none"
    """

    def __init__(
        self,
        dPol,
        chrom_indices,
        max_cols=3,
        *,
        prefill_cache=False,
        prefill_step=None,
        cache_tol=None,
        progress=None,# "text" | None
    ):
        self.dPol = dPol
        self.IndNickNames = [Ind_Nickname(name) for name in dPol.indNames]
        self.ChrNickNames = [Chr_Nickname(name) for name in dPol.chrNames]
        self.max_cols = max_cols

        # ---- validate chromosomes ----
        self.chrom_indices = self._validate_chrom_indices(chrom_indices)

        # ---- ordering state ----
        self.indNameFont = 6

        # ---- cache (per instance) ----
        # cache maps DI_value(float) -> payload
        # payload = (global_summaries, chrom_retained, per_chr_summaries_dict)
        self._cache = {}
        self._cache_keys_sorted = []

        def _cache_set(di, payload):
            di = float(di)
            if di not in self._cache:
                bisect.insort(self._cache_keys_sorted, di)
            self._cache[di] = payload

        def _cache_get_nearest(di, tol):
            if not self._cache_keys_sorted:
                return None
            di = float(di)
            keys = self._cache_keys_sorted
            j = bisect.bisect_left(keys, di)

            candidates = []
            if 0 <= j < len(keys):
                candidates.append(keys[j])
            if 0 <= j - 1 < len(keys):
                candidates.append(keys[j - 1])

            best = None
            best_dist = None
            for k in candidates:
                dist = abs(k - di)
                if best_dist is None or dist < best_dist:
                    best = k
                    best_dist = dist

            if best is None or best_dist is None or best_dist > tol:
                return None
            return self._cache[best]

        self._cache_set = _cache_set
        self._cache_get_nearest = _cache_get_nearest

        # ---- initial DI snapshot (no filter) ----
        self.chrom_counts, self.chrom_retained = statewise_genomes_summary_given_DI(
            self.dPol, float("-inf")
        )

        # authoritative whole-genome summaries (from statewise counts)
        global_summaries = summaries_from_statewise_counts(self.chrom_counts)
        self.global_HI = global_summaries[0]
        self.indHIorder = np.argsort(self.global_HI)

        # build per-chromosome summaries once (initial DI)
        self.chrom_summaries = {}
        for idx in self.chrom_indices:
            self.chrom_summaries[idx] = summaries_from_statewise_counts([self.chrom_counts[idx]])

        # ---- grid layout ----
        n_plots = len(self.chrom_indices)
        n_cols = min(self.max_cols, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig_w = 4.5 * n_cols
        fig_h = 3.5 * n_rows

        self.fig, self.axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_w, fig_h),
            squeeze=False,
            sharey=True
        )

        self.fig.subplots_adjust(
            left=0.06,
            right=0.98,
            top=0.92,
            bottom=0.45,
            hspace=0.60,
            wspace=0.25
        )

        # ---- draw plots ----
        self.lines = {}
        axes_flat = self.axes.flatten()

        colours = Flatten([['red'], diemColours[1:], ['gray']])
        global_hi_colour = "cyan"

        self.chrom_axes = {}
        self.global_hi_lines = {}

        for ax, chrom_idx in zip(axes_flat, self.chrom_indices):
            self.chrom_axes[chrom_idx] = ax
            summaries = self.chrom_summaries[chrom_idx]

            chrom_lines = []
            for summary, colour in zip(summaries, colours):
                line, = ax.plot(
                    summary[self.indHIorder],
                    color=colour,
                    marker='.',
                    linewidth=0.8
                )
                chrom_lines.append(line)

            self.lines[chrom_idx] = chrom_lines

            # global HI overlay
            global_hi_line, = ax.plot(
                self.global_HI[self.indHIorder],
                color=global_hi_colour,
                linestyle="-",
                linewidth=1.5,
                alpha=0.8,
            )
            self.global_hi_lines[chrom_idx] = global_hi_line

            ax.set_ylim(0, 1)
            num, denom = self.chrom_retained[chrom_idx]
            ax.set_title(f"{self.ChrNickNames[chrom_idx]} | {num:,}/{denom:,} sites", fontsize=10)
            ax.tick_params(axis='x', rotation=55)

            ax.set_xticks(
                np.arange(len(self.IndNickNames)),
                np.array(self.IndNickNames)[self.indHIorder],
                fontsize=self.indNameFont,
                ha='right'
            )

        # hide unused axes
        for ax in axes_flat[len(self.chrom_indices):]:
            ax.axis("off")

        # legend once
        axes_flat[0].legend(
            ['HIc', 'HOM1', 'HET', 'HOM2', 'U', 'HIg'],
            fontsize=8,
            frameon=False
        )

        # ---- widgets ----
        self._init_widgets()
        # Force an initial DI computation at the *max* endpoint (or whatever value you want) EURG!
        #DI_span = get_DI_span(self.dPol)
        #self._on_DI_change(float(DI_span[1]))

        # ---- OPTIONAL: incremental cache prefill ----
        if prefill_cache:
            DI_span = get_DI_span(self.dPol)
            di_min, di_max = float(DI_span[0]), float(DI_span[1])

            if prefill_step is None:
                span = di_max - di_min
                prefill_step = span / 200.0 if span > 0 else 1.0

            if cache_tol is None:
                cache_tol = float(prefill_step) / 2.0

            # Build DI grid including endpoints
            if di_max > di_min:
                n_steps = int(np.floor((di_max - di_min) / prefill_step))
                di_grid = di_min + prefill_step * np.arange(n_steps + 1, dtype=float)
                di_grid = np.clip(di_grid, di_min, di_max)
                di_grid[-1] = di_max
            else:
                di_grid = np.array([di_min], dtype=float)

            #di_grid = np.asarray(di_values, dtype=float)

            inc = StatewiseDIIncrementalCache(
                self.dPol,
                di_grid=di_grid,
                progress=("text" if progress == "text" else None),
                label="GenomeMultiSummary cache prefill",
            ).prefill()

            # Store only what this plot needs:
            # - global summaries (all chr)
            # - chrom_retained (all chr)
            # - per-chr summaries for selected chrom_indices
            for di_key, (chrom_counts, chrom_retained) in inc._snapshots.items():
                global_summaries = summaries_from_statewise_counts(chrom_counts)

                per_chr = {}
                for idx in self.chrom_indices:
                    per_chr[idx] = summaries_from_statewise_counts([chrom_counts[idx]])

                self._cache_set(di_key, (global_summaries, chrom_retained, per_chr))

            self._cache_tol = float(cache_tol)
        else:
            self._cache_tol = 0.0

        # ---- coordinate display ----
        self._install_format_coord()

        plt.show()

    # ==================================================
    # Validation
    # ==================================================

    def _validate_chrom_indices(self, chrom_indices):
        max_idx = len(self.dPol.chrLengths) - 1
        valid, rejected = [], []

        for idx in chrom_indices:
            if isinstance(idx, (int, np.integer)) and 0 <= int(idx) <= max_idx:
                valid.append(int(idx))
            else:
                rejected.append(idx)

        if rejected:
            print("GenomeMultiSummaryPlot: rejected chromosome indices:", rejected)

        if not valid:
            raise ValueError("GenomeMultiSummaryPlot: no valid chromosome indices.")

        return valid

    # ==================================================
    # Hover
    # ==================================================

    def _install_format_coord(self):
        n = len(self.dPol.indNames)
        tolerance = 0.03

        axes_flat = self.axes.flatten()[:len(self.chrom_indices)]
        for ax, chrom_idx in zip(axes_flat, self.chrom_indices):
            chrom_lines = self.lines[chrom_idx]

            def make_format_coord(chrom_lines_local):
                def format_coord(x, y):
                    fallback = "\u2007" * 30
                    i = int(round(x))
                    if i < 0 or i >= n:
                        return fallback

                    for line in chrom_lines_local:
                        ydata = line.get_ydata()
                        if abs(y - ydata[i]) < tolerance:
                            return f"IndID: {self.dPol.indNames[self.indHIorder[i]]}"
                    return fallback
                return format_coord

            ax.format_coord = make_format_coord(chrom_lines)

    # ==================================================
    # Widgets
    # ==================================================

    def _init_widgets(self):
        DI_span = get_DI_span(self.dPol)

        ax_DI = self.fig.add_axes([0.15, 0.18, 0.7, 0.03])
        self.DI_slider = Slider(ax_DI, "DI", DI_span[0], DI_span[1], valinit=DI_span[0])
        self.DI_slider.on_changed(self._on_DI_change)

        ax_FS = self.fig.add_axes([0.25, 0.12, 0.1, 0.03])
        self.FONT_slider = Slider(ax_FS, "IndLabel font", 4, 16,
                                  valinit=self.indNameFont, valstep=1)
        self.FONT_slider.on_changed(self._on_font_change)

        ax_RE = self.fig.add_axes([0.75, 0.115, 0.15, 0.045])
        self.reorder_button = Button(
            ax_RE,
            "Reorder by global HI",
            hovercolor="0.95",
            color="cyan"
        )
        self.reorder_button.on_clicked(self._on_reorder)

    # ==================================================
    # Callbacks
    # ==================================================

    def _on_DI_change(self, val):
        payload = None
        if self._cache_tol > 0:
            payload = self._cache_get_nearest(val, self._cache_tol)

        if payload is None:
            # Lazy compute full statewise summary once
            chrom_counts, chrom_retained = statewise_genomes_summary_given_DI(self.dPol, val)
            global_summaries = summaries_from_statewise_counts(chrom_counts)

            per_chr = {}
            for idx in self.chrom_indices:
                per_chr[idx] = summaries_from_statewise_counts([chrom_counts[idx]])

            payload = (global_summaries, chrom_retained, per_chr)
            self._cache_set(val, payload)

        global_summaries, self.chrom_retained, per_chr = payload
        self.global_HI = global_summaries[0]

        # Update plotted data for each chromosome
        for idx in self.chrom_indices:
            summaries = per_chr[idx]
            self.chrom_summaries[idx] = summaries

            for line, summary in zip(self.lines[idx], summaries):
                line.set_ydata(summary[self.indHIorder])

            self.global_hi_lines[idx].set_ydata(self.global_HI[self.indHIorder])

            num, denom = self.chrom_retained[idx]
            self.chrom_axes[idx].set_title(
                f"{self.ChrNickNames[idx]} | {num:,}/{denom:,} sites",
                fontsize=10
            )

        self.fig.canvas.draw_idle()

    def _on_font_change(self, val):
        self.indNameFont = int(val)
        labels = np.array(self.IndNickNames)[self.indHIorder]

        for ax in self.axes.flatten()[:len(self.chrom_indices)]:
            ax.set_xticklabels(labels, fontsize=self.indNameFont)

        self.fig.canvas.draw_idle()

    def _on_reorder(self, event=None):
        """
        Reorder individuals by *current* whole-genome HI (statewise),
        without recomputing anything expensive.
        """
        self.indHIorder = np.argsort(self.global_HI)

        labels = np.array(self.IndNickNames)[self.indHIorder]

        for idx in self.chrom_indices:
            # global HI overlay
            self.global_hi_lines[idx].set_ydata(self.global_HI[self.indHIorder])

            # chromosome lines
            summaries = self.chrom_summaries[idx]
            for line, summary in zip(self.lines[idx], summaries):
                line.set_ydata(summary[self.indHIorder])

        for ax in self.axes.flatten()[:len(self.chrom_indices)]:
            ax.set_xticks(
                np.arange(len(labels)),
                labels,
                fontsize=self.indNameFont,
                ha="right"
            )

        self.fig.canvas.draw_idle()

"""________________________________________ END GenomeMultiSummaryPlot ___________________"""


"""________________________________________ START GenomicDeFinettiPlot ___________________"""


class GenomicDeFinettiPlot:
    """
    Plots a genomic de Finetti plot with DI filtering and interactive widgets.
    Cursor hover displays individual IDs.

    c.f. 
    Figure 2, Figure 4:
    Petružela, J., Nürnberger, B., Ribas, A., Koutsovoulos, G., Čížková, 
    D., Fornůsková, A., Aghová, T., Blaxter, M., de Bellocq, J.G. and Baird, S.J.E. 
    (2025), Comparative Genomic Analysis of Co-Occurring Hybrid Zones 
    of House Mouse Parasites Pneumocystis murina and Syphacia obvelata 
    Using Genome Polarisation. Mol Ecol, 34: e70044. https://doi.org/10.1111/mec.70044

    Figure 4:
    Ebdon, S., Laetsch, D. R., Vila, R., Baird, S. J. E., & Lohse, K. (2025). 
    Genomic regions of current low hybridisation mark long-term barriers to gene flow 
    in scarce swallowtail butterflies. PLoS Genetics, 21(4), 30. 
    doi:https://doi.org/10.1371/journal.pgen.1011655

    Drop-in extension:
      - optional incremental cache prefill using StatewiseDIIncrementalCache
      - DI slider uses cached results (nearest match within tol)
      - keeps output + hover behaviour the same

    Uses:
      - summaries_from_statewise_counts(statewise counts)
      - StatewiseDIIncrementalCache (fast prefill)

    Args:
        dPol: DiemType object containing genomic data.
    """

    def __init__(
        self,
        dPol,
        *,
        prefill_cache=False,      # NEW
        prefill_step=None,        # NEW
        cache_tol=None,           # NEW
        progress=None,          # NEW: "text" | "none"
    ):
        self.dPol = dPol

        # ---- initial state ----
        self.marker_size = 60
        self.indHIorder = np.arange(len(dPol.indNames))

        # ---- cache (per instance) ----
        # maps DI_value(float) -> (summaries, DInumer, DIdenom)
        self._cache = {}
        self._cache_keys_sorted = []

        def _cache_set(di, payload):
            di = float(di)
            if di not in self._cache:
                bisect.insort(self._cache_keys_sorted, di)
            self._cache[di] = payload

        def _cache_get_nearest(di, tol):
            if not self._cache_keys_sorted:
                return None
            di = float(di)
            keys = self._cache_keys_sorted
            j = bisect.bisect_left(keys, di)

            candidates = []
            if 0 <= j < len(keys):
                candidates.append(keys[j])
            if 0 <= j - 1 < len(keys):
                candidates.append(keys[j - 1])

            best = None
            best_dist = None
            for k in candidates:
                dist = abs(k - di)
                if best_dist is None or dist < best_dist:
                    best = k
                    best_dist = dist

            if best is None or best_dist is None or best_dist > tol:
                return None
            return self._cache[best]

        self._cache_set = _cache_set
        self._cache_get_nearest = _cache_get_nearest

        # ---- initial summaries (no DI filter) ----
        # Keep your original semantics, but compute via statewise so cache path matches exactly.
        chrom_counts, chrom_retained = statewise_genomes_summary_given_DI(self.dPol, float("-inf"))
        self.summaries = summaries_from_statewise_counts(chrom_counts)
        self.DInumer = sum(n for n, _ in chrom_retained)
        self.DIdenom = sum(d for _, d in chrom_retained)

        # unpack summaries
        self.HOM1 = self.summaries[1]
        self.HET  = self.summaries[2]
        self.HOM2 = self.summaries[3]
        self.U    = self.summaries[4]

        # ---- figure & axes ----
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self._setup_axes()
        self.ax.set_title('Genomic de Finetti; no DI filter')

        # background
        self._draw_triangle()
        self._draw_hwe_curve()

        # points
        self.scatter = self._draw_points()

        # widgets
        self._init_widgets()

        # ---- OPTIONAL: prefill incremental cache ----
        if prefill_cache:
            DI_span = get_DI_span(self.dPol)
            di_min, di_max = float(DI_span[0]), float(DI_span[1])

            if prefill_step is None:
                span = di_max - di_min
                prefill_step = span / 200.0 if span > 0 else 1.0

            if cache_tol is None:
                cache_tol = float(prefill_step) / 2.0

            # Build DI grid including endpoints
            if di_max > di_min:
                n_steps = int(np.floor((di_max - di_min) / prefill_step))
                di_values = [di_min + k * prefill_step for k in range(n_steps + 1)]
                if not di_values or di_values[-1] < di_max:
                    di_values.append(di_max)
            else:
                di_values = [di_min]

            di_grid = np.asarray(di_values, dtype=float)

            inc = StatewiseDIIncrementalCache(
                self.dPol,
                di_grid=di_grid,
                progress=("text" if progress == "text" else None),
                label="GenomicDeFinetti cache prefill",
            ).prefill()

            # Store summaries snapshots only (small memory)
            for di_key, (chrom_counts, chrom_retained) in inc._snapshots.items():
                summaries = summaries_from_statewise_counts(chrom_counts)
                DInumer = sum(n for n, _ in chrom_retained)
                DIdenom = sum(d for _, d in chrom_retained)
                self._cache_set(di_key, (summaries, DInumer, DIdenom))

            self._cache_tol = float(cache_tol)
        else:
            self._cache_tol = 0.0

        # coordinate display
        self._install_format_coord()

        plt.show()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    @staticmethod
    def _to_triangle_coords(hom1, het, hom2):
        x = hom2 + 0.5 * het
        y = (np.sqrt(3) / 2) * het
        return x, y

    def _update_title(self, DIval):
        prop = self.DInumer / self.DIdenom if self.DIdenom > 0 else 0.0
        self.ax.set_title(
            "Genomic de Finetti plot  DI ≥ {:.2f}  {} SNVs  ({:.1f}% divergent across barrier)"
            .format(DIval, self.DInumer, 100 * prop),
            fontsize=12,
            pad=12
        )

    # --------------------------------------------------
    # Axes / background
    # --------------------------------------------------

    def _setup_axes(self):
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(-0.05, np.sqrt(3) / 2 + 0.05)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        self.ax.set_title("Genomic de Finetti plot")

    def _draw_triangle(self):
        h = np.sqrt(3) / 2
        triangle = np.array([[0, 0], [1, 0], [0.5, h]])
        self.ax.add_patch(
            Polygon(triangle, closed=True, fill=False, lw=1.2, color="black")
        )

        self.ax.text(0, -0.04, "HOM1", ha="center", va="top", fontsize=9)
        self.ax.text(1, -0.04, "HOM2", ha="center", va="top", fontsize=9)
        self.ax.text(0.5, h + 0.03, "HET", ha="center", va="bottom", fontsize=9)

    def _draw_hwe_curve(self):
        p = np.linspace(0, 1, 400)
        hom1 = p**2
        het  = 2 * p * (1 - p)
        hom2 = (1 - p)**2

        x, y = self._to_triangle_coords(hom1, het, hom2)
        self.ax.plot(x, y, color="black", lw=0.8, alpha=0.5)

    # --------------------------------------------------
    # Points
    # --------------------------------------------------

    def _blend_colours(self):
        weights = np.column_stack([self.HOM1, self.HET, self.HOM2, self.U])

        base_colours = np.array([
            to_rgb(diemColours[1]),  # HOM1
            to_rgb(diemColours[2]),  # HET
            to_rgb(diemColours[3]),  # HOM2
            to_rgb(diemColours[0]),  # U
        ])  # (4,3)

        rgb = weights @ base_colours
        rgb = np.clip(rgb, 0.0, 1.0)
        return rgb

    def _draw_points(self):
        x, y = self._to_triangle_coords(
            self.HOM1[self.indHIorder],
            self.HET[self.indHIorder],
            self.HOM2[self.indHIorder],
        )
        colours = self._blend_colours()[self.indHIorder]

        return self.ax.scatter(
            x, y,
            s=self.marker_size,
            c=colours,
            edgecolor="black",
            linewidth=0.3,
        )

    def _update_points(self):
        self.HOM1 = self.summaries[1]
        self.HET  = self.summaries[2]
        self.HOM2 = self.summaries[3]
        self.U    = self.summaries[4]

        x, y = self._to_triangle_coords(
            self.HOM1[self.indHIorder],
            self.HET[self.indHIorder],
            self.HOM2[self.indHIorder],
        )

        self.scatter.set_offsets(np.column_stack([x, y]))
        self.scatter.set_facecolors(self._blend_colours()[self.indHIorder])

    # --------------------------------------------------
    # Widgets
    # --------------------------------------------------

    def _init_widgets(self):
        DI_span = get_DI_span(self.dPol)
        self.fig.subplots_adjust(bottom=0.25)

        ax_DI = self.fig.add_axes([0.15, 0.12, 0.7, 0.03])
        self.DI_slider = Slider(
            ax_DI, "DI",
            DI_span[0], DI_span[1],
            valinit=DI_span[0]
        )
        self.DI_slider.on_changed(self.DIupdate)

        ax_SZ = self.fig.add_axes([0.25, 0.10, 0.1, 0.03])
        self.size_slider = Slider(
            ax_SZ, "Symbol size",
            10, 300,
            valinit=self.marker_size
        )
        self.size_slider.on_changed(self.SIZEupdate)

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------

    def DIupdate(self, val):
        payload = None
        if self._cache_tol > 0:
            payload = self._cache_get_nearest(val, self._cache_tol)

        if payload is None:
            # Lazy compute via statewise (consistent with cached path)
            chrom_counts, chrom_retained = statewise_genomes_summary_given_DI(self.dPol, val)
            summaries = summaries_from_statewise_counts(chrom_counts)
            DInumer = sum(n for n, _ in chrom_retained)
            DIdenom = sum(d for _, d in chrom_retained)

            payload = (summaries, DInumer, DIdenom)
            self._cache_set(val, payload)

        self.summaries, self.DInumer, self.DIdenom = payload

        # unpack (important)
        self.HOM1 = self.summaries[1]
        self.HET  = self.summaries[2]
        self.HOM2 = self.summaries[3]
        self.U    = self.summaries[4]

        self._update_points()
        self._update_title(val)
        self.fig.canvas.draw_idle()

    def SIZEupdate(self, val):
        self.marker_size = val
        self.scatter.set_sizes(np.full(len(self.dPol.indNames), val))
        self.fig.canvas.draw_idle()

    # --------------------------------------------------
    # Coordinate display
    # --------------------------------------------------

    def _install_format_coord(self):
        tol = 0.03

        def format_coord(x, y):
            fallback = "\u2007" * 30
            pts = self.scatter.get_offsets()
            d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            i = np.argmin(d)
            if d[i] < tol:
                return f"IndID: {self.dPol.indNames[self.indHIorder[i]]}"
            return fallback

        self.ax.format_coord = format_coord

"""________________________________________ END GenomicDeFinettiPlot ___________________"""


"""________________________________________ START GenomicMultiDeFinettiPlot ___________________"""


class GenomicMultiDeFinettiPlot:
    """
    Multiple de Finetti plots, one per chromosome,
    all controlled by a shared DI slider and size slider.

    Uses statewise_genomes_summary_given_DI

    c.f. 
    Figure 2, Figure 4:
    Petružela, J., Nürnberger, B., Ribas, A., Koutsovoulos, G., Čížková, 
    D., Fornůsková, A., Aghová, T., Blaxter, M., de Bellocq, J.G. and Baird, S.J.E. 
    (2025), Comparative Genomic Analysis of Co-Occurring Hybrid Zones 
    of House Mouse Parasites Pneumocystis murina and Syphacia obvelata 
    Using Genome Polarisation. Mol Ecol, 34: e70044. https://doi.org/10.1111/mec.70044
    
    Figure 4:
    Ebdon, S., Laetsch, D. R., Vila, R., Baird, S. J. E., & Lohse, K. (2025). 
    Genomic regions of current low hybridisation mark long-term barriers to gene flow 
    in scarce swallowtail butterflies. PLoS Genetics, 21(4), 30. 
    doi:https://doi.org/10.1371/journal.pgen.1011655

    Drop-in extension:
      - optional incremental cache prefill using StatewiseDIIncrementalCache
      - DI slider uses cached results (nearest match within tol)
      - output + hover semantics unchanged

    Uses statewise_genomes_summary_given_DI + summaries_from_statewise_counts.

    Args:
        dPol: DiemType object containing genomic data.
        chrom_indices: List of chromosome indices to plot.
    """

    def __init__(
        self,
        dPol,
        chrom_indices,
        max_cols=3,
        *,
        prefill_cache=False,      # NEW
        prefill_step=None,        # NEW
        cache_tol=None,           # NEW
        progress=None,          # NEW: "text" | "none"
    ):
        self.dPol = dPol
        self.chrom_indices = self._validate_chrom_indices(chrom_indices)
        self.ChrNickNames = [Chr_Nickname(name) for name in dPol.chrNames]
        self.max_cols = max_cols

        self.marker_size = 60
        self.n_ind = len(dPol.indNames)
        self.indHIorder = np.arange(self.n_ind)

        # ---------- cache (per instance) ----------
        # maps DI(float) -> (chrom_counts, chrom_retained)
        self._cache = {}
        self._cache_keys_sorted = []
        self._cache_tol = 0.0

        def _cache_set(di, payload):
            di = float(di)
            if di not in self._cache:
                bisect.insort(self._cache_keys_sorted, di)
            self._cache[di] = payload

        def _cache_get_nearest(di, tol):
            if not self._cache_keys_sorted:
                return None
            di = float(di)
            keys = self._cache_keys_sorted
            j = bisect.bisect_left(keys, di)

            candidates = []
            if 0 <= j < len(keys):
                candidates.append(keys[j])
            if 0 <= j - 1 < len(keys):
                candidates.append(keys[j - 1])

            best = None
            best_dist = None
            for k in candidates:
                dist = abs(k - di)
                if best_dist is None or dist < best_dist:
                    best = k
                    best_dist = dist

            if best is None or best_dist is None or best_dist > tol:
                return None
            return self._cache[best]

        self._cache_set = _cache_set
        self._cache_get_nearest = _cache_get_nearest

        # ---------- initial statewise computation ----------
        self.chrom_counts, self.chrom_retained = \
            statewise_genomes_summary_given_DI(self.dPol, float("-inf"))

        # global summaries (authoritative ordering)
        self.global_summaries = summaries_from_statewise_counts(self.chrom_counts)
        self.global_HI = self.global_summaries[0]
        self.indHIorder = np.argsort(self.global_HI)

        # ---------- layout ----------
        n_plots = len(self.chrom_indices)
        n_cols = min(self.max_cols, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))

        self.fig, self.axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.8 * n_cols, 4.6 * n_rows),
            squeeze=False
        )

        self.fig.subplots_adjust(
            left=0.06, right=0.98,
            top=0.92, bottom=0.32,
            hspace=0.45, wspace=0.25
        )

        # ---------- draw ----------
        self.scatters = {}
        self.chrom_axes = {}

        axes_flat = self.axes.flatten()

        for ax, idx in zip(axes_flat, self.chrom_indices):
            self.chrom_axes[idx] = ax
            self._setup_axes(ax)
            self._draw_triangle(ax)
            self._draw_hwe_curve(ax)

            summaries = summaries_from_statewise_counts([self.chrom_counts[idx]])
            _, HOM1, HET, HOM2, U = summaries

            x, y = self._to_triangle_coords(
                HOM1[self.indHIorder],
                HET[self.indHIorder],
                HOM2[self.indHIorder]
            )

            colours = self._blend_colours(HOM1, HET, HOM2, U)

            sc = ax.scatter(
                x, y,
                s=self.marker_size,
                c=colours[self.indHIorder],
                edgecolor="black",
                linewidth=0.3
            )

            num, denom = self.chrom_retained[idx]
            ax.set_title(f"{self.ChrNickNames[idx]} | {num:,}/{denom:,} sites", fontsize=10)

            self.scatters[idx] = sc

        for ax in axes_flat[len(self.chrom_indices):]:
            ax.axis("off")

        # ---------- widgets ----------
        self._init_widgets()

        # ---------- OPTIONAL: prefill incremental cache ----------
        if prefill_cache:
            DI_span = get_DI_span(self.dPol)
            di_min, di_max = float(DI_span[0]), float(DI_span[1])

            if prefill_step is None:
                span = di_max - di_min
                prefill_step = span / 200.0 if span > 0 else 1.0

            if cache_tol is None:
                cache_tol = float(prefill_step) / 2.0

            # Build DI grid including endpoints
            if di_max > di_min:
                n_steps = int(np.floor((di_max - di_min) / prefill_step))
                di_values = [di_min + k * prefill_step for k in range(n_steps + 1)]
                if not di_values or di_values[-1] < di_max:
                    di_values.append(di_max)
            else:
                di_values = [di_min]

            di_grid = np.asarray(di_values, dtype=float)

            inc = StatewiseDIIncrementalCache(
                self.dPol,
                di_grid=di_grid,
                progress=("text" if progress == "text" else None),
                label="GenomicMultiDeFinetti cache prefill",
            ).prefill()

            # Store snapshots (small memory compared to per-site)
            for di_key, payload in inc._snapshots.items():
                self._cache_set(di_key, payload)

            self._cache_tol = float(cache_tol)

        # ---------- hover ----------
        self._install_format_coord()

        plt.show()

    # ======================================================
    # Helpers
    # ======================================================

    @staticmethod
    def _to_triangle_coords(hom1, het, hom2):
        x = hom2 + 0.5 * het
        y = (np.sqrt(3) / 2) * het
        return x, y

    def _blend_colours(self, HOM1, HET, HOM2, U):
        weights = np.column_stack([HOM1, HET, HOM2, U])
        base = np.array([
            to_rgb(diemColours[1]),
            to_rgb(diemColours[2]),
            to_rgb(diemColours[3]),
            to_rgb(diemColours[0]),
        ])
        return np.clip(weights @ base, 0, 1)

    def _setup_axes(self, ax):
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, np.sqrt(3)/2 + 0.05)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def _draw_triangle(self, ax):
        h = np.sqrt(3)/2
        ax.add_patch(Polygon([[0,0],[1,0],[0.5,h]], fill=False, lw=1.2))
        ax.text(0, -0.04, "HOM1", ha="center", va="top", fontsize=8)
        ax.text(1, -0.04, "HOM2", ha="center", va="top", fontsize=8)
        ax.text(0.5, h + 0.03, "HET", ha="center", va="bottom", fontsize=8)

    def _draw_hwe_curve(self, ax):
        p = np.linspace(0,1,400)
        x, y = self._to_triangle_coords(p*p, 2*p*(1-p), (1-p)**2)
        ax.plot(x, y, color="black", lw=0.8, alpha=0.5)

    # ======================================================
    # Widgets
    # ======================================================

    def _init_widgets(self):
        DI_span = get_DI_span(self.dPol)

        ax_DI = self.fig.add_axes([0.15, 0.20, 0.70, 0.035])
        self.DI_slider = Slider(ax_DI, "DI", *DI_span, valinit=DI_span[0])
        self.DI_slider.on_changed(self._on_DI_change)

        ax_SZ = self.fig.add_axes([0.25, 0.16, 0.1, 0.03])
        self.size_slider = Slider(ax_SZ, "Symbol size", 10, 300, valinit=self.marker_size)
        self.size_slider.on_changed(self._on_size_change)

    # ======================================================
    # Callbacks
    # ======================================================

    def _on_DI_change(self, val):
        payload = None
        if self._cache_tol > 0:
            payload = self._cache_get_nearest(val, self._cache_tol)

        if payload is None:
            # fallback (no prefill or outside tol): compute and (optionally) cache
            chrom_counts, chrom_retained = statewise_genomes_summary_given_DI(self.dPol, val)
            payload = (chrom_counts, chrom_retained)
            # opportunistic cache
            self._cache_set(val, payload)

        self.chrom_counts, self.chrom_retained = payload

        # global summaries (authoritative ordering)
        self.global_summaries = summaries_from_statewise_counts(self.chrom_counts)
        self.indHIorder = np.argsort(self.global_summaries[0])

        totNumer = 0
        totDenom = 0

        for idx in self.chrom_indices:
            summaries = summaries_from_statewise_counts([self.chrom_counts[idx]])
            _, H1, Ht, H2, U = summaries

            x, y = self._to_triangle_coords(
                H1[self.indHIorder],
                Ht[self.indHIorder],
                H2[self.indHIorder]
            )

            sc = self.scatters[idx]
            sc.set_offsets(np.column_stack([x, y]))
            sc.set_facecolors(self._blend_colours(H1, Ht, H2, U)[self.indHIorder])

            num, denom = self.chrom_retained[idx]
            self.chrom_axes[idx].set_title(
                f"{self.ChrNickNames[idx]} | {num:,}/{denom:,} sites", fontsize=10
            )

            totNumer += num
            totDenom += denom

        # (you previously computed prop but didn't display it; keep behaviour unchanged)
        self.fig.canvas.draw_idle()

    def _on_size_change(self, val):
        self.marker_size = int(val)
        for sc in self.scatters.values():
            sc.set_sizes(np.full(self.n_ind, self.marker_size))
        self.fig.canvas.draw_idle()

    # ======================================================
    # Hover
    # ======================================================

    def _install_format_coord(self):
        tol = 0.03
        names = self.dPol.indNames

        for idx, sc in self.scatters.items():
            ax = self.chrom_axes[idx]

            def make_fmt(scatter):
                def fmt(x, y):
                    pts = scatter.get_offsets()
                    d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
                    i = np.argmin(d)
                    if d[i] < tol:
                        return f"IndID: {names[self.indHIorder[i]]}"
                    return "\u2007" * 30
                return fmt

            ax.format_coord = make_fmt(sc)

    # ======================================================
    # Validation
    # ======================================================

    def _validate_chrom_indices(self, chrom_indices):
        max_idx = len(self.dPol.chrLengths) - 1
        valid = []
        rejected = []
        for i in chrom_indices:
            try:
                ii = int(i)
                if 0 <= ii <= max_idx:
                    valid.append(ii)
                else:
                    rejected.append(i)
            except Exception:
                rejected.append(i)

        if rejected:
            print("GenomicMultiDeFinettiPlot: rejected chromosome indices:", rejected)
        if not valid:
            raise ValueError("No valid chromosome indices")
        return valid

"""________________________________________ END GenomicMultiDeFinettiPlot ___________________"""


"""________________________________________ START GenomicContributionsPlot ___________________"""


class GenomicContributionsPlot:
    """
    Plots per-chromosome genomic contributions (HOM1, HET, HOM2, U, excluded)
    with DI filtering and interactive widgets.

    Drop-in extension:
      - optional incremental cache prefill using StatewiseDIIncrementalCache
      - DI slider uses cached statewise snapshots (nearest within tol)
      - output unchanged

    Uses statewise_genomes_summary_given_DI (or cached equivalent).

    Args:
    dPol: DiemType object containing genomic data.
    """

    def __init__(
        self,
        dPol,
        chrom_indices=None,
        *,
        prefill_cache=False,      # NEW
        prefill_step=None,        # NEW
        cache_tol=None,           # NEW
        progress=None,          # NEW: "text" | "none"
    ):
        self.dPol = dPol
        self.chrom_indices = chrom_indices
        self.fontsize = 8

        # ---- cache (per instance) ----
        # maps DI(float) -> (chrom_counts, chrom_retained)
        self._cache = {}
        self._cache_keys_sorted = []
        self._cache_tol = 0.0

        def _cache_set(di, payload):
            di = float(di)
            if di not in self._cache:
                bisect.insort(self._cache_keys_sorted, di)
            self._cache[di] = payload

        def _cache_get_nearest(di, tol):
            if not self._cache_keys_sorted:
                return None
            di = float(di)
            keys = self._cache_keys_sorted
            j = bisect.bisect_left(keys, di)

            candidates = []
            if 0 <= j < len(keys):
                candidates.append(keys[j])
            if 0 <= j - 1 < len(keys):
                candidates.append(keys[j - 1])

            best = None
            best_dist = None
            for k in candidates:
                dist = abs(k - di)
                if best_dist is None or dist < best_dist:
                    best = k
                    best_dist = dist

            if best is None or best_dist is None or best_dist > tol:
                return None
            return self._cache[best]

        self._cache_set = _cache_set
        self._cache_get_nearest = _cache_get_nearest

        # --------------------------------------------------
        # Initial compute (no filter)
        # --------------------------------------------------
        self.DInumer = 0
        self.DIdenom = 0
        self._compute_contributions(float("-inf"))

        # ---- figure & axes ----
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.format_coord = None
        self.fig.subplots_adjust(bottom=0.40, right=0.85)

        self._draw_bars()
        self._init_widgets()

        # --------------------------------------------------
        # OPTIONAL: prefill incremental cache
        # --------------------------------------------------
        if prefill_cache:
            DI_span = get_DI_span(self.dPol)
            di_min, di_max = float(DI_span[0]), float(DI_span[1])

            if prefill_step is None:
                span = di_max - di_min
                prefill_step = span / 200.0 if span > 0 else 1.0

            if cache_tol is None:
                cache_tol = float(prefill_step) / 2.0

            # Build DI grid including endpoints
            if di_max > di_min:
                n_steps = int(np.floor((di_max - di_min) / prefill_step))
                di_values = [di_min + k * prefill_step for k in range(n_steps + 1)]
                if not di_values or di_values[-1] < di_max:
                    di_values.append(di_max)
            else:
                di_values = [di_min]

            di_grid = np.asarray(di_values, dtype=float)

            inc = StatewiseDIIncrementalCache(
                self.dPol,
                di_grid=di_grid,
                progress=("text" if progress == "text" else None),
                label="GenomicContributions cache prefill",
            ).prefill()

            # Store snapshots
            for di_key, payload in inc._snapshots.items():
                self._cache_set(di_key, payload)

            self._cache_tol = float(cache_tol)

        plt.show()

    # --------------------------------------------------
    # Core computation
    # --------------------------------------------------

    def _compute_contributions(self, DIval):
        """
        Compute contributions at DIval.
        Uses cache if available (nearest within tol), otherwise computes directly.
        """

        payload = None
        if self._cache_tol > 0:
            payload = self._cache_get_nearest(DIval, self._cache_tol)

        if payload is None:
            chrom_counts, chrom_retained = statewise_genomes_summary_given_DI(self.dPol, DIval)
            payload = (chrom_counts, chrom_retained)
            # opportunistic cache (even without prefill, harmless)
            self._cache_set(DIval, payload)
        else:
            chrom_counts, chrom_retained = payload

        # --------------------------------------------------
        # Restrict chromosomes if requested (same as your code)
        # --------------------------------------------------
        if self.chrom_indices is not None:
            kept = []
            n_chr = len(chrom_counts)

            for ci in self.chrom_indices:
                if isinstance(ci, (int, np.integer)) and 0 <= int(ci) < n_chr:
                    kept.append(int(ci))

            if not kept:
                raise ValueError("GenomicContributionsPlot: no valid chromosome indices")

        else:
            kept = range(len(chrom_counts))

        # --------------------------------------------------
        # Aggregate over kept chromosomes
        # --------------------------------------------------
        self.DInumer = 0
        self.DIdenom = 0

        kept_list = list(kept)
        self.chrom_labels = []
        self.props = np.zeros((len(kept_list), 5))  # HOM1, HET, HOM2, U, excluded

        for out_i, chr_i in enumerate(kept_list):
            chr_name = Chr_Nickname(self.dPol.chrNames[chr_i])
            self.chrom_labels.append(chr_name)

            counts = chrom_counts[chr_i]
            kept_sites, total_sites = chrom_retained[chr_i]

            self.DInumer += kept_sites
            self.DIdenom += total_sites

            c0 = float(np.sum(counts["counts0"]))
            c1 = float(np.sum(counts["counts1"]))
            c2 = float(np.sum(counts["counts2"]))
            c3 = float(np.sum(counts["counts3"]))

            total_alleles = float(total_sites) * float(np.sum(self.dPol.chrPloidies[chr_i]))
            if total_alleles == 0:
                continue

            self.props[out_i, :] = [
                c1 / total_alleles,                                   # HOM1
                c2 / total_alleles,                                   # HET
                c3 / total_alleles,                                   # HOM2
                c0 / total_alleles,                                   # U
                (1.0 - kept_sites / total_sites) if total_sites else 0
            ]

        self.current_DI = float(DIval)

    # --------------------------------------------------
    # Drawing
    # --------------------------------------------------

    def _draw_bars(self):
        self.ax.clear()

        x = np.arange(len(self.chrom_labels))
        bottoms = np.zeros(len(x))

        colours = [
            diemColours[1],  # HOM1
            diemColours[2],  # HET
            diemColours[3],  # HOM2
            "lightgray",     # U
            "white",         # excluded
        ]

        labels = ["HOM1", "HET", "HOM2", "U", "<DI"]

        for i in range(5):
            self.ax.bar(
                x,
                self.props[:, i],
                bottom=bottoms,
                color=colours[i],
                edgecolor="black" if i == 4 else None,
                linewidth=0.4 if i == 4 else 0,
                label=labels[i],
            )
            bottoms += self.props[:, i]

        self.ax.set_xlim(-0.5, len(x) - 0.5)
        self.ax.set_ylim(0, 1)

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(
            self.chrom_labels,
            rotation=90,
            fontsize=self.fontsize,
            ha="center",
        )

        self.ax.set_ylabel("Proportion of SNVs")

        prop = self.DInumer / self.DIdenom if self.DIdenom > 0 else 0.0

        self.ax.set_title(
            "Genomic contributions plot  DI ≥ {:.2f}  {} SNVs  ({:.1f}% divergent across barrier)"
            .format(self.current_DI, self.DInumer, 100 * prop),
            fontsize=12,
            pad=12
        )

        self.ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=8,
            frameon=False,
        )

        self.fig.canvas.draw_idle()

    # --------------------------------------------------
    # Widgets
    # --------------------------------------------------

    def _init_widgets(self):
        DI_span = get_DI_span(self.dPol)

        ax_DI = self.fig.add_axes([0.15, 0.20, 0.70, 0.03])
        self.DI_slider = Slider(
            ax_DI,
            "DI",
            DI_span[0],
            DI_span[1],
            valinit=DI_span[0],
        )
        self.DI_slider.on_changed(self.DIupdate)

        ax_FS = self.fig.add_axes([0.15, 0.13, 0.15, 0.03])
        self.font_slider = Slider(
            ax_FS,
            "Label font size",
            4,
            16,
            valinit=self.fontsize,
            valstep=1,
        )
        self.font_slider.on_changed(self.FONTupdate)

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------

    def DIupdate(self, val):
        self._compute_contributions(val)
        self._draw_bars()

    def FONTupdate(self, val):
        self.fontsize = int(val)
        self._draw_bars()



"""________________________________________ END GenomicContributions__________________"""



"""________________________________________ START IndGenomicContributions ___________________"""


class IndGenomicContributionsPlot:
    """
    Stacked-bar genomic contributions per INDIVIDUAL (HOM1, HET, HOM2, U, excluded),
    with DI filtering and widgets.

    - bars correspond to individuals (like GenomeSummaryPlot focus)
    - visuals correspond to GenomicContributionsPlot
    - reorder-by-HI button (HI computed at current DI)
    - optional incremental cache prefill via StatewiseDIIncrementalCache

    Uses: statewise_genomes_summary_given_DI

    c.f. 
    Figure 2, Figure 4:
    Petružela, J., Nürnberger, B., Ribas, A., Koutsovoulos, G., Čížková, 
    D., Fornůsková, A., Aghová, T., Blaxter, M., de Bellocq, J.G. and Baird, S.J.E. 
    (2025), Comparative Genomic Analysis of Co-Occurring Hybrid Zones 
    of House Mouse Parasites Pneumocystis murina and Syphacia obvelata 
    Using Genome Polarisation. Mol Ecol, 34: e70044. https://doi.org/10.1111/mec.70044
    """

    def __init__(
        self,
        dPol,
        *,
        prefill_cache=False,      # NEW
        prefill_step=None,        # NEW
        cache_tol=None,           # NEW
        progress=None,          # NEW: "text" | "none"
    ):
        self.dPol = dPol

        # labels + ordering
        self.IndNickNames = [Ind_Nickname(name) for name in dPol.indNames]
        self.indNameFont = 6
        self.indHIorder = np.arange(len(dPol.indNames), dtype=int)

        # plotted state
        self.current_DI = float("-inf")
        self.DInumer = 0
        self.DIdenom = 0
        self.global_HI = None
        self.props = None  # (nInd, 5)

        # ---- cache (per instance) ----
        # maps DI(float) -> (chrom_counts, chrom_retained)
        self._cache = {}
        self._cache_keys_sorted = []
        self._cache_tol = 0.0

        def _cache_set(di, payload):
            di = float(di)
            if di not in self._cache:
                bisect.insort(self._cache_keys_sorted, di)
            self._cache[di] = payload

        def _cache_get_nearest(di, tol):
            if not self._cache_keys_sorted:
                return None
            di = float(di)
            keys = self._cache_keys_sorted
            j = bisect.bisect_left(keys, di)

            candidates = []
            if 0 <= j < len(keys):
                candidates.append(keys[j])
            if 0 <= j - 1 < len(keys):
                candidates.append(keys[j - 1])

            best = None
            best_dist = None
            for k in candidates:
                dist = abs(k - di)
                if best_dist is None or dist < best_dist:
                    best = k
                    best_dist = dist

            if best is None or best_dist is None or best_dist > tol:
                return None
            return self._cache[best]

        self._cache_set = _cache_set
        self._cache_get_nearest = _cache_get_nearest

        # ---- figure & axes ----
        self.fig, self.ax = plt.subplots(figsize=(11, 4.8))
        self.fig.subplots_adjust(bottom=0.35, right=0.88)
        self.ax.format_coord = None  # no hover

        # ---- initial compute ----
        self._compute_props(float("-inf"))

        # ---- INITIAL ORDER BY HI (NEW) ----
        if self.global_HI is not None:
            self.indHIorder = np.argsort(self.global_HI)

        # ---- draw ----
        self._draw_bars()

        # ---- widgets ----
        self._init_widgets()

        # ---- OPTIONAL: incremental cache prefill ----
        if prefill_cache:
            DI_span = get_DI_span(self.dPol)
            di_min, di_max = float(DI_span[0]), float(DI_span[1])

            if prefill_step is None:
                span = di_max - di_min
                prefill_step = span / 200.0 if span > 0 else 1.0

            if cache_tol is None:
                cache_tol = float(prefill_step) / 2.0

            # Build DI grid including endpoints
            if di_max > di_min:
                n_steps = int(np.floor((di_max - di_min) / prefill_step))
                di_values = [di_min + k * prefill_step for k in range(n_steps + 1)]
                if not di_values or di_values[-1] < di_max:
                    di_values.append(di_max)
            else:
                di_values = [di_min]

            di_grid = np.asarray(di_values, dtype=float)

            inc = StatewiseDIIncrementalCache(
                self.dPol,
                di_grid=di_grid,
                progress=("text" if progress == "text" else None),
                label="IndGenomicContrib cache prefill",
            ).prefill()

            # Store snapshots
            for di_key, payload in inc._snapshots.items():
                self._cache_set(di_key, payload)

            self._cache_tol = float(cache_tol)

        self.fig.canvas.draw()   # force first render NOW (fix for issue 1)
        plt.show()

    # --------------------------------------------------
    # Core computation
    # --------------------------------------------------

    def _get_statewise_payload(self, DIval):
        """
        Return (chrom_counts, chrom_retained) using cache if available.
        """
        payload = None

        used_cache = False   # <-- ALWAYS define this

        if self._cache_tol > 0:
            payload = self._cache_get_nearest(DIval, self._cache_tol)

        if payload is None:
            chrom_counts, chrom_retained = statewise_genomes_summary_given_DI(self.dPol, DIval)
            payload = (chrom_counts, chrom_retained)
            # opportunistic cache (safe even without prefill)
            self._cache_set(DIval, payload)

        return payload

    def _compute_props(self, DIval):
        """
        Compute per-individual stacked proportions:
          HOM1, HET, HOM2, U, excluded

        Denominator is TOTAL alleles (including excluded DI-filtered-out sites),
        so excluded portion is meaningful per individual even under variable ploidy.
        """
        chrom_counts, chrom_retained = self._get_statewise_payload(DIval)



        nChr = len(chrom_counts)
        nInd = chrom_counts[0]["counts0"].shape[0]


        # totals of sites retained/total (SNVs, not alleles) for title
        self.DInumer = sum(n for (n, _) in chrom_retained)
        self.DIdenom = sum(d for (_, d) in chrom_retained)

        self.current_DI = float(DIval)

        # Sum ploidy-weighted counts over chromosomes for each individual
        sum0 = np.zeros(nInd, dtype=float)
        sum1 = np.zeros(nInd, dtype=float)
        sum2 = np.zeros(nInd, dtype=float)
        sum3 = np.zeros(nInd, dtype=float)

        # Total alleles per individual across ALL sites (retained + excluded)
        total_alleles_all = np.zeros(nInd, dtype=float)

        for chr_i in range(nChr):
            c = chrom_counts[chr_i]

            sum0 += c["counts0"]
            sum1 += c["counts1"]
            sum2 += c["counts2"]
            sum3 += c["counts3"]

            # total alleles for this chromosome for each individual
            total_sites_chr = chrom_retained[chr_i][1]  # denom
            w = np.asarray(self.dPol.chrPloidies[chr_i], dtype=float)  # (nInd,)
            total_alleles_all += float(total_sites_chr) * w

        retained_alleles = sum0 + sum1 + sum2 + sum3

        # Avoid division by 0 (rare but possible)
        denom = np.where(total_alleles_all > 0, total_alleles_all, 1.0)

        props = np.zeros((nInd, 5), dtype=float)
        props[:, 0] = sum1 / denom  # HOM1
        props[:, 1] = sum2 / denom  # HET
        props[:, 2] = sum3 / denom  # HOM2
        props[:, 3] = sum0 / denom  # U
        props[:, 4] = 1.0 - (retained_alleles / denom)  # excluded (DI-filtered-out)

        # Clamp minor floating error
        props = np.clip(props, 0.0, 1.0)

        self.props = props

        # Compute global HI for reorder-by-HI (authoritative from statewise counts)
        global_summaries = summaries_from_statewise_counts(chrom_counts)
        self.global_HI = global_summaries[0]  # HI is first summary


    # --------------------------------------------------
    # Drawing
    # --------------------------------------------------

    def _draw_bars(self):
 
        self.ax.clear()

        nInd = len(self.dPol.indNames)
        order = self.indHIorder

        x = np.arange(nInd)
        bottoms = np.zeros(nInd, dtype=float)


        colours = [
            diemColours[1],   # HOM1
            diemColours[2],   # HET
            diemColours[3],   # HOM2
            "lightgray",      # U
            "white",          # excluded
        ]
        labels = ["HOM1", "HET", "HOM2", "U", "<DI"]

        for k in range(5):
            y = np.asarray(self.props[order, k], dtype=float)
            self.ax.bar(
                x,
                y,
                bottom=bottoms.copy(),   # freeze for this layer
                color=colours[k],
                edgecolor="black" if k == 4 else None,
                linewidth=0.4 if k == 4 else 0,
                label=labels[k],
            )
            bottoms = bottoms + y        # NOT in-place

        self.ax.set_xlim(-0.5, nInd - 0.5)
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Proportion of genotypes")

        # xticks
        labels = np.array(self.IndNickNames)[order]
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(
            labels,
            rotation=90,
            fontsize=self.indNameFont,
            ha="center",
        )

        # title
        prop_sites = self.DInumer / self.DIdenom if self.DIdenom > 0 else 0.0
        self.ax.set_title(
            "Individual genomic contributions  DI ≥ {:.2f}  {} SNVs  ({:.1f}% divergent across barrier)"
            .format(self.current_DI, self.DInumer, 100 * prop_sites),
            fontsize=12,
            pad=12
        )

        # legend outside
        self.ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=8,
            frameon=False,
        )

        self.fig.canvas.draw_idle()
    # --------------------------------------------------
    # Widgets
    # --------------------------------------------------

    def _init_widgets(self):
        DI_span = get_DI_span(self.dPol)

        # DI slider
        ax_DI = self.fig.add_axes([0.18, 0.18, 0.64, 0.03])
        self.DI_slider = Slider(
            ax_DI,
            "DI",
            DI_span[0],
            DI_span[1],
            valinit=DI_span[0],
        )
        self.DI_slider.on_changed(self.DIupdate)

        # font size slider
        ax_FS = self.fig.add_axes([0.18, 0.11, 0.14, 0.03])
        self.font_slider = Slider(
            ax_FS,
            "Label font",
            4,
            16,
            valinit=self.indNameFont,
            valstep=1,
        )
        self.font_slider.on_changed(self.FONTupdate)

        # reorder button
        ax_RE = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])#[0.70, 0.105, 0.15, 0.045]) # EURG
        self.reorder_button = Button(
            ax_RE,
            "Reorder by HI",
            hovercolor="0.95",
            color="red",
        )
        self.reorder_button.on_clicked(self.reorder)

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------

    def DIupdate(self, val):

        # recompute props + HI at this DI
        self._compute_props(val)
        

        # keep current ordering unless user presses reorder
        self._draw_bars()

    def FONTupdate(self, val):
        self.indNameFont = int(val)
        self._draw_bars()


    def reorder(self, event=None):
        """
        Reorder individuals by HI given the CURRENT DI threshold.
        """        
        if self.global_HI is None:
            print("  global_HI is None -> returning")
            return

        self.indHIorder = np.argsort(self.global_HI)

        print("  order AFTER reorder head:", self.indHIorder[:12])
        print("  HI head (new order):", np.round(self.global_HI[self.indHIorder][:10], 4))

        self._draw_bars()


        
"""________________________________________ END IndGenomicContributions ___________________"""




"""________________________________________ START diemPairsPlot ___________________"""

def pwmatrixFromDiemType(aDT, DIthreshold=float("-inf")):
    """
    Compute a DI-filtered pairwise distance matrix from a DiemType object.

    DMBCtoucher

    Args:
        aDT : DiemType
        DIthreshold : float
            Only sites with DI >= DIthreshold are retained.

    Returns:
        M : (N, N) numpy array
            Symmetric pairwise distance matrix.
    """

    # -------------------------------------------------
    # Dimensions
    # -------------------------------------------------
    n_ind = aDT.DMBC[0].shape[0]

    # -------------------------------------------------
    # Pairwise weight matrix (codes 0..3)
    # -------------------------------------------------
    W = np.zeros((4, 4), dtype=float)

    W[1, 2] = W[2, 1] = 1
    W[1, 3] = W[3, 1] = 2
    W[2, 2] = 1
    W[2, 3] = W[3, 2] = 1
    # (1,1) and (3,3) remain 0

    # -------------------------------------------------
    # Accumulators
    # -------------------------------------------------
    num = np.zeros((n_ind, n_ind), dtype=float)
    den = np.zeros((n_ind, n_ind), dtype=float)

    # -------------------------------------------------
    # Single pass over chromosomes
    # -------------------------------------------------
    for chr_idx, SM in enumerate(aDT.DMBC):
        # SM shape: (n_ind, n_sites)
        DIvals = aDT.DIByChr[chr_idx]
        keep = DIvals >= DIthreshold

        if not np.any(keep):
            continue

        SMf = SM[:, keep]

        # iterate retained sites
        for s in range(SMf.shape[1]):
            col = SMf[:, s]

            valid = col != 0
            idx = np.where(valid)[0]

            if idx.size < 2:
                continue

            vals = col[idx]

            # pairwise contribution
            for ii, i in enumerate(idx):
                ai = vals[ii]
                for jj in range(ii + 1, len(idx)):
                    j = idx[jj]
                    aj = vals[jj]
                    w = W[ai, aj]

                    num[i, j] += w
                    num[j, i] += w
                    den[i, j] += 1
                    den[j, i] += 1

    # -------------------------------------------------
    # Final matrix
    # -------------------------------------------------
    M = np.full((n_ind, n_ind), np.nan)
    mask = den > 0
    M[mask] = num[mask] / den[mask]
    #np.fill_diagonal(M, 0.0) an example of ChatGPT 5.2 suggestion that is not biologically valid

    return M



# for parallel computation of pairwise distance matrix rows
def _pwmatrix_row(i, G, W):
    """
    Compute one row of the pairwise distance matrix.
    """
    n = G.shape[0]
    row = np.zeros(n, dtype=float)

    ai = G[i]
    for j in range(n):
        aj = G[j]
        valid = (ai != 0) & (aj != 0)
        denom = valid.sum()
        if denom == 0:
            row[j] = np.nan
        else:
            row[j] = W[ai[valid], aj[valid]].sum() / denom

    return i, row



#------------------version without char sidestep-------------------------------



def _pwmatrix_row_numeric(i, G, W):
    """
    Compute one row of the pairwise distance matrix from numeric codes.

    Distance definition (see also diem2fasta.py):
      - exclude sites where either is Unencodable (0)
      - per-site contributions via W
      - normalize by 2 so HET×HET -> 0.5 and HOM1×HOM2 -> 1.0
      - diagonal is within-individual: 0.5 per HET site, 0 otherwise, excluding Unencodable
    """
    n = G.shape[0]
    row = np.full(n, np.nan, dtype=float)  # default NaN if no valid sites

    ai = G[i]
    ai_ok = (ai != 0)

    for j in range(n):
        aj = G[j]

        if i == j:
            # within-individual: exclude unencodable sites
            valid = ai_ok
        else:
            # pairwise: exclude sites where either is unencodable
            valid = ai_ok & (aj != 0)

        denom = int(valid.sum())
        if denom:
            # divide by 2 to implement "average over phases" (partial credit)
            row[j] = W[ai[valid], aj[valid]].sum() / (2.0 * denom)

    return i, row


def PARApwmatrixFromDiemType(
    aDT,
    DIthreshold=float("-inf"),
    chrom_indices=None,
    n_jobs=-1,
    backend="loky",
):
    """
    Parallel computation of pairwise distance matrix.

    Args
    ----
    aDT : DiemType
    DIthreshold : float
    n_jobs : int
        Number of cores (-1 = all)
    backend : str
        joblib backend ("loky" recommended)

    Returns
    -------
    M : (n_ind, n_ind) numpy array
    """

    # -------------------------------------------------
    # DI-filtered genomes, optionally chromosome-restricted
    # -------------------------------------------------
    chunks = []
    n_ind = aDT.DMBC[0].shape[0]

    # Default: all chromosomes
    if chrom_indices is None:
        chrom_indices = range(len(aDT.DMBC))

    for chr_idx in chrom_indices:
        SM = aDT.DMBC[chr_idx]
        DI = aDT.DIByChr[chr_idx]

        keep = DI >= DIthreshold
        if not np.any(keep):
            continue


        # DMBC toucher: any int8 greater than 3 goes lto Derek state 0: U
        SMk = SM[:, keep]
        SMf = np.where((SMk >= 0) & (SMk <= 3), SMk, 0).astype(np.int8)

        chunks.append(SMf)

    if not chunks:
        return np.full((n_ind, n_ind), np.nan)

    # Concatenate retained chromosomes
    G = np.concatenate(chunks, axis=1)
    n = G.shape[0]

    # -------------------------------------------------
    # Distance weight matrix
    # -------------------------------------------------
    W = np.zeros((4, 4), dtype=float)
    W[1, 2] = W[2, 1] = 1
    W[1, 3] = W[3, 1] = 2
    W[2, 2] = 1
    W[2, 3] = W[3, 2] = 1
    # (0,* and *,0 excluded by valid mask)

    # -------------------------------------------------
    # Parallel row computation
    # -------------------------------------------------
    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        prefer="processes",
        batch_size=1,
    )(
        delayed(_pwmatrix_row_numeric)(i, G, W)
        for i in range(n)
    )

    # -------------------------------------------------
    # Assemble matrix
    # -------------------------------------------------
    M = np.zeros((n, n), dtype=float)
    for i, row in results:
        M[i, :] = row

    return M
#-------------------------------------------------

class diemPairsPlot:
    """
    Pairwise distance plot using BRICK rectangles (no imshow), now with:
      - DI slider
      - Reorder by HI button
      - optional incremental cache prefill (pairwise + optional HI/statewise)
    """
    """
    Pairwise distance plot using brickDiagram semantics.

    Uses genomes_summary_given_DI and PARApwmatrixFromDiemType

    c.f. Figure 2, Figure 4:
    Petružela, J., Nürnberger, B., Ribas, A., Koutsovoulos, G., Čížková, 
    D., Fornůsková, A., Aghová, T., Blaxter, M., de Bellocq, J.G. and Baird, S.J.E. 
    (2025), Comparative Genomic Analysis of Co-Occurring Hybrid Zones 
    of House Mouse Parasites Pneumocystis murina and Syphacia obvelata 
    Using Genome Polarisation. Mol Ecol, 34: e70044. https://doi.org/10.1111/mec.70044

    Coding co-pilot: ChatGPT 5.2

    Left panel:
        Square heatmap of pairwise distances (brick rectangles),
        ordered by Hybrid Index at the specified DI threshold.

    Right panel:
        Vertical colour key.

    Hover:
        Shows "IndA × IndB : distance".

    New features:
      - DI slider
      - Reorder by HI button
      - incremental cache prefill (pairwise + optional HI/statewise)
    """

    def __init__(
        self,
        dPol,
        DIthreshold=float("-inf"),
        figsize=(9, 6),
        chrom_indices=None,

        # caching options
        prefill_cache=True,
        prefill_step=None,
        cache_tol=None,
        progress=None,                 # None | "text"
        cache_statewise_for_HI=True,   # True recommended if you already have StatewiseDIIncrementalCache
    ):
        self.dPol = dPol
        self.chrom_indices = chrom_indices
        self.current_DI = float(DIthreshold)

        # ---------- DI span / grid ----------
        DI_span = get_DI_span(self.dPol)
        self.di_min, self.di_max = float(DI_span[0]), float(DI_span[1])

        if prefill_step is None:
            span = self.di_max - self.di_min
            prefill_step = span / 200.0 if span > 0 else 1.0

        if cache_tol is None:
            cache_tol = float(prefill_step) / 2.0
        self._cache_tol = float(cache_tol)

        self.di_grid = self._build_di_grid(self.di_min, self.di_max, float(prefill_step))

        # ---------- caches ----------
        self._inc_pairwise = None
        self._inc_statewise = None

        if prefill_cache:
            # Pairwise incremental cache
            self._inc_pairwise = PairwiseDIIncrementalCache(
                self.dPol,
                di_grid=self.di_grid,
                chrom_indices=self.chrom_indices,
                progress=("text" if progress == "text" else None),
                label="PairsPlot matrix cache prefill",
                snapshot_dtype=np.float32,
            ).prefill()

            # Optional HI ordering cache (fast ordering under slider motion)
            if cache_statewise_for_HI:
                self._inc_statewise = StatewiseDIIncrementalCache(
                    self.dPol,
                    di_grid=self.di_grid,
                    progress=("text" if progress == "text" else None),
                    label="PairsPlot HI cache prefill",
                ).prefill()

        # ---------- figure layout ----------
        self.fig = plt.figure(figsize=figsize)
        gs = self.fig.add_gridspec(
            nrows=1, ncols=2,
            width_ratios=[20, 1],
            wspace=0.08
        )
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.cax = self.fig.add_subplot(gs[0, 1])

        # room at bottom for widgets
        self.fig.subplots_adjust(bottom=0.22)

        # ---------- colormap ----------
        self.cmap = LinearSegmentedColormap.from_list(
            "soft_coolwarm",
            ["#1e90ff", "white", "#fff266", "#ff1a1a"]
        )

        # ---------- initial compute ----------
        self._compute_for_DI(self.current_DI, force_reorder=True)

        # ---------- draw once ----------
        self._setup_axes()
        self._init_bricks()          # main matrix bricks (persistent)
        self._init_colour_key()      # colour key bricks (persistent)

        # ---------- widgets ----------
        self._init_widgets()

        # ---------- hover ----------
        self._install_format_coord()

        plt.show()

    # =================================================
    # DI grid helpers
    # =================================================

    @staticmethod
    def _build_di_grid(di_min, di_max, step):
        if di_max > di_min:
            n_steps = int(np.floor((di_max - di_min) / step))
            vals = [di_min + k * step for k in range(n_steps + 1)]
            if not vals or vals[-1] < di_max:
                vals.append(di_max)
        else:
            vals = [di_min]
        return np.asarray(vals, dtype=float)

    def _nearest_grid_value(self, di):
        di = float(di)
        grid = self.di_grid
        j = int(np.argmin(np.abs(grid - di)))
        return float(grid[j])

    # =================================================
    # Computation
    # =================================================
    
    def _get_HI_and_retained_for_DI(self, DIthreshold):
        """
        Return (HI, DInumer, DIdenom) at DIthreshold.
        DInumer/DIdenom are SNV counts retained/total across all chromosomes.
        """
        if self._inc_statewise is not None:
            di_key = self._nearest_grid_value(DIthreshold)
            chrom_counts, chrom_retained = self._inc_statewise._snapshots[di_key]
            HI = summaries_from_statewise_counts(chrom_counts)[0]
            DInumer = sum(n for (n, _) in chrom_retained)
            DIdenom = sum(d for (_, d) in chrom_retained)
            return HI, DInumer, DIdenom

        # fallback (no statewise cache)
        # Use the same statewise function IndGenomicContributionsPlot uses
        chrom_counts, chrom_retained = statewise_genomes_summary_given_DI(self.dPol, float(DIthreshold))
        HI = summaries_from_statewise_counts(chrom_counts)[0]
        DInumer = sum(n for (n, _) in chrom_retained)
        DIdenom = sum(d for (_, d) in chrom_retained)
        return HI, DInumer, DIdenom

    def _get_M_for_DI(self, DIthreshold):
        if self._inc_pairwise is not None:
            return self._inc_pairwise.get(DIthreshold)

        # Fallback: the existing parallel matrix builder
        return PARApwmatrixFromDiemType(
            self.dPol,
            DIthreshold=float(DIthreshold),
            chrom_indices=self.chrom_indices,
        )

    def _compute_for_DI(self, DIthreshold, *, force_reorder=False):
        """
        Compute matrix + (optionally) compute HI ordering.
        If force_reorder=False, keep existing order and only update M accordingly.
        """
        self.current_DI = float(DIthreshold)

        # matrix (unordered)
        M_raw = self._get_M_for_DI(self.current_DI)

        # ordering
        HI, DInumer, DIdenom = self._get_HI_and_retained_for_DI(self.current_DI)
        self.DInumer = int(DInumer)
        self.DIdenom = int(DIdenom)
        if force_reorder or not hasattr(self, "ind_order") or self.ind_order is None:
            self.ind_order = np.argsort(HI)

        # apply ordering
        self.indNames = np.array(self.dPol.indNames)[self.ind_order]
        self.n = len(self.indNames)
        self.M = M_raw[self.ind_order][:, self.ind_order]

        # range
        self.vmin = float(np.nanmin(self.M)) if np.any(np.isfinite(self.M)) else 0.0
        self.vmax = float(np.nanmax(self.M)) if np.any(np.isfinite(self.M)) else 1.0

    # =================================================
    # Drawing setup
    # =================================================

    def _setup_axes(self):
        self.ax.set_xlim(0, self.n)
        self.ax.set_ylim(0, self.n)
        self.ax.set_aspect("equal")

        centers = np.arange(self.n) + 0.5
        self.ax.set_xticks(centers)
        self.ax.set_yticks(centers)

        # initial label fontsize (linked to slider)
        self.label_fontsize = 4
        self.ax.set_xticklabels(self.indNames, rotation=90, fontsize=self.label_fontsize)
        self.ax.set_yticklabels(self.indNames, fontsize=self.label_fontsize)

        prop_sites = self.DInumer / self.DIdenom if self.DIdenom > 0 else 0.0
        self.ax.set_title(
            "Pairwise distances  DI ≥ {:.2f}  {} SNVs  ({:.1f}% divergent across barrier)"
            .format(self.current_DI, self.DInumer, 100 * prop_sites),
            pad=10,
        )

    def _color_for_val(self, val, norm):
        if not np.isfinite(val):
            return "black"
        return self.cmap(norm(val))

    # =================================================
    # Main bricks: create once, update facecolors later
    # =================================================

    def _init_bricks(self):
        # Remove only previously created bricks (do NOT use self.ax.patches.clear())
        if hasattr(self, "_bricks"):
            for r in self._bricks:
                try:
                    r.remove()
                except Exception:
                    pass

        norm = plt.Normalize(self.vmin, self.vmax)
        self._norm = norm

        self._bricks = []

        # IMPORTANT: preserve your original orientation: val = M[j, i] drawn at (i, j)
        for i in range(self.n):
            for j in range(self.n):
                val = self.M[j, i]
                rect = Rectangle(
                    (i, j), 1, 1,
                    facecolor=self._color_for_val(val, norm),
                    edgecolor="none"
                )
                self.ax.add_patch(rect)
                self._bricks.append(rect)

        self.fig.canvas.draw_idle()

    def _update_bricks(self):
        # update normalisation if needed
        norm = plt.Normalize(self.vmin, self.vmax)
        self._norm = norm

        # update all facecolors
        k = 0
        for i in range(self.n):
            for j in range(self.n):
                val = self.M[j, i]
                self._bricks[k].set_facecolor(self._color_for_val(val, norm))
                k += 1

        # update title
        #self.ax.set_title(f"Pairwise distances (DI ≥ {self.current_DI:.2f})", pad=10)
        prop_sites = self.DInumer / self.DIdenom if self.DIdenom > 0 else 0.0
        self.ax.set_title(
            "Pairwise distances  DI ≥ {:.2f}  {} SNVs  ({:.1f}% divergent across barrier)"
            .format(self.current_DI, self.DInumer, 100 * prop_sites),
            pad=10,
        )

        self.fig.canvas.draw_idle()

    # =================================================
    # Colour key: bricks, not imshow
    # =================================================

    def _init_colour_key(self):
        self.cax.clear()

        # key as a vertical stack of small bricks
        self._key_bins = 256
        self._key_rects = []

        # axis coordinates:
        # x in [0,1], y in [0, key_bins]
        self.cax.set_xlim(0, 1)
        self.cax.set_ylim(0, self._key_bins)

        norm = plt.Normalize(self.vmin, self.vmax)
        for y in range(self._key_bins):
            # map y to value in [vmin, vmax]
            frac = y / (self._key_bins - 1)
            val = self.vmin + frac * (self.vmax - self.vmin)
            rect = Rectangle(
                (0, y), 1, 1,
                facecolor=self.cmap(norm(val)),
                edgecolor="none"
            )
            self.cax.add_patch(rect)
            self._key_rects.append(rect)

        # ticks + labels
        self.cax.set_xticks([])
        self.cax.set_yticks([0, self._key_bins - 1])
        self.cax.set_yticklabels([f"{self.vmin:.2f}", f"{self.vmax:.2f}"], fontsize=8)
        self.cax.set_title("Distance", fontsize=9)

        self.fig.canvas.draw_idle()

    def _update_colour_key(self):
        # recolour bricks according to updated vmin/vmax
        norm = plt.Normalize(self.vmin, self.vmax)
        for y in range(self._key_bins):
            frac = y / (self._key_bins - 1)
            val = self.vmin + frac * (self.vmax - self.vmin)
            self._key_rects[y].set_facecolor(self.cmap(norm(val)))

        self.cax.set_yticklabels([f"{self.vmin:.2f}", f"{self.vmax:.2f}"], fontsize=8)
        self.fig.canvas.draw_idle()

    # =================================================
    # Widgets
    # =================================================

    def _init_widgets(self):
        # DI slider
        ax_DI = self.fig.add_axes([0.18, 0.10, 0.52, 0.03])
        self.DI_slider = Slider(ax_DI, "DI", self.di_min, self.di_max, valinit=self.di_min)
        self.DI_slider.on_changed(self._on_DI_change)

        # reorder button
        ax_RE = self.fig.add_axes([0.82, 0.10, 0.14, 0.035])
        self.reorder_button = Button(ax_RE, "Reorder by HI", hovercolor="0.95", color="red")
        self.reorder_button.on_clicked(self._on_reorder)

        # font slider: beneath colour key (your original style)
        pos = self.cax.get_position()
        ax_FS = self.fig.add_axes([pos.x0, 0.04, pos.width, 0.03])

        self.font_slider = Slider(ax_FS, "Labels", 0, 8, valinit=self.label_fontsize, valstep=1)
        self.font_slider.on_changed(self._on_fontsize_change)

    # =================================================
    # Callbacks
    # =================================================

    def _on_DI_change(self, val):
        # If cached, snap to nearest grid DI
        if self._inc_pairwise is not None or self._inc_statewise is not None:
            di_eff = self._nearest_grid_value(val)
        else:
            di_eff = float(val)

        # IMPORTANT: keep current ordering during slider motion
        self._compute_for_DI(di_eff, force_reorder=False)

        # update bricks + key
        self._update_bricks()
        self._update_colour_key()

    def _on_reorder(self, event=None):
        # recompute ordering at current DI, then redraw labels + bricks
        self._compute_for_DI(self.current_DI, force_reorder=True)

        # update axis tick labels (same ticks, new labels)
        centers = np.arange(self.n) + 0.5
        self.ax.set_xticks(centers)
        self.ax.set_yticks(centers)
        self.ax.set_xticklabels(self.indNames, rotation=90, fontsize=self.label_fontsize)
        self.ax.set_yticklabels(self.indNames, fontsize=self.label_fontsize)

        self._update_bricks()
        self._update_colour_key()

    def _on_fontsize_change(self, val):
        fs = int(val)
        self.label_fontsize = fs
        self.ax.set_xticklabels(self.indNames, rotation=90, fontsize=fs)
        self.ax.set_yticklabels(self.indNames, fontsize=fs)
        self.fig.canvas.draw_idle()

    # =================================================
    # Hover
    # =================================================

    def _install_format_coord(self):
        n = self.n

        def format_coord(x, y):
            fallback = " " * 40
            i = int(np.floor(x))
            j = int(np.floor(y))
            if 0 <= i < n and 0 <= j < n:
                a = self.indNames[j]
                b = self.indNames[i]
                d = self.M[j, i]
                if np.isfinite(d):
                    return f"{a} × {b} : {d:.3f}"
                return f"{a} × {b} : NA"
            return fallback

        self.ax.format_coord = format_coord

"""________________________________________ END diemPairsPlot ___________________"""



"""________________________________________ START DiemMultiPairsPlot__________________"""




class diemMultiPairsPlot:
    """
    Multi-chromosome version of diemPairsPlot.

    One brick heatmap per chromosome, ordered by global Hybrid Index,
    arranged in a grid. The top-right grid cell contains the shared
    colour key.

    Widgets:
      - DI slider (updates matrices, keeps current order)
      - Reorder by HI button (recomputes order at current DI)
      - Label font size slider (all subplots)

    Optional:
      - prefill caching (pairwise per chromosome + optional HI/statewise)
    """

    def __init__(
        self,
        dPol,
        chrom_indices,
        DIthreshold=float("-inf"),
        max_cols=3,
        figsize=(12, 8),

        # caching options
        prefill_cache=True,
        prefill_step=None,
        cache_tol=None,
        progress=None,                 # None | "text"
        cache_statewise_for_HI=True,   # True recommended if using cache

        # layout tweak
        row_hspace=0.60,
        col_wspace=0.35,
    ):
        self.dPol = dPol
        self.chrom_indices = [int(i) for i in chrom_indices]
        self.chrom_indices = self._validate_chrom_indices(chrom_indices)
        self.ChrNickNames = [Chr_Nickname(name) for name in dPol.chrNames]
        self.current_DI = float(DIthreshold)

        # ---------- DI span / grid ----------
        DI_span = get_DI_span(self.dPol)
        self.di_min, self.di_max = float(DI_span[0]), float(DI_span[1])

        if prefill_step is None:
            span = self.di_max - self.di_min
            prefill_step = span / 200.0 if span > 0 else 1.0

        if cache_tol is None:
            cache_tol = float(prefill_step) / 2.0
        self._cache_tol = float(cache_tol)

        self.di_grid = self._build_di_grid(self.di_min, self.di_max, float(prefill_step))

        # ---------- optional caches ----------
        self._inc_statewise = None
        self._inc_pairwise_by_chr = {}  # chr_idx -> PairwiseDIIncrementalCache

        if prefill_cache:
            for chr_idx in self.chrom_indices:
                self._inc_pairwise_by_chr[chr_idx] = PairwiseDIIncrementalCache(
                    self.dPol,
                    di_grid=self.di_grid,
                    chrom_indices=[chr_idx],
                    progress=("text" if progress == "text" else None),
                    label=f"MultiPairsPlot chr{chr_idx} matrix cache prefill",
                    snapshot_dtype=np.float32,
                ).prefill()

            if cache_statewise_for_HI:
                self._inc_statewise = StatewiseDIIncrementalCache(
                    self.dPol,
                    di_grid=self.di_grid,
                    progress=("text" if progress == "text" else None),
                    label="MultiPairsPlot HI cache prefill",
                ).prefill()

        # ---------- initial ordering by HI at current DI ----------
        self.ind_order = None
        self._compute_order_for_DI(self.current_DI)

        self.indNames = np.array(self.dPol.indNames)[self.ind_order]
        self.n = len(self.indNames)

        # ---------- initial matrices at current DI ----------
        self.Ms = []
        self._compute_matrices_for_DI(self.current_DI, keep_order=True)

        # ---------- figure / grid layout ----------
        n_plots = len(self.chrom_indices)
        n_cols = min(max_cols, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))

        self.fig = plt.figure(figsize=figsize)
        gs = self.fig.add_gridspec(
            n_rows,
            n_cols + 1,
            width_ratios=[20] * n_cols + [1.6],
            hspace=float(row_hspace),
            wspace=float(col_wspace),
        )

        self.axes = []
        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                if idx < n_plots:
                    self.axes.append(self.fig.add_subplot(gs[r, c]))

        self.cax = self.fig.add_subplot(gs[0, -1])  # colour key

        # room at bottom for widgets
        self.fig.subplots_adjust(bottom=0.20)

        # ---------- colormap ----------
        self.cmap = LinearSegmentedColormap.from_list(
            "soft_coolwarm",
            ["#1e90ff", "white", "#fff266", "#ff1a1a"],
        )

        # ---------- style state ----------
        self.fontsize = 4

        # ---------- draw once (persistent patches) ----------
        self._setup_axes_all()
        self._init_bricks_all()
        self._init_colour_key()

        # ---------- widgets ----------
        self._init_widgets()

        # ---------- hover ----------
        self._install_format_coord()

        plt.show()

    # =================================================
    # DI grid helpers
    # =================================================
    @staticmethod
    def _build_di_grid(di_min, di_max, step):
        if di_max > di_min:
            n_steps = int(np.floor((di_max - di_min) / step))
            vals = di_min + step * np.arange(n_steps + 1, dtype=float)
            vals = np.clip(vals, di_min, di_max)
            vals[-1] = di_max
        else:
            vals = np.array([di_min], dtype=float)
        return np.asarray(vals, dtype=float)

    def _nearest_grid_value(self, di):
        di = float(di)
        j = int(np.argmin(np.abs(self.di_grid - di)))
        return float(self.di_grid[j])

    # =================================================
    # Computation
    # =================================================
    
    def _get_HI_and_retained_for_DI(self, DIthreshold):
        if self._inc_statewise is not None:
            di_key = self._nearest_grid_value(DIthreshold)
            chrom_counts, chrom_retained = self._inc_statewise._snapshots[di_key]
            HI = summaries_from_statewise_counts(chrom_counts)[0]
        else:
            chrom_counts, chrom_retained = statewise_genomes_summary_given_DI(self.dPol, float(DIthreshold))
            HI = summaries_from_statewise_counts(chrom_counts)[0]

        DInumer = sum(n for (n, _) in chrom_retained)
        DIdenom = sum(d for (_, d) in chrom_retained)
        return HI, int(DInumer), int(DIdenom)

    def _compute_order_for_DI(self, DIthreshold):
        HI, DInumer, DIdenom = self._get_HI_and_retained_for_DI(DIthreshold)
        self.ind_order = np.argsort(HI)
        self.DInumer = DInumer
        self.DIdenom = DIdenom

    def _get_M_chr_for_DI(self, chr_idx, DIthreshold):
        chr_idx = int(chr_idx)
        if chr_idx in self._inc_pairwise_by_chr:
            return self._inc_pairwise_by_chr[chr_idx].get(DIthreshold)

        return PARApwmatrixFromDiemType(
            self.dPol,
            DIthreshold=float(DIthreshold),
            chrom_indices=[chr_idx],
        )

    def _compute_matrices_for_DI(self, DIthreshold, *, keep_order=True):
        """
        Recompute per-chromosome matrices at DIthreshold.
        If keep_order=True, use existing self.ind_order.
        """
        self.current_DI = float(DIthreshold)

        if not keep_order or self.ind_order is None:
            self._compute_order_for_DI(self.current_DI)

        self.indNames = np.array(self.dPol.indNames)[self.ind_order]
        self.n = len(self.indNames)

        Ms = []
        vmins, vmaxs = [], []

        for chr_idx in self.chrom_indices:
            M = self._get_M_chr_for_DI(chr_idx, self.current_DI)
            M = M[self.ind_order][:, self.ind_order]
            Ms.append(M)

            finite = np.isfinite(M)
            if np.any(finite):
                vmins.append(np.nanmin(M))
                vmaxs.append(np.nanmax(M))

        self.Ms = Ms
        self.vmin = min(vmins) if vmins else 0.0
        self.vmax = max(vmaxs) if vmaxs else 1.0

    # =================================================
    # Drawing setup
    # =================================================
    def _setup_axes_all(self):
        centers = np.arange(self.n) + 0.5

        for ax, chr_idx in zip(self.axes, self.chrom_indices):
            ax.set_xlim(0, self.n)
            ax.set_ylim(0, self.n)
            ax.set_aspect("equal")

            ax.set_xticks(centers)
            ax.set_yticks(centers)

            ax.set_xticklabels(self.indNames, rotation=90, fontsize=self.fontsize)
            ax.set_yticklabels(self.indNames, fontsize=self.fontsize)

            ax.set_title(self.ChrNickNames[int(chr_idx)], fontsize=10, pad=8)

        prop_sites = self.DInumer / self.DIdenom if self.DIdenom > 0 else 0.0
        self.fig.suptitle(
            "Pairwise distances DI ≥ {:.2f}  {} SNVs  ({:.1f}% divergent across barrier)"
            .format(self.current_DI, self.DInumer, 100 * prop_sites),
            y=0.98
        )

    def _color_for_val(self, val, norm):
        if not np.isfinite(val):
            return "black"
        return self.cmap(norm(val))

    # =================================================
    # Bricks: create once, update facecolors later
    # =================================================
    def _init_bricks_all(self):
        self._bricks_by_ax = []  # list aligned with self.axes / self.Ms
        norm = plt.Normalize(self.vmin, self.vmax)
        self._norm = norm

        for ax, M in zip(self.axes, self.Ms):
            bricks = []
            for i in range(self.n):
                for j in range(self.n):
                    val = M[j, i]
                    rect = Rectangle(
                        (i, j), 1, 1,
                        facecolor=self._color_for_val(val, norm),
                        edgecolor="none",
                    )
                    ax.add_patch(rect)
                    bricks.append(rect)
            self._bricks_by_ax.append(bricks)

        self.fig.canvas.draw_idle()

    def _update_bricks_all(self):
        norm = plt.Normalize(self.vmin, self.vmax)
        self._norm = norm

        for bricks, M, ax in zip(self._bricks_by_ax, self.Ms, self.axes):
            k = 0
            for i in range(self.n):
                for j in range(self.n):
                    val = M[j, i]
                    bricks[k].set_facecolor(self._color_for_val(val, norm))
                    k += 1

        prop_sites = self.DInumer / self.DIdenom if self.DIdenom > 0 else 0.0
        self.fig.suptitle(
            "Pairwise distances DI ≥ {:.2f}  {} SNVs  ({:.1f}% divergent across barrier)"
            .format(self.current_DI, self.DInumer, 100 * prop_sites),
            y=0.98
        )
        self.fig.canvas.draw_idle()

    # =================================================
    # Colour key
    # =================================================
    def _init_colour_key(self):
        self.cax.clear()
        self._key_bins = 256
        gradient = np.linspace(self.vmin, self.vmax, self._key_bins).reshape(-1, 1)

        self._key_im = self.cax.imshow(
            gradient,
            aspect="auto",
            cmap=self.cmap,
            origin="lower",
        )

        self.cax.set_xticks([])
        self.cax.set_yticks([0, self._key_bins - 1])
        self.cax.set_yticklabels([f"{self.vmin:.2f}", f"{self.vmax:.2f}"], fontsize=8)
        self.cax.set_title("Distance", fontsize=9)

        self.fig.canvas.draw_idle()

    def _update_colour_key(self):
        gradient = np.linspace(self.vmin, self.vmax, self._key_bins).reshape(-1, 1)
        self._key_im.set_data(gradient)
        self._key_im.set_clim(self.vmin, self.vmax)

        self.cax.set_yticklabels([f"{self.vmin:.2f}", f"{self.vmax:.2f}"], fontsize=8)
        self.fig.canvas.draw_idle()

    # =================================================
    # Widgets
    # =================================================
    def _init_widgets(self):
        # DI slider
        ax_DI = self.fig.add_axes([0.15, 0.10, 0.60, 0.03])
        self.DI_slider = Slider(ax_DI, "DI", self.di_min, self.di_max, valinit=self.di_min)
        self.DI_slider.on_changed(self._on_DI_change)

        # Reorder button
        ax_RE = self.fig.add_axes([0.88, 0.10, 0.08, 0.035])
        self.reorder_button = Button(ax_RE, "Reorder by HI", hovercolor="0.95", color="red")
        self.reorder_button.on_clicked(self._on_reorder)

        # Font slider under colour key (your original location)
        pos = self.cax.get_position()
        ax_FS = self.fig.add_axes([pos.x0, 0.04, pos.width*2, 0.03])
        self.font_slider = Slider(ax_FS, "Labels", 0, 8, valinit=self.fontsize, valstep=1)
        self.font_slider.on_changed(self._on_font_change)

    # =================================================
    # Callbacks
    # =================================================
    def _on_DI_change(self, val):
        # If cached, snap to nearest grid DI
        if self._inc_pairwise_by_chr or (self._inc_statewise is not None):
            di_eff = self._nearest_grid_value(val)
        else:
            di_eff = float(val)
        # update retained/total counts for title (no reorder)
        _, self.DInumer, self.DIdenom = self._get_HI_and_retained_for_DI(di_eff)

        # keep current ordering during slider motion
        self._compute_matrices_for_DI(di_eff, keep_order=True)

        # update bricks + key
        self._update_bricks_all()
        self._update_colour_key()

    def _on_reorder(self, event=None):
        # recompute ordering at current DI, and recompute matrices with new order
        self._compute_matrices_for_DI(self.current_DI, keep_order=False)

        # update tick labels everywhere
        centers = np.arange(self.n) + 0.5
        for ax, chr_idx in zip(self.axes, self.chrom_indices):
            ax.set_xticks(centers)
            ax.set_yticks(centers)
            ax.set_xticklabels(self.indNames, rotation=90, fontsize=self.fontsize)
            ax.set_yticklabels(self.indNames, fontsize=self.fontsize)
            ax.set_title(self.ChrNickNames[int(chr_idx)], fontsize=10, pad=8)

        self._update_bricks_all()
        self._update_colour_key()

    def _on_font_change(self, val):
        self.fontsize = int(val)
        for ax in self.axes:
            ax.set_xticklabels(self.indNames, rotation=90, fontsize=self.fontsize)
            ax.set_yticklabels(self.indNames, fontsize=self.fontsize)
        self.fig.canvas.draw_idle()

    # ==================================================
    # Validation
    # ==================================================

    def _validate_chrom_indices(self, chrom_indices):
        max_idx = len(self.dPol.chrLengths) - 1
        valid, rejected = [], []

        for idx in chrom_indices:
            if isinstance(idx, (int, np.integer)) and 0 <= int(idx) <= max_idx:
                valid.append(int(idx))
            else:
                rejected.append(idx)

        if rejected:
            print("diemMultiPairsPlot: rejected chromosome indices:", rejected)

        if not valid:
            raise ValueError("diemMultiPairsPlot: no valid chromosome indices.")

        return valid
    

    # =================================================
    # Hover logic
    # =================================================
    def _install_format_coord(self):
        # Formatter that always reads current matrices (no stale closures)
        n = self.n
        ax_to_idx = {ax: k for k, ax in enumerate(self.axes)}

        def make_formatter(ax):
            def format_coord(x, y):
                fallback = " " * 40
                i = int(np.floor(x))
                j = int(np.floor(y))
                if 0 <= i < self.n and 0 <= j < self.n:
                    k = ax_to_idx.get(ax, None)
                    if k is None:
                        return fallback
                    M = self.Ms[k]
                    a = self.indNames[j]
                    b = self.indNames[i]
                    d = M[j, i]
                    if np.isfinite(d):
                        return f"{a} × {b} : {d:.3f}"
                    return f"{a} × {b} : NA"
                return fallback
            return format_coord

        for ax in self.axes:
            ax.format_coord = make_formatter(ax)


"""________________________________________ END DiemMultiPairsPlot __________________"""


"""________________________________________ START DiemPlotPrep__________________"""
    
    
class DiemPlotPrep:
    """ 
    Prepares data for DI-based plotting, including filtering, smoothing, dithering, and label generation.
    Args:
        plot_theme: Theme for plotting.
        ind_ids: List of individual IDs.
        chrRefLengths: Dictionary of chromosome reference lengths. Perhaps
        polarised_data: DataFrame containing polarised genomic data.
        di_threshold: DI threshold for filtering.
        di_column: Column name for DI values.
        diemStringPyCol: Column name for Diem genotype strings.
        genome_pixels: Number of genome pixels for dithering.
        ticks: Optional ticks for plotting.
        smooth: Optional smoothing parameter.
    """
    def __init__(self, plot_theme, ind_ids, chrRefLengths, polarised_data, di_threshold, di_column, diemStringPyCol, genome_pixels, ticks=None, chrRelativeRecRates=None, smooth=None):
        self.polarised_data = polarised_data
        self.di_threshold = di_threshold
        self.di_column = di_column
        self.diemStringPyCol = diemStringPyCol
        self.genome_pixels = genome_pixels
        self.plot_theme = plot_theme
        self.ind_ids = ind_ids
        self.chrRefLengths = chrRefLengths
        self.chrRelativeRecRates=chrRelativeRecRates
        self.ticks = ticks
        self.smooth = smooth

        self.diemPlotLabel = None
        self.DIfilteredDATA = None
        self.DIfilteredGenomes = None
        self.DIfilteredHIs = None
        self.DIfilteredBED = None
        self.DIpercent = None
        self.DIfilteredScafRLEs = None
        self.diemDITgenomes = None
        self.DIfilteredGenomes_unsmoothed = None
        self.DIfilteredBED_formatted = None
        self.IndIDs_ordered = None
        self.unit_plot_prep = []
        self.plot_ordered = None
        self.length_of_chromosomes = {}
        self.iris_plot_prep = {}
        self.diemDITgenomes_ordered = None
        self.nBasesDithered = None
        self.chrom_keys = None
        self.MapBC = None # primitive map positions of each SNV

        self.diem_plot_prep()

    def diem_plot_prep(self):
        """ Perform DI filtering, dithering, and label generation """
        self.filter_data()
        
        if self.smooth:
            self.initMapcoords() # before smoothing
            self.kernel_smooth(self.smooth)
        self.diem_dithering()

        self.generate_plot_label(self.plot_theme)
        self.format_bed_data() 


    def initMapcoords(self):
        """Initialize map coordinates based on filtered BED data"""
        if self.DIfilteredBED_formatted is None:
            self.MapBC = None
            return

        self.MapBC = []
        for i in range(len(self.chrom_keys)):
            bed_positions = self.DIfilteredBED_formatted[i]
            ref_len = self.chrRefLengths[i]

            # normalize to chromosome-relative coordinate [0, 1]
            self.MapBC.append(bed_positions / ref_len)


    # format_bed_data reworked by ChatGPT 5.2 for speed, but mostly clarity
    def format_bed_data(self):
        # -------------------------------------------------
        # 1. Compute ordering by HI (vectorised + clearer)
        # -------------------------------------------------
        HI_values = np.array([
            float(b[0]) if b[0] is not None else np.nan
            for b in self.DIfilteredHIs
        ])
    
        # stable sort: NaNs last
        sorted_indices = np.argsort(
            np.isnan(HI_values), kind="stable"
        )
        sorted_indices = sorted_indices[np.argsort(HI_values[sorted_indices], kind="stable")]
    
        self.plot_ordered = list(zip(HI_values[sorted_indices], sorted_indices + 1))
        self.IndIDs_ordered = [self.ind_ids[i] for i in sorted_indices]
        self.diemDITgenomes_ordered = [self.diemDITgenomes[i] for i in sorted_indices]
    
        # -------------------------------------------------
        # 2. Prepare unit_plot_prep (slice once, reuse)
        # -------------------------------------------------
        self.unit_plot_prep = []
        
        start = 0
        for bed_data in self.DIfilteredBED_formatted:
            end = start + len(bed_data)
        
            sublist = [genome[start:end] for genome in self.DIfilteredGenomes]
            self.unit_plot_prep.append([sublist[idx] for idx in sorted_indices])
        
            start = end
    
    
    def filter_data(self):
        """ Apply DI threshold filtering on the data """
        if isinstance(self.di_threshold, str):  # No filtering if threshold is a string
            self.DIfilteredDATA = self.polarised_data
        elif isinstance(self.di_threshold, int) or isinstance(self.di_threshold, float):  # Filter above if threshold is just one number
            self.DIfilteredDATA = self.polarised_data[self.polarised_data.DI >= self.di_threshold]
        else:  # Filter within an interval if threshold is a tuple or list
            self.DIfilteredDATA = self.polarised_data[(self.di_threshold[0] <= self.polarised_data.DI) & (self.polarised_data.DI <= self.di_threshold[1])]
    
        # Extract relevant data after filtering
        self.DIfilteredGenomes = StringTranspose(self.DIfilteredDATA['diem_genotype'])[1:] # slice off the 'S' column
        self.DIfilteredHIs = [pHetErrOnString(genome) for genome in self.DIfilteredGenomes]
        self.DIfilteredBED = self.DIfilteredDATA[['chrom','start']].values.tolist()
        self.DIpercent = round(100 * len(self.DIfilteredDATA) / len(self.polarised_data))
        self.DIfilteredScafRLEs = RichRLE(self.DIfilteredDATA['chrom'].values.tolist())

        # the following WAS the first operation of dithering section!
        # now here, so map of refpos can be set before smoothing...
        # -------------------------------------------------
        # 1. Group BED entries by chromosome (single pass)
        # -------------------------------------------------
        grouped = defaultdict(list)
        for key, value in self.DIfilteredBED:
            grouped[key].append(value)
    
        self.chrom_keys = list(grouped.keys())
        self.DIfilteredBED_formatted = [
            np.asarray(grouped[k]) for k in self.chrom_keys
        ]       


    def kernel_smooth(self, scale): # ChatGPT drop-in
    
        # --------------------------------------------------
        # 1. Precompute scaffold → indices
        # --------------------------------------------------
        scaffold_indices = defaultdict(list)
        for idx, (scaffold, _) in enumerate(self.DIfilteredBED):
            scaffold_indices[scaffold].append(idx)
    
        # --------------------------------------------------
        # 2. Precompute scaffold → positions array
        #    Use MapBC if available (chromosome-relative metric)
        # --------------------------------------------------
        if self.MapBC is not None:
            scaffold_arrays = {
                chrom: self.MapBC[i]
                for i, chrom in enumerate(self.chrom_keys)
            }
        else:
            scaffold_positions = defaultdict(list)
            for scaffold, pos in self.DIfilteredBED:
                scaffold_positions[scaffold].append(pos)

            scaffold_arrays = {
                scaffold: np.asarray(positions)
                for scaffold, positions in scaffold_positions.items()
            }

    
        # --------------------------------------------------
        # 3. Split genomes by scaffold (numeric form)
        # --------------------------------------------------
        # scaffold_haplotypes[scaffold] = list of np.arrays (one per individual)
        scaffold_haplotypes = {
            scaffold: [] for scaffold in scaffold_indices
        }
    
        for genome in self.DIfilteredGenomes:
            for scaffold, indices in scaffold_indices.items():
                # extract once, convert once
                s = ''.join(genome[i] for i in indices)
                s = s.replace("U", "3")     # DMBCtoucher {U,0,1,2} → {0,1,2,3}
                s = s.replace("_", "3")     # DMBCtoucher {U,0,1,2} → {0,1,2,3}
                scaffold_haplotypes[scaffold].append(
                    np.fromiter((ord(c) - 48 for c in s), dtype=np.int8)
                )
    
        # --------------------------------------------------
        # 4. Smooth ALL haplotypes per scaffold (key speedup)
        # --------------------------------------------------
        smoothed_scaffold_haplotypes = {}
    
        for chr_i, scaffold in enumerate(self.chrom_keys):
            haplos = scaffold_haplotypes[scaffold]
            haplo_matrix = np.vstack(haplos)

            rec_rate = self.chrRelativeRecRates[chr_i]

            if rec_rate <= 0:
                raise ValueError(
                    f"Invalid recombination rate for {scaffold}: {rec_rate}"
                )

            effective_scale = scale / rec_rate

            smoothed = smooth.laplace_smooth_multiple_haplotypes(
                scaffold_arrays[scaffold],
                haplo_matrix,
                effective_scale
            )

            smoothed_scaffold_haplotypes[scaffold] = smoothed

    
        # --------------------------------------------------
        # 5. Reassemble genomes (string form)
        # --------------------------------------------------
        n_individuals = len(self.DIfilteredGenomes)
        smoothed_split_genomes = [
            {} for _ in range(n_individuals)
        ]
    
        for scaffold, smoothed_matrix in smoothed_scaffold_haplotypes.items():
            for i in range(n_individuals):
                arr = smoothed_matrix[i]
                chars = np.where(arr == 3, "_", arr.astype(str)) # # DMBCtoucher {U,0,1,2} → {0,1,2,3}
                smoothed_split_genomes[i][scaffold] = ''.join(chars.tolist())
    
        # --------------------------------------------------
        # 6. Finalise
        # --------------------------------------------------
        self.DIfilteredGenomes_unsmoothed = self.DIfilteredGenomes
        self.DIfilteredGenomes = self._reconstruct_genomes(
            smoothed_split_genomes,
            scaffold_indices
        )

    def _reconstruct_genomes(self, smoothed_split_genomes, scaffold_indices):
        reconstructed_genomes = []
    
        for individual in smoothed_split_genomes:
            full_genome = ['0'] * len(self.DIfilteredBED)
    
            for scaffold, indices in scaffold_indices.items():
                scaffold_str = individual[scaffold]
                for i, idx in enumerate(indices):
                    full_genome[idx] = scaffold_str[i]
    
            reconstructed_genome = ''.join(full_genome)
            reconstructed_genomes.append(reconstructed_genome)
    
        return reconstructed_genomes
    
    
    def diem_dithering(self):

        
        # -------------------------------------------------
        # 2. Precompute chromosome spans
        # -------------------------------------------------
        self.length_of_chromosomes = {}
    
        start = 0
        for key, bed_data in zip(self.chrom_keys, self.DIfilteredBED_formatted):
            end = start + len(bed_data)
            self.length_of_chromosomes[key] = (start, end, len(bed_data))
            start = end
        # -------------------------------------------------
        # 3. Prepare iris_plot_prep ticks (vectorised shift)
        # -------------------------------------------------
        for idx, (key, bed) in enumerate(zip(self.chrom_keys, self.DIfilteredBED_formatted), start=1):
            x_ticks = fractional_positions_of_multiples(bed, self.ticks)
    
            offset = self.length_of_chromosomes[key][0]
            x_ticks[:, 1] += offset
    
            self.iris_plot_prep[idx] = x_ticks

        # -------------------------------------------------
        # 4. Calculate nBasesDithered SJEB 24 Jan 2026
        # -------------------------------------------------
        ringSpanInBases = 0;
        for chrRefPoses in self.DIfilteredBED_formatted:
            ringSpanInBases = ringSpanInBases + chrRefPoses[-1] - chrRefPoses[0] + 1
        
        # Input argument 'genome_pixels' is number of dithering 'pixels' along genome (pixels may be curved for iris plots)
        # Here, GappedQuotientSplitLengths takes the number of bases that should be dithered together.
        self.nBasesDithered = max(1,round(ringSpanInBases/self.genome_pixels))
        
        # -------------------------------------------------
        # 5. Perform dithering on the filtered data give nBasesDithered
        # -------------------------------------------------
        diem_dit_genomes_bed = [list(group) for _, group in groupby(self.DIfilteredBED, key=lambda x: x[0])]
        processed_diemDITgenomes = []
        for chr in diem_dit_genomes_bed:
            length_data = [row[1] for row in chr]
            split_lengths = self.GappedQuotientSplitLengths(length_data, self.nBasesDithered)# nBasesDithered was self.genome_pixels SJEB 24 Jan 2026 
            processed_diemDITgenomes.append(split_lengths)
        
        #processed_diemDITgenomes = Flatten(processed_diemDITgenomes)
        # IMPORTANT: EURG
        # processed_diemDITgenomes is now:
        #   List[chromosome][segment_length]
        
        diemDITgenomes = []

        # IMPORTANT: need chromosome order consistent with how processed_diemDITgenomes was built potential BUG
        chrom_keys = list(self.length_of_chromosomes.keys())

        for genome in self.DIfilteredGenomes:
            per_chr = []

            # iterate chromosomes in the SAME order as processed_diemDITgenomes
            for chr_i, chr_lengths in enumerate(processed_diemDITgenomes):

                chrom = chrom_keys[chr_i]
                g0, g1, _ = self.length_of_chromosomes[chrom]   # global [start,end) slice in concatenated genome

                genome_chr = genome[g0:g1]  # chromosome-local genome string
 
                # now take segments from the chromosome-local string
                string_take_result = StringTakeList(genome_chr, chr_lengths)

                state_count = Map(sStateCount, string_take_result)
                combined = list(zip(state_count, chr_lengths))
                compressed = self.DITcompress(combined)
                per_chr.append(self.Lengths2StartEnds(compressed))

            diemDITgenomes.append(per_chr)

        self.diemDITgenomes = diemDITgenomes
    
    def generate_plot_label(self, plot_theme):
        """ Generate the label for the plot """
        self.diemPlotLabel = f"{plot_theme} @ DI = {self.di_threshold}: {len(self.DIfilteredDATA)} sites ({self.DIpercent}%) {self.nBasesDithered} bases dithered."
    
    @staticmethod
    def GappedQuotientSplit(lst, Q):
        """
        Splits the list `lst` into sublists where consecutive elements share the same quotient when divided by `Q`.
        """
        quotients = [x // Q for x in lst]
    
        groups = []
        current_group = [lst[0]]
    
        for i in range(1, len(lst)):
            if quotients[i] == quotients[i - 1]:
                current_group.append(lst[i])
            else:
                groups.append(current_group)
                current_group = [lst[i]]
    
        groups.append(current_group)
        return groups
    
    def GappedQuotientSplitLengths(self, lst, Q):
        """
        Returns the lengths of the sublists produced by `gapped_quotient_split`.
        """
        return Map(len, self.GappedQuotientSplit(lst, Q))
    
    @staticmethod
    def normalize_4list(lst):
        """
        Normalizes a 4list by converting each element to its ratio of the total sum.
        Uses Fraction for precise comparison without floating-point errors.
        """
        total = sum(lst)
        if total == 0:
            return tuple(0 for _ in lst)  # Handle case where total is 0
        return tuple(Fraction(x, total) for x in lst)
    
    def DITcompress(self, DITl):
        """
        Compresses the list of {4list, length} tuples.
        """
        grouped_data = [list(group) for _, group in groupby(DITl, key=lambda x: self.normalize_4list(x[0]))]
        final_data = []
        for group in grouped_data:
            summed_states = [sum(x) for x in zip(*(item[0] for item in group))]
            summed_value = sum(item[1] for item in group)
            result = (summed_states, summed_value)
            final_data.append(result)
        return final_data
    
    @staticmethod
    def Lengths2StartEnds(stateNlen):
        lengths = [x[1] for x in stateNlen]
        ends = np.cumsum(lengths)
    
        # Calculate the start positions (end positions minus length plus 1)
        starts = ends - np.array(lengths) + 1
    
        # Combine states, starts, and ends into a list of triplets
        result = [(state, int(start), int(end)) for (state, start, end) in zip([x[0] for x in stateNlen], starts, ends)]
    
        return result
    


def flatten_ring_with_offsets(per_chr_ring, length_of_chromosomes):
    """
    Convert per-chromosome ring representation into a single
    global-coordinate ring suitable for IrisPlot / LongPlot.

    per_chr_ring:
        [
          [(w,s,e), ...],   # chromosome 0 (local coords)
          [(w,s,e), ...],   # chromosome 1
          ...
        ]

    length_of_chromosomes:
        dict preserving chromosome order:
          chrom -> (start, end, length)
    """
    flat = []
    chrom_keys = list(length_of_chromosomes.keys())

    for chr_idx, chr_segments in enumerate(per_chr_ring):
        chrom = chrom_keys[chr_idx]
        chrom_start = length_of_chromosomes[chrom][0]

        for weights, s, e in chr_segments:
            flat.append((
                weights,
                chrom_start + s - 1,   # preserve your 1→0 convention
                chrom_start + e
            ))

    return flat

def prefill_slider_cache(
    *,
    cache,                  # PlotCache instance
    namespace: str,
    basekey,
    di_values: Iterable[float],
    compute_fn: Callable[[float], object],  # returns payload to cache
    tol: Optional[float] = None,            # optional: skip if already cached "nearby"
    progress: str = "text",                 # "text", "tqdm", or "none"
    label: str = "Prefill cache",
):
    di_values = list(di_values)
    n = len(di_values)
    if n == 0:
        return

    use_tqdm = (progress == "tqdm")
    pbar = None
    if use_tqdm:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=n, desc=label)
        except Exception:
            use_tqdm = False  # fall back silently

    t0 = time.time()
    done = 0

    for i, di in enumerate(di_values, start=1):
        # optional "near" skip
        if tol is not None:
            hit = cache.get_nearest_float_key(
                namespace=namespace,
                basekey=basekey,
                x=float(di),
                tol=float(tol),
            )
            if hit is not None:
                done += 1
                if use_tqdm:
                    pbar.update(1)
                continue

        payload = compute_fn(float(di))
        cache.set_float_key(
            namespace=namespace,
            basekey=basekey,
            x=float(di),
            value=payload,
        )
        done += 1

        if use_tqdm:
            pbar.update(1)
        elif progress == "text":
            # lightweight text progress every ~5%
            if i == 1 or i == n or (i % max(1, n // 20) == 0):
                dt = time.time() - t0
                print(f"{label}: {i}/{n} ({100*i/n:.0f}%)  elapsed {dt:.1f}s")

    if use_tqdm and pbar is not None:
        pbar.close()


def _validate_polarized_diemtype_for_plots(diem_obj):
    """
    Validate DiemType-like object has the minimum polarized data required
    to build plotting prep structures directly in memory.
    """
    required_attrs = [
        "DMBC",
        "DIByChr",
        "posByChr",
        "chrNames",
        "chrLengths",
        "indNames",
    ]
    missing = [name for name in required_attrs if not hasattr(diem_obj, name)]
    if missing:
        raise ValueError(
            "DiemType object is missing required attributes for plotting: "
            + ", ".join(missing)
        )

    if diem_obj.DIByChr is None:
        raise ValueError(
            "DiemType object is not polarized. DIByChr is None. "
            "Run polarize() first."
        )

    n_chr = len(diem_obj.chrNames)
    if not (len(diem_obj.DMBC) == len(diem_obj.DIByChr) == len(diem_obj.posByChr) == n_chr):
        raise ValueError(
            "Inconsistent chromosome-wise lengths across DMBC, DIByChr, posByChr, and chrNames."
        )

    n_inds = len(diem_obj.indNames)
    for chr_i in range(n_chr):
        state_mat = np.asarray(diem_obj.DMBC[chr_i])
        di = np.asarray(diem_obj.DIByChr[chr_i])
        poses = np.asarray(diem_obj.posByChr[chr_i])

        if state_mat.ndim != 2:
            raise ValueError(f"DMBC[{chr_i}] must be a 2D array")
        if state_mat.shape[0] != n_inds:
            raise ValueError(
                f"DMBC[{chr_i}] has {state_mat.shape[0]} individuals but "
                f"indNames has {n_inds}"
            )
        if not (state_mat.shape[1] == di.shape[0] == poses.shape[0]):
            raise ValueError(
                f"Chromosome {diem_obj.chrNames[chr_i]} has inconsistent site counts: "
                f"DMBC sites={state_mat.shape[1]}, DI={di.shape[0]}, positions={poses.shape[0]}"
            )


def _state_matrix_to_diem_site_strings(state_matrix):
    """
    Convert one chromosome state matrix (nInds x nSites, states in {0,1,2,3})
    into BED-style diem_genotype site strings (prefix 'S', then chars in {_,0,1,2}).
    """
    state_mat = np.asarray(state_matrix)
    chars = np.full(state_mat.shape, "_", dtype="<U1")
    chars[state_mat == 1] = "0"
    chars[state_mat == 2] = "1"
    chars[state_mat == 3] = "2"
    return ["S" + "".join(site_chars) for site_chars in chars.T]


def _build_plot_inputs_from_diemtype(diem_obj):
    """
    Build the same core inputs used by DiemPlotPrep without reading BED/meta files.

    Returns:
        polarised_data (pd.DataFrame): columns chrom, start, DI, diem_genotype
        ind_ids (np.ndarray)
        chr_ref_lengths (np.ndarray)
        chr_relative_rec_rates (np.ndarray)
    """
    _validate_polarized_diemtype_for_plots(diem_obj)

    records = []
    for chr_i, chr_name in enumerate(diem_obj.chrNames):
        di = np.asarray(diem_obj.DIByChr[chr_i], dtype=float)
        poses = np.asarray(diem_obj.posByChr[chr_i], dtype=int)
        starts = np.maximum(poses - 1, 0)
        site_strings = _state_matrix_to_diem_site_strings(diem_obj.DMBC[chr_i])

        for j in range(len(di)):
            records.append(
                {
                    "chrom": chr_name,
                    "start": int(starts[j]),
                    "DI": float(di[j]),
                    "diem_genotype": site_strings[j],
                }
            )

    polarised_data = pd.DataFrame.from_records(
        records,
        columns=["chrom", "start", "DI", "diem_genotype"],
    )

    ind_ids = np.asarray(diem_obj.indNames)
    chr_ref_lengths = np.asarray(diem_obj.chrLengths)

    rr_dict = getattr(diem_obj, "relativeRecRateDict", None)
    if rr_dict is None:
        chr_relative_rec_rates = np.ones(len(diem_obj.chrNames), dtype=float)
    else:
        chr_relative_rec_rates = np.asarray(
            [float(rr_dict.get(chr_name, 1.0)) for chr_name in diem_obj.chrNames],
            dtype=float,
        )

    return polarised_data, ind_ids, chr_ref_lengths, chr_relative_rec_rates


def diemPlotPrepFromDiemType(plot_theme, diem_obj, di_threshold, genome_pixels, ticks, smooth=None):
    """
    Prepare iris/long plotting data directly from a polarized DiemType object.

    This bypasses BED/meta file reads and reuses the same downstream DiemPlotPrep
    logic used by diemPlotPrepFromBedMeta.
    """
    pzbed, bmIndIDs, chrRefLengths, chrRelativeRecRates = _build_plot_inputs_from_diemtype(diem_obj)

    prep = DiemPlotPrep(
        plot_theme=plot_theme,
        ind_ids=bmIndIDs,
        chrRefLengths=chrRefLengths,
        polarised_data=pzbed,
        di_threshold=di_threshold,
        diemStringPyCol=10,
        di_column=13,
        genome_pixels=genome_pixels,
        ticks=ticks,
        chrRelativeRecRates=chrRelativeRecRates,
        smooth=smooth,
    )

    return prep


def diemPlotPrepFromBedMeta(plot_theme, bed_file_path, meta_file_path,di_threshold,genome_pixels,ticks, smooth = None):

    pzbed, bmIndIDs, chrRefLengths, chrRelativeRecRates = read_diem_bed_4_plots(bed_file_path, meta_file_path)

    prep = DiemPlotPrep(
        plot_theme=plot_theme,
        ind_ids=bmIndIDs,
        chrRefLengths=chrRefLengths,
        polarised_data=pzbed,
        di_threshold=di_threshold,
        diemStringPyCol=10,
        di_column=13,
        genome_pixels=genome_pixels,
        ticks=ticks,
        chrRelativeRecRates=chrRelativeRecRates,
        smooth=smooth
    )
    
    return prep    
"""________________________________________ END DiemPlotPrep ___________________"""



"""________________________________________ START DiemIris ___________________"""

class WheelDiagram:
    """
    Utility class for creating wheel diagrams (iris plots).
    Args:
        subplot: Matplotlib subplot to draw on.
        center: Center coordinates of the wheel.
        radius: Outer radius of the wheel.
        number_of_rings: Number of concentric rings in the wheel.
        cutout_angle: Angle of the cutout section (default is 13 degrees).
    """
    def __init__(self, subplot, center, radius, number_of_rings, cutout_angle=13):
        self.subplot = subplot
        self.center = center
        self.radius = radius
        self.center_radius = radius / 2
        self.number_of_rings = number_of_rings
        self.cutout_angle = cutout_angle
        self.rings_added = 0

    def add_wedge(self, radius, from_angle, to_angle, color):
        self.subplot.add_artist(
            Wedge(self.center, radius, from_angle, to_angle, color=color, clip_on=False) # SJEB this last avoids a world of pain
        )

    def add_ring(self, list_of_thingies):
 #       print(f'Adding ring: {self.rings_added + 1}')
        available_angle = 360 - self.cutout_angle
        angle_scale = available_angle / list_of_thingies[-1][-1]
        colors = np.array(Map(mcolors.to_rgb,diemColours))

        ring_radius = self.radius - self.rings_added * (self.radius - self.center_radius) / self.number_of_rings

        start_angle_offset = 90
        for index, thing in enumerate(list_of_thingies):
            weights = np.array(thing[0])
            total_weight = np.sum(weights)
            if total_weight == 0:
                blended_rgb = (0, 0, 0)
            else:
                blended_rgb = np.sum(colors.T * weights, axis=1) / total_weight
            blended_hex = mcolors.to_hex(blended_rgb)
            from_angle = start_angle_offset + 360 - (angle_scale * (thing[1] - 1))
            to_angle = start_angle_offset + 360 - (angle_scale * thing[2])
            self.add_wedge(ring_radius, to_angle,from_angle, blended_hex)

        self.rings_added += 1

    def add_heatmap_ring(self, heatmap):
    # needs work. This version is specific to Honza's MolEcol figures.
        available_angle = 360 - self.cutout_angle
        angle_scale = available_angle / int(heatmap[-1][-1])
        keys = ["barr", "int", "ovm"]
        values = ["Red", "Blue", "Yellow"]
        color_map = dict(zip(keys, values))

        ring_radius = self.radius + 2 * (self.radius - self.center_radius) / self.number_of_rings

        start_angle_offset = 90
        for index, thing in enumerate(heatmap):
            from_angle = start_angle_offset + 360 - (angle_scale * (int(thing[1]) - 1))
            to_angle = start_angle_offset + 360 - (angle_scale * int(thing[2]))
            self.add_wedge(ring_radius, to_angle, from_angle, color_map[thing[0]])

    def clear_center(self):
        self.add_wedge(self.center_radius, 0, 360, "white")


"""________________________________________ END WheelDiagram ___________________"""

"""________________________________________ START diemIrisPlot ___________________"""




"""________________________________________ START Iris and Long helper function ___________________"""
def _restrict_chromosomes(
    *,
    input_data,
    refposes,
    length_of_chromosomes,
    bed_info=None,
    chrom_indices=None,
):
    """
    Restrict per-chromosome genome data to a subset of chromosomes and,
    if requested, pack them contiguously into a genome-global coordinate system.

    Canonical assumptions:
      - input_data[ind][chr] = list of (weights, start, end), chromosome-local
      - start/end are 1-based and inclusive (as produced by PlotPrep)
      - NO flattening occurs here
    """

    # -------------------------------------------------
    # Trivial case: no restriction
    # -------------------------------------------------
    if chrom_indices is None:
        return input_data, refposes, length_of_chromosomes, bed_info

    if length_of_chromosomes is None:
        raise ValueError("chrom_indices requires length_of_chromosomes")

    chrom_keys = list(length_of_chromosomes.keys())
    n_chr = len(chrom_keys)

    # -------------------------------------------------
    # Validate chromosome indices
    # -------------------------------------------------
    kept = []
    rejected = []

    for ci in chrom_indices:
        if isinstance(ci, (int, np.integer)) and 0 <= int(ci) < n_chr:
            kept.append(int(ci))
        else:
            rejected.append(ci)

    if rejected:
        print("restrict_chromosomes: rejected chromosome indices:", rejected)

    if not kept:
        raise ValueError("restrict_chromosomes: no valid chromosome indices")

    kept = sorted(kept)

    # -------------------------------------------------
    # Build packed chromosome offsets
    # -------------------------------------------------
    chrom_offsets = {}
    packed_cursor = 0.0

    for chr_idx in kept:
        chrom = chrom_keys[chr_idx]
        _, _, L = length_of_chromosomes[chrom]

        chrom_offsets[chr_idx] = packed_cursor
        packed_cursor += L

    # -------------------------------------------------
    # Remap input_data (per individual)
    # -------------------------------------------------
    new_input_data = []

    for indiv in input_data:
        new_ring = []

        for chr_idx in kept:
            offset = chrom_offsets[chr_idx]
            chr_segments = indiv[chr_idx]

            for weights, s, e in chr_segments:
                # s,e are chromosome-local (1-based)
                new_ring.append((
                    weights,
                    offset + s,
                    offset + e
                ))

        new_input_data.append(new_ring)

    # -------------------------------------------------
    # Restrict refposes (order preserved)
    # -------------------------------------------------
    new_refposes = [refposes[i] for i in kept]

    # -------------------------------------------------
    # Remap chromosome lengths (geometry only)
    # -------------------------------------------------
    length_of_chromosomes_remapped = {}
    for chr_idx in kept:
        chrom = chrom_keys[chr_idx]
        _, _, L = length_of_chromosomes[chrom]

        start = chrom_offsets[chr_idx]
        end = start + L
        length_of_chromosomes_remapped[chrom] = (start, end, L)


    
    # -------------------------------------------------
    # Remap bed_info (outer ticks) — GLOBAL → PACKED
    # -------------------------------------------------
    new_bed = None
    if bed_info is not None:
        new_bed = {}

        for chr_idx in kept:
            bed_key = chr_idx + 1  # iris-style 1-based indexing
            if bed_key not in bed_info:
                continue

            chrom = chrom_keys[chr_idx]
            chrom_start, _, _ = length_of_chromosomes[chrom]
            offset = chrom_offsets[chr_idx]

            positions = []
            for label, pos in bed_info[bed_key]:
                # pos is GLOBAL; convert to chromosome-local then pack
                pos2 = offset + (pos - chrom_start)
                positions.append((label, pos2))

            if positions:
                new_bed[chrom] = positions


    return (
        new_input_data,
        new_refposes,
        length_of_chromosomes_remapped,
        new_bed,
    )


"""________________________________________ END Iris and Long helper function ___________________"""



def diemIrisFromPlotPrep(prepped, chrom_indices=None):
    """
    Uses per-chromosome diemDITgenomes_ordered as the canonical form.

    - If chrom_indices is None:
        flatten to whole-genome rings for WheelDiagram
    - If chrom_indices is provided:
        pass per-chromosome structure through unchanged
        (_restrict_chromosomes will select + pack)
    """

    if chrom_indices is None:
        # Whole-genome plot → flatten for WheelDiagram
        input_data = [
            flatten_ring_with_offsets(
                ring,
                prepped.length_of_chromosomes
            )
            for ring in prepped.diemDITgenomes_ordered
        ]
    else:
        # Chromosome-restricted plot → MUST stay per-chromosome
        input_data = prepped.diemDITgenomes_ordered

    diemIrisPlot(
        title=prepped.diemPlotLabel,
        input_data=input_data,
        refposes=prepped.DIfilteredBED_formatted,
        names=prepped.IndIDs_ordered,
        bed_info=prepped.iris_plot_prep,
        length_of_chromosomes=prepped.length_of_chromosomes,
        chrom_indices=chrom_indices,
    )


def diemIrisFromDiemType(
    plot_theme,
    diem_obj,
    di_threshold,
    genome_pixels,
    ticks,
    smooth=None,
    chrom_indices=None,
):
    """
    Convenience wrapper: build plotting prep from a polarized DiemType object
    and render an iris plot in one call.

    Returns:
        DiemPlotPrep object used for plotting.
    """
    prepped = diemPlotPrepFromDiemType(
        plot_theme=plot_theme,
        diem_obj=diem_obj,
        di_threshold=di_threshold,
        genome_pixels=genome_pixels,
        ticks=ticks,
        smooth=smooth,
    )
    diemIrisFromPlotPrep(prepped, chrom_indices=chrom_indices)
    return prepped

"""________________________________________ END DiemIris ___________________"""


"""________________________________________ START diemLongPlot ___________________"""



class BrickDiagram:
    """
    Utility class for creating linear (brick) genome diagrams.

    Each ring is a horizontal band.
    Each brick spans a genomic interval [start, end).
    """

    def __init__(self, subplot, x_min, x_max, y_min, y_max, number_of_rings):
        self.subplot = subplot
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.number_of_rings = number_of_rings
        self.rings_added = 0

        self.ring_height = (y_max - y_min) / number_of_rings

    def add_brick(self, x0, x1, ring_idx, color):
        y0 = self.y_max - (ring_idx + 1) * self.ring_height
        width = x1 - x0

        rect = Rectangle(
            (x0, y0),
            width,
            self.ring_height,
            facecolor=color,
            edgecolor=None,
            linewidth=0,
            clip_on=False
        )
        self.subplot.add_patch(rect)

    def add_ring(self, list_of_thingies, colors):
        """
        list_of_thingies: [(weights, start_pos, end_pos), ...]
        colors: RGB base colours (same as iris)
        """
        ring_idx = self.rings_added

        for thing in list_of_thingies:
            weights, start, end = thing
            weights = np.asarray(weights)

            total = weights.sum()
            if total == 0:
                blended_rgb = (0, 0, 0)
            else:
                blended_rgb = (colors.T * weights).sum(axis=1) / total

            self.add_brick(start, end, ring_idx, mcolors.to_hex(blended_rgb))

        self.rings_added += 1





class BrickDiagram:
    """
    Draws horizontal rings made of rectangles spanning [x0,x1) in data coords.
    """
    def __init__(self, ax, n_rings, y_min=0.0, y_max=1.0):
        self.ax = ax
        self.n_rings = n_rings
        self.y_min = y_min
        self.y_max = y_max
        self.ring_h = (y_max - y_min) / n_rings

    def add_brick(self, x0, x1, ring_idx, color):
        y0 = self.y_max - (ring_idx + 1) * self.ring_h
        self.ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                self.ring_h,
                facecolor=color,
                edgecolor="none",
                linewidth=0,
                clip_on=False
            )
        )






def diemLongFromPlotPrep(prepped, chrom_indices=None):
    """
    Uses per-chromosome diemDITgenomes_ordered as the canonical form.

    - If chrom_indices is None:
        flatten to whole-genome rings for WheelDiagram
    - If chrom_indices is provided:
        pass per-chromosome structure through unchanged
        (_restrict_chromosomes will select + pack)
    """

    if chrom_indices is None:
        # Whole-genome plot → flatten for WheelDiagram
        input_data = [
            flatten_ring_with_offsets(
                ring,
                prepped.length_of_chromosomes
            )
            for ring in prepped.diemDITgenomes_ordered
        ]
    else:
        # Chromosome-restricted plot → MUST stay per-chromosome
        input_data = prepped.diemDITgenomes_ordered

    diemLongPlot(
        title=prepped.diemPlotLabel,
        input_data=input_data,
        refposes=prepped.DIfilteredBED_formatted,
        names=prepped.IndIDs_ordered,
        bed_info=prepped.iris_plot_prep,
        length_of_chromosomes=prepped.length_of_chromosomes,
        chrom_indices=chrom_indices,
    )


def diemLongFromDiemType(
    plot_theme,
    diem_obj,
    di_threshold,
    genome_pixels,
    ticks,
    smooth=None,
    chrom_indices=None,
):
    """
    Convenience wrapper: build plotting prep from a polarized DiemType object
    and render a long plot in one call.

    Returns:
        DiemPlotPrep object used for plotting.
    """
    prepped = diemPlotPrepFromDiemType(
        plot_theme=plot_theme,
        diem_obj=diem_obj,
        di_threshold=di_threshold,
        genome_pixels=genome_pixels,
        ticks=ticks,
        smooth=smooth,
    )
    diemLongFromPlotPrep(prepped, chrom_indices=chrom_indices)
    return prepped


"""________________________________________ END diemLongPlot ___________________"""


def diemIrisPlot(
    input_data,
    refposes,
    title=None,
    names=None,
    bed_info=None,
    length_of_chromosomes=None,
    heatmap=None,
    chrom_indices=None,   # optional (same as long)
    show_outer_ticks=True,
):
    # -------------------------------------------------
    # Figure & axes
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    if title is not None:
        ax.set_title(title, pad=20)

    # Move axes DOWN to make room for title (no resizing distortion)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.96])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # keep wheel circular regardless of margins
    ax.set_box_aspect(1)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # -------------------------------------------------
    # Chromosome restriction (shared helper)
    # -------------------------------------------------
    (
        input_data,
        refposes,
        length_of_chromosomes_remapped,
        bed_info_remapped,
    ) = _restrict_chromosomes(
        input_data=input_data,
        refposes=refposes,
        length_of_chromosomes=length_of_chromosomes,
        bed_info=bed_info,
        chrom_indices=chrom_indices,
    )

    # -------------------------------------------------
    # Wheel geometry
    # -------------------------------------------------
    center = np.array((0.5, 0.5))
    radius = 0.48
    cutout_angle = 20
    number_of_rings = len(input_data)

    wd = WheelDiagram(
        ax,
        center,
        radius,
        number_of_rings + (1 if heatmap is not None else 0),
        cutout_angle=cutout_angle,
    )

    if heatmap is not None:
        wd.add_heatmap_ring(heatmap)


    for ring in input_data:
        wd.add_ring(ring)

    wd.clear_center()

    # -------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------
    available_angle = 360 - cutout_angle
    start_angle_offset = 90

    if length_of_chromosomes_remapped is not None:
        max_position = max(end for (_, end, _) in length_of_chromosomes_remapped.values())
    else:
        max_position = input_data[0][-1][2]

    ring_width = (radius - wd.center_radius) / number_of_rings

    # -------------------------------------------------
    # Inner chromosome wedges + labels
    # -------------------------------------------------
    chrom_ranges = []
    if length_of_chromosomes_remapped is not None:
        for chrom, (start, end, _) in length_of_chromosomes_remapped.items():
            chrom_ranges.append((chrom, float(start), float(end)))

    if chrom_ranges:
        for idx, (chrom, start, end) in enumerate(chrom_ranges):
            start_angle = start_angle_offset + 360 - available_angle * start / max_position
            end_angle   = start_angle_offset + 360 - available_angle * end   / max_position

            if idx % 2 == 1:
                ax.add_artist(
                    Wedge(center, radius / 2, end_angle, start_angle,
                          color="lightgrey", alpha=0.3)
                )

            midpoint = 0.5 * (start + end)
            mid_angle = start_angle_offset + 360 - available_angle * midpoint / max_position
            mid_rad = np.deg2rad(mid_angle)

            label_xy = center + (radius - 0.28) * np.array([np.cos(mid_rad), np.sin(mid_rad)])

            ax.text(
                label_xy[0], label_xy[1],
                Chr_Nickname(chrom),
                ha="center", va="center",
                fontsize=8,
                rotation=mid_angle,
                rotation_mode="anchor",
            )

        ax.add_artist(Wedge(center, 0.18, 0, 360, color="white"))

    # -------------------------------------------------
    # Outer ticks (iris)
    # -------------------------------------------------
    if show_outer_ticks and bed_info_remapped is not None:
        outer_radius = radius + (0.035 if heatmap is not None else 0.015)

        for positions in bed_info_remapped.values():
            for label, position in positions:
                position = float(position)
                angle = start_angle_offset + 360 - available_angle * position / max_position
                ang_rad = np.deg2rad(angle)

                base = center + outer_radius * np.array([np.cos(ang_rad), np.sin(ang_rad)])
                text_pos = base + 0.006 * np.array([np.cos(ang_rad), np.sin(ang_rad)])

                ax.text(
                    text_pos[0], text_pos[1],
                    str(int(label)),
                    ha="left", va="center",
                    fontsize=6,
                    rotation=angle,
                    rotation_mode="anchor",
                )

                line_start = base - 0.01 * np.array([np.cos(ang_rad), np.sin(ang_rad)])
                line_end   = line_start + 0.01 * np.array([np.cos(ang_rad), np.sin(ang_rad)])

                ax.plot([line_start[0], line_end[0]],
                        [line_start[1], line_end[1]],
                        color="black", linewidth=0.5)

    # -------------------------------------------------
    # Ring labels (optional)
    # -------------------------------------------------
    if names is not None and len(names) == number_of_rings:
        for i, name in enumerate(names):
            ring_radius = radius - (i + 0.5) * ring_width
            label_xy = center + ring_radius * np.array([0, 1])
            ax.text(label_xy[0], label_xy[1], name, ha="right", va="center", fontsize=4)

    # -------------------------------------------------
    # Hover (make it actually visible)
    # -------------------------------------------------
    # Key change: do NOT return all-spaces fallback (looks like “nothing” in some backends).
    # Use a quiet but visible fallback.
    fallback = ""

    def iris_format_coord(x, y):
        dx = x - center[0]
        dy = y - center[1]
        r = np.hypot(dx, dy)

        # outside wheel or inside hole
        if r < wd.center_radius or r > radius:
            return fallback

        angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        rel_angle = (start_angle_offset + 360 - angle) % 360
        if rel_angle > available_angle:
            return fallback

        raw_pos = rel_angle / available_angle * max_position

        # chromosome lookup
        chrom_label = None
        bp = None
        for chrom_idx, (chrom, start, end) in enumerate(chrom_ranges):
            if start <= raw_pos < end:
                frac = (raw_pos - start) / (end - start)
                ref = refposes[chrom_idx]
                if len(ref) == 0:
                    return fallback
                ref_idx = int(np.clip(round(frac * (len(ref) - 1)), 0, len(ref) - 1))
                bp = ref[ref_idx]
                chrom_label = Chr_Nickname(chrom)
                break

        if chrom_label is None or bp is None:
            return fallback

        # ring lookup
        ring_idx = int((radius - r) / ring_width)
        if ring_idx < 0 or ring_idx >= number_of_rings:
            return fallback

        sample = names[ring_idx] if (names is not None and len(names) == number_of_rings) else f"ring {ring_idx}"
        return f"{chrom_label}   bp={int(bp):,}   sample={sample}"

    ax.format_coord = iris_format_coord
    plt.show()




def diemLongPlot(
    input_data,
    refposes,
    chrom_indices=None,              # NOW OPTIONAL (matches iris)
    title=None,
    names=None,
    bed_info=None,
    length_of_chromosomes=None,
    show_outer_ticks=True,
):
    """
    Linear analogue of diemIrisPlot, using BrickDiagram.

    Same semantics as iris:
      - chrom_indices is optional.
      - If chrom_indices is provided, chromosomes are remapped into a packed coordinate system.
      - Hover reports chrom + bp + sample.
      - Outer ticks are supported (bed_info) and drawn above the rings.
    """

    if length_of_chromosomes is None:
        raise ValueError("diemLongPlot: length_of_chromosomes is required.")

    # -------------------------------------------------
    # Chromosome restriction (shared helper)
    # -------------------------------------------------
    (
        input_data,
        refposes,
        length_of_chromosomes_remapped,
        bed_info_remapped,
    ) = _restrict_chromosomes(
        input_data=input_data,
        refposes=refposes,
        length_of_chromosomes=length_of_chromosomes,
        bed_info=bed_info,
        chrom_indices=chrom_indices,
    )

    # -------------------------------------------------
    # Normalize bed_info keys for unrestricted Long plot
    # -------------------------------------------------
    if chrom_indices is None and bed_info_remapped is not None:
        # Convert 1-based BED keys → chromosome-name keys
        chrom_keys = list(length_of_chromosomes.keys())
        bed_info_remapped = {
            chrom_keys[k - 1]: v
            for k, v in bed_info_remapped.items()
            if 1 <= k <= len(chrom_keys)
        }


    # -------------------------------------------------
    # Packed chromosome ranges
    # -------------------------------------------------
    chrom_ranges = []
    for chrom, v in (length_of_chromosomes_remapped or length_of_chromosomes).items():
        start, end = float(v[0]), float(v[1])
        chrom_ranges.append((chrom, start, end))

    if not chrom_ranges:
        raise ValueError("diemLongPlot: no chromosomes available after restriction.")

    packed_len = max(end for (_, _, end) in chrom_ranges)

    # -------------------------------------------------
    # Figure & axes
    # -------------------------------------------------
    n_rings = len(input_data)
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.86)

    if title is not None:
        ax.set_title(title, pad=16)
    # Move axes DOWN to make room for outer ticks (no geometry distortion)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.94])

    ax.set_xlim(0, packed_len)

    # Allocate room above rings for outer ticks
    ax.set_ylim(-0.9, n_rings + 0.7)

    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # -------------------------------------------------
    # Draw rings as bricks (already packed if chrom_indices was used)
    # -------------------------------------------------
    colors = np.array(list(map(mcolors.to_rgb, diemColours)))
    bd = BrickDiagram(ax, n_rings, y_min=0, y_max=n_rings)

    for ring_idx, ring in enumerate(input_data):
        for weights, x0, x1 in ring:
        
            # conditionally convert 1-based to 0-based
            if True or chrom_indices is None:
                # global genome: 1-based → 0-based
                x0 = float(x0 - 1)
                x1 = float(x1)
            else:
                # already packed by _restrict_chromosomes
                x0 = float(x0)
                x1 = float(x1)
            x1 = float(x1)
            if x1 <= x0:
                continue

            w = np.asarray(weights)
            tot = float(np.sum(w))
            if tot == 0:
                blended_rgb = (0, 0, 0)
            else:
                blended_rgb = (colors.T * w).sum(axis=1) / tot

            bd.add_brick(x0, x1, ring_idx, mcolors.to_hex(blended_rgb))

    # -------------------------------------------------
    # Chromosome bricks + labels at base
    # -------------------------------------------------
    base_y = -0.62
    for i, (chrom, p0, p1) in enumerate(chrom_ranges):
        if i % 2 == 1:
            ax.axvspan(p0, p1, color="grey", alpha=0.35,
                       ymin=-0.18, ymax=0.01, clip_on=False)

        ax.text(
            0.5 * (p0 + p1),
            base_y - 0.12,
            Chr_Nickname(chrom),
            ha="center",
            va="top",
            fontsize=8,
            rotation=90
        )


    # -------------------------------------------------
    # Outer ticks (linear analogue of iris outer ticks)
    # -------------------------------------------------
    if show_outer_ticks and bed_info_remapped is not None:
        tick_y0 = n_rings + 0 + 0.05
        tick_y1 = n_rings + 1 + 0.18
        text_y  = n_rings + 2 + 0.22

        # NOTE: positions are already packed coords

        for chrom, p0, p1 in chrom_ranges:
            if chrom not in bed_info_remapped:
                continue
            for label, pos in bed_info_remapped[chrom]:
        #for positions in bed_info_remapped.values():
        #    for label, pos in positions:
                x = float(pos)
                ax.plot([x, x], [tick_y0, tick_y1],
                        color="black", linewidth=0.5, clip_on=False)
                ax.text(
                    x, text_y,
                    str(int(label)),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                    clip_on=False
                )

    # -------------------------------------------------
    # Ring labels (optional)
    # -------------------------------------------------
    if names is not None and len(names) == n_rings:
        for i, name in enumerate(names):
            y = n_rings - i - 0.5
            ax.text(
                -0.01 * packed_len,
                y,
                name,
                ha="right",
                va="center",
                fontsize=6,
                clip_on=False
            )

    # -------------------------------------------------
    # Hover (same semantics as iris)
    # -------------------------------------------------
    fallback = ""

    def long_format_coord(x, y):
        # ring index from y
        ring_idx = int(np.floor(n_rings - y))
        if ring_idx < 0 or ring_idx >= n_rings:
            return fallback

        # find chromosome interval in packed coords
        chrom_hit = None
        for chrom_i, (chrom, p0, p1) in enumerate(chrom_ranges):
            if p0 <= x < p1:
                chrom_hit = (chrom_i, chrom, p0, p1)
                break
        if chrom_hit is None:
            return fallback

        chrom_i, chrom, p0, p1 = chrom_hit
        if p1 <= p0:
            return fallback

        frac = (x - p0) / (p1 - p0)
        ref = refposes[chrom_i]
        if len(ref) == 0:
            return fallback

        ref_i = int(np.clip(round(frac * (len(ref) - 1)), 0, len(ref) - 1))
        bp = ref[ref_i]

        sample = names[ring_idx] if (names is not None and len(names) == n_rings) else f"ring {ring_idx}"
        return f"{Chr_Nickname(chrom)}   bp={int(bp):,}   sample={sample}"

    ax.format_coord = long_format_coord
    plt.show()