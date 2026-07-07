from . import polarize as pol
from . import smooth as ks
# import kernel_smoothing_functions as ks
import numpy as np
import copy
import pandas as pd

def check_both_none(thing1,thing2):
    '''
    Utility function to check if two things are both None.
    Returns True if both are None, False otherwise.
    '''
    if thing1 is None and thing2 is None:
        return True
    return False    

def check_differ_none(thing1,thing2):
    '''
    Utility function to check if only one of two things is None.
    Returns True if one is None and the other is not, False otherwise.
    '''
    if thing1 is None and thing2 is not None:
        return True
    if thing1 is not None and thing2 is None:
        return True
    return False
    

def compare_DiemTypes(dTest,dSolution):

    testPassed = True
    AttributesFailedList = []

    # check that the attributes of the test DiemType match those of the solution DiemType
    # we assume that both dTest and dSolution are DiemType objects
    # we also assume that they have the same number of chromosomes and individuals
    # we do not check that here, but we could add that if needed
    # we also assume that the order of individuals and chromosomes is the same in both objects
    # if that is not the case, we could add code to reorder them to match
    # but for now we just assume they are the same
   

    # dTest could have additional attributes than dSolution.
    # we only check attributes that (should) be shared between them
    # we check list attributes, non-list attributes, and float attributes separately, then we check if the contig matrices are the same

    print("Now checking list attributes")
    for key in ['DMBC','PolByChr','DIByChr','SupportByChr','posByChr','MapBC','chrPloidies','initialPolByChr']:
        print('checking',key)

        if check_both_none(dSolution.__dict__[key], dTest.__dict__[key]):
            #print("Both are None, so continuing")
            continue
        if check_differ_none(dSolution.__dict__[key], dTest.__dict__[key]):
            print("Attribute ",key," differs (one is None, the other is not)")
            testPassed = False
            AttributesFailedList.append(key)
            continue
        
        # in all other cases, just compare the arrays within the lists
        for idx in range(len(dSolution.DMBC)):
            arr1 = dSolution.__dict__[key][idx]
            arr2 = dTest.__dict__[key][idx]
            if not np.array_equal(arr1,arr2):
                print("Attribute ",key," differs for chromosome ",idx)
                testPassed = False
                AttributesFailedList.append(key)
    
    print("Now checking non-list attributes")
    for key in ['indNames','chrNames','chrLengths','HIs']:
        #this code is to handle the case where HIs have not been computed, and are thus None

        if check_both_none(dSolution.__dict__[key], dTest.__dict__[key]):
            #print("Both are None, so continuing")
            continue
        if check_differ_none(dSolution.__dict__[key], dTest.__dict__[key]):
            print("Attribute ",key," differs (one is None, the other is not)")
            testPassed = False
            AttributesFailedList.append(key)
            continue

        # in all other cases, just compare the arrays
        arr1 = dSolution.__dict__[key]
        arr2 = dTest.__dict__[key]
        if not np.array_equal(arr1,arr2):
            print("Attribute ",key," differs")
            testPassed = False
            AttributesFailedList.append(key)


    print("Now checking float attributes")
    for key in ['threshold','smoothScale']:
        if check_both_none(dSolution.__dict__[key], dTest.__dict__[key]):
            #print("Both are None, so continuing")
            continue
        if check_differ_none(dSolution.__dict__[key], dTest.__dict__[key]):
            print("Attribute ",key," differs (one is None, the other is not)")
            testPassed = False
            AttributesFailedList.append(key)
            continue

        # in all other cases, just compare the values
        if dSolution.__dict__[key] != dTest.__dict__[key]:
            print("Attribute ",key," differs")
            testPassed = False
            AttributesFailedList.append(key)

    # Compare relativeRecRateDict attribute
    print("Now checking relativeRecRateDict attribute")
    key = 'relativeRecRateDict'
    # Check if both have the attribute
    has_attr_solution = hasattr(dSolution, key)
    has_attr_test = hasattr(dTest, key)
    if not has_attr_solution and not has_attr_test:
        pass  # Both do not have the attribute, skip
    elif has_attr_solution != has_attr_test:
        print(f"Attribute {key} differs (one object has it, the other does not)")
        testPassed = False
        AttributesFailedList.append(key)
    else:
        # Both have the attribute, compare values
        val1 = getattr(dSolution, key)
        val2 = getattr(dTest, key)
        if check_both_none(val1, val2):
            pass
        elif check_differ_none(val1, val2):
            print(f"Attribute {key} differs (one is None, the other is not)")
            testPassed = False
            AttributesFailedList.append(key)
        elif val1 != val2:
            print(f"Attribute {key} differs")
            testPassed = False
            AttributesFailedList.append(key)

    print('now checking individual exclusions')
    key = 'indExclusions'
    if check_both_none(dSolution.__dict__[key], dTest.__dict__[key]):
        #print("Both are None, so continuing")
        pass
    elif check_differ_none(dSolution.__dict__[key], dTest.__dict__[key]):
        print("Attribute ",key," differs (one is None, the other is not)")
        testPassed = False
        AttributesFailedList.append(key)
    else:
        same = np.array_equal(np.sort(dSolution.__dict__[key]), np.sort(dTest.__dict__[key]))
        if not same:
            print("Attribute ",key," differs")
            testPassed = False
            AttributesFailedList.append(key)
    
    print('now checking site exclusions')
    key = 'siteExclusionsByChr'
    if check_both_none(dSolution.__dict__[key], dTest.__dict__[key]):
        #print("Both are None, so continuing")
        pass
    elif check_differ_none(dSolution.__dict__[key], dTest.__dict__[key]):
        print("Attribute ",key," differs (one is None, the other is not)")
        testPassed = False
        AttributesFailedList.append(key)
    else:
        for idx in range(len(dSolution.__dict__['chrNames'])):
            arr1 = dSolution.__dict__[key][idx]
            arr2 = dTest.__dict__[key][idx]
            if check_both_none(arr1, arr2):
                #print("Both are None, so continuing")
                continue
            if check_differ_none(arr1, arr2):
                print("Attribute ",key," differs for chromosome ",dSolution.__dict__['chrNames'][idx]," (one is None, the other is not)")
                testPassed = False
                AttributesFailedList.append(key)
                break
            if not np.array_equal(np.sort(arr1), np.sort(arr2)):
                print("Attribute ",key," differs for chromosome ",dSolution.__dict__['chrNames'][idx])
                testPassed = False
                AttributesFailedList.append(key)
                break


    if testPassed == False:
        print('previous attributes failed, so skipping contigMatrix check')
    else:
        print("now checking contigMatrix attribute")
        key = 'contigMatrix'
        if check_both_none(dSolution.__dict__[key], dTest.__dict__[key]):
            #print("Both are None, so continuing")
            pass
        elif check_differ_none(dSolution.__dict__[key], dTest.__dict__[key]):
            print("Attribute ",key," differs (one is None, the other is not)")
            testPassed = False
            AttributesFailedList.append(key)
        else:
            if not compare_contig_matrices(dSolution.__dict__[key],dTest.__dict__[key]):
                print("Attribute ",key," differs (neither is None) ")
                testPassed = False
                AttributesFailedList.append(key)
            


    if testPassed:
        print("All tests passed!")
    else:
        AttributesFailedList = list(set(AttributesFailedList))
        print("The following attributes differed: ",AttributesFailedList)   


def compare_intervals(iv1,iv2):
    # compares two Interval objects
    # returns True if they are the same, False otherwise
    attrs = ['chrName','indName','idxl','idxr','l','r','mapl','mapr','span','mapspan','state']
    for attr in attrs:
        if getattr(iv1,attr) != getattr(iv2,attr):
            print("Attribute ",attr," differs: ",getattr(iv1,attr)," vs ",getattr(iv2,attr))
            return False
    return True

def compare_interval_lists(list1,list2):
    # compares two lists of Interval objects
    # returns True if they are the same, False otherwise
    if len(list1) != len(list2):
        print("Lists differ in length: ",len(list1)," vs ",len(list2))
        return False
    for iv1,iv2 in zip(list1,list2):
        if not compare_intervals(iv1,iv2):
            return False
    return True


def compare_contig(c1,c2):
    # compares two Contig objects
    # returns True if they are the same, False otherwise

    if c1.chrName != c2.chrName or c1.indName != c2.indName:
        print("Contigs differ in chrName or indName: ",c1.chrName,c1.indName," vs ",c2.chrName,c2.indName)
        return False
    if not compare_interval_lists(c1.intervals,c2.intervals):
        print("Contigs differ in intervals")
        return False
    return True


def compare_contig_matrices(cm1,cm2):
    # compares the contigMatrix of two different contig matrices objects. It is an array with rows = chromosomes and columns = individuals.

    # returns True if they are the same, False otherwise. Could return more information about where they differ if needed.
    # print(cm1)
    # print(cm2)
    if check_both_none(cm1,cm2):
        #print("Both contigMatrices are None")
        return True
    elif check_differ_none(cm1,cm2):
        print("contigMatrix differs (one is None, the other is not)")
        return False
    else:
        nChrs = len(cm1)
        nInds = len(cm1[0])
        for idxChr in range(nChrs):
            for idxInd in range(nInds):
                c1 = cm1[idxChr][idxInd]
                c2 = cm2[idxChr][idxInd]
                if not compare_contig(c1,c2):
                    print("Contigs differ at chr index ",idxChr," and ind index ",idxInd)
                    return False
        return True