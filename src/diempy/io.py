import pandas as pd
import numpy as np
from . import diemtype as dt

# this is a sketch of a function for importing the bed file data and constructing a diemtype object from it
# uses the output of vcf2diembed version from Mid october that stuart sent us  

# AI generated faster version of read_diem_bed approximately 2x speed up
def read_diem_bed(bed_file_path, meta_file_path):
    """
    Fast version of read_diem_bed with significant performance improvements.
    
    Parameters:
    bed_file_path (str): Path to the diem BED file.
    meta_file_path (str): Path to the diem metadata file.

    Returns:
    DiemType: DiemType object containing the diem BED data.
    """
    
    # Read metadata - no changes needed here as it's already fast
    df_meta = pd.read_csv(meta_file_path, sep='\t')
    chrNames = np.array(df_meta['#Chrom'].values)
    chrLengths = np.array(df_meta['RefEnd0'].values) - np.array(df_meta['RefStart0'].values)
    chrRecRates = np.array(df_meta['relativeRecRates'].values)
    sampleNames = np.array(df_meta.columns[7:])
    
    ploidyByChr = []
    for chr in chrNames:
        row = df_meta[df_meta['#Chrom'] == chr]
        ploidy = np.array(row.iloc[0,7:].values, dtype=int)
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
    
    # Create chromosome lookup for faster filtering
    chr_groups = df_bed.groupby('chrom')
    
    # Fixed vectorized genotype mapping
    def map_gt_vectorized(gt_strings):
        """Vectorized genotype mapping - handles '_' and other characters properly"""
        # Convert to numpy array of characters, skipping first character
        gt_chars = np.array([list(s)[1:] for s in gt_strings])
        
        # Create result array
        result = np.zeros(gt_chars.shape, dtype=np.int8)
        
        # Map characters using boolean indexing (much safer than ASCII lookup)
        result[gt_chars == '0'] = 1
        result[gt_chars == '1'] = 2
        result[gt_chars == '2'] = 3
        # Everything else (including '_') stays as 0
        
        return result.T
    
    # Process data by chromosome - avoid repeated filtering
    positionByChr = []
    DMBC = []
    
    if hasPolarity:
        polarityByChr = []
        initialPolByChr = []
        DIByChr = []
        supportByChr = []
        siteExclusionsByChr = []
        allExclusionsAreNone = True
    
    for chr in chrNames:
        thisDF = chr_groups.get_group(chr)
        
        # Positions
        positions = thisDF['end'].values.astype(int)
        positionByChr.append(positions)
        
        # Genotypes - vectorized processing
        allele_matrix = map_gt_vectorized(thisDF['diem_genotype'].values)
        DMBC.append(allele_matrix)
        
        if hasPolarity:
            # Process all polarity-related data at once
            polarityByChr.append(thisDF['polarity'].values.astype(int))
            initialPolByChr.append(thisDF['nullPolarity'].values.astype(int))
            DIByChr.append(thisDF['DI'].values.astype(np.float64))
            supportByChr.append(thisDF['Support'].values.astype(np.float64))
            
            # Site exclusions
            masked_vals = thisDF['masked'].values.astype(int)
            if np.all(masked_vals == 0):
                siteExclusionsByChr.append(None)
            else:
                siteExclusionsByChr.append(np.where(masked_vals == 1)[0])
                allExclusionsAreNone = False
    
    # Create DiemType object
    d = dt.DiemType(DMBC, sampleNames, ploidyByChr, chrNames, positionByChr, chrLengths)
    d.indExclusions = individualsMasked
    d.relativeRecRateDict = dict(zip(chrNames, chrRecRates))
    if hasPolarity:
        d.PolByChr = polarityByChr
        d.DIByChr = DIByChr
        d.SupportByChr = supportByChr
        d.initialPolByChr = initialPolByChr
       
        
        if allExclusionsAreNone:
            d.siteExclusionsByChr = None
        else:
            d.siteExclusionsByChr = siteExclusionsByChr
        
        # Vectorized polarity flipping
        for idx in range(len(chrNames)):
            d.DMBC[idx] = dt.flip_polarity(d.DMBC[idx], d.PolByChr[idx])
    
        d.HIs = d.computeHIs()
    return d




# note: by copying the input bed file and adding columns to it, we preserve all original information. It also means that even if diemtype object being saved has had its individuals reordered by HI, the output bed file will still have the original order.  
# if we store all the added info in the diemtype object so that we do not need to re-read the input bed file when saving the polarized data, we also need to keep track of whether the data has been reordered or not.

#AI written version. approximately 3x speed up. 
def write_polarized_bed(inputFilePath, outputFilePath, diemTypeObj):
    """
    Fast version of write_polarized_bed - streams data instead of loading everything into memory.
    
    Parameters:
    inputFilePath (str): Path to the original DiEM BED file.
    outputFilePath (str): Path to the output polarized DiEM BED file.
    diemTypeObj (DiemType): DiemType object containing polarity information.
    """
    
    if diemTypeObj.PolByChr is None:
        raise ValueError("DiemType object does not contain polarity information.")
    if inputFilePath is None or outputFilePath is None:
        raise ValueError("Input and output file paths must be provided.")
    if inputFilePath == outputFilePath:
        raise ValueError("Input and output file paths must be different to avoid overwriting.")

    # Prepare site masking data
    sitesMasked = [np.zeros(len(diemTypeObj.posByChr[i]), dtype=int) for i in range(len(diemTypeObj.chrNames))]
    
    if diemTypeObj.siteExclusionsByChr is not None:
        for i in range(len(diemTypeObj.chrNames)):
            if diemTypeObj.siteExclusionsByChr[i] is not None:
                sitesMasked[i][diemTypeObj.siteExclusionsByChr[i]] = 1
    
    # Flatten all the new column data
    nullPolarity_flat = np.hstack(diemTypeObj.initialPolByChr)
    polarity_flat = np.hstack(diemTypeObj.PolByChr)
    DI_flat = np.hstack(diemTypeObj.DIByChr)
    support_flat = np.hstack(diemTypeObj.SupportByChr)
    siteMasked_flat = np.hstack(sitesMasked)
    
    # Stream processing
    with open(inputFilePath, 'r') as f_in, open(outputFilePath, 'w') as f_out:
        
        # Write preamble
        if diemTypeObj.indExclusions is not None:
            masked_inds = '##IndividualsMasked=' + ','.join(diemTypeObj.indExclusions)
            f_out.write(masked_inds + '\n')
        else:
            f_out.write('##IndividualsMasked=None\n')
        
        # Process header line
        header = next(f_in).strip()
        new_header = header + '\tNullPolarity\tPolarity\tDiagnosticIndex\tSupport\tSiteMasked\n'
        f_out.write(new_header)
        
        # Process data lines
        row_idx = 0
        for line in f_in:
            line = line.strip()
            if line:  # Skip empty lines
                new_line = (f"{line}\t{nullPolarity_flat[row_idx]}\t{polarity_flat[row_idx]}\t"
                           f"{DI_flat[row_idx]}\t{support_flat[row_idx]}\t{siteMasked_flat[row_idx]}\n")
                f_out.write(new_line)
                row_idx += 1


def update_meta(metaFilePathIn, metaFilePathOut,ploidyFilePath=None, recFilePath = None):
    """

    Update metadata file with new ploidy and/or recombination rate information.

    Parameters:
        :param metaFilePathIn: input metadata file path (original metadata file)
        :param metaFilePathOut: output metadata file path (updated metadata file)
        :param ploidyFilePath: ploidy file path (optional)
        :param recFilePath: recombination rate file path (optional)

    Returns:
        None (writes updated metadata to output file)
    """
    if ploidyFilePath is None and recFilePath is None:
        raise ValueError("At least one of ploidyFilePath or recFilePath must be provided.")
    
    # Read metadata
    df_meta = pd.read_csv(metaFilePathIn, sep='\t')

    # Read ploidy data
    if ploidyFilePath is not None:

        print('udpating ploidy information from file: ' + ploidyFilePath)

        df_ploidy = pd.read_csv(ploidyFilePath, sep='\t', header=0)
        individual_col = df_ploidy.columns[0]  # Individual names column

        # Initialize tracking sets
        chromosomes_not_in_meta = set()
        individuals_not_in_meta = set()
        individuals_not_in_ploidy = set()
        
        # Get sets for comparison
        meta_chromosomes = set(df_meta['#Chrom'])
        meta_individuals = set(df_meta.columns[7:])
        ploidy_individuals = set(df_ploidy[individual_col])
        ploidy_chromosomes = set(df_ploidy.columns[1:])

        # Check for individuals not in ploidy file
        individuals_not_in_ploidy = meta_individuals - ploidy_individuals
        individuals_not_in_meta = ploidy_individuals - meta_individuals

        
        # For each chromosome column in the ploidy file (except the individual names column)
        for chrom_col in df_ploidy.columns[1:]:

            if chrom_col not in meta_chromosomes:
                chromosomes_not_in_meta.add(chrom_col)
                continue

            # Create ploidy dictionary for this chromosome
            ploidy_dict = dict(zip(df_ploidy[individual_col], df_ploidy[chrom_col]))
            
            # Find the row for this chromosome in metadata
            target_row_mask = df_meta['#Chrom'] == chrom_col
            
            # Update ploidy for each individual
            for individual in df_meta.columns[7:]:
                if individual in ploidy_dict:
                    df_meta.loc[target_row_mask, individual] = ploidy_dict[individual]
                # Note: individuals not in ploidy file are already tracked above
            
            print(f"Updated ploidy for {chrom_col}")

        # Report any issues found
        if chromosomes_not_in_meta:
            print(f"\nWarning: The following chromosomes from the ploidy file were not found in the metadata file: {sorted(chromosomes_not_in_meta)}")
            print("No ploidy updates were made for these chromosomes")
        if individuals_not_in_meta:
            print(f"\nWarning: The following individuals from the ploidy file were not found in the metadata file: {sorted(individuals_not_in_meta)}")
            print("No ploidy updates were made for these individuals")
        if individuals_not_in_ploidy:
            print(f"\nWarning: The following individuals from the metadata file were not found in the ploidy file: {sorted(individuals_not_in_ploidy)}")
            print("The ploidy for these individuals remains diploid for all chromosomes (the default value)")
        # Summary
        if not (chromosomes_not_in_meta or individuals_not_in_meta or individuals_not_in_ploidy):
            print("\nAll chromosomes and individuals matched successfully.")
        else:
            print("\nPloidy update completed with warnings as noted above. Please review your data before proceeding!!!")
 
        print('done updating ploidy information\n')
    print()


    if recFilePath is not None:

        print('udpating recombination rate information from file: ' + recFilePath+'\n')

        # Read recombination rate data
        df_rec = pd.read_csv(recFilePath, sep='\t', header=0)
        rec_dict = dict(zip(df_rec['chromosome'], df_rec['relative_rate']))
        

        
        notInRecFile = set(df_meta['#Chrom']) - set(rec_dict.keys())
        if notInRecFile:
            print(f"Note: The following chromosomes from metadata were not found in the recombination rate file and were not updated:\n {sorted(notInRecFile)}")

        notInMetaFile = set(rec_dict.keys()) - set(df_meta['#Chrom'])
        if notInMetaFile:
            print(f"Note: The following chromosomes from the recombination rate file were not found in the metadata file:\n {sorted(notInMetaFile)}")

        print('\nthe following chromosomes were updated with new recombination rates:')
        # Update relativeRecRates in metadata
        for chrom in df_meta['#Chrom']:
            if chrom in rec_dict:
                print(f"\t{chrom}: {rec_dict[chrom]}")
                df_meta.loc[df_meta['#Chrom'] == chrom, 'relativeRecRates'] = rec_dict[chrom]

    # Write updated metadata to output file
    df_meta.to_csv(metaFilePathOut, sep='\t', index=False)
    print(f"\n done updating recobination rates")

    print(f"\nUpdated metadata saved to {metaFilePathOut}")




    # Alternative version if you want to handle multiple chromosomes
def update_ploidy(ploidyFilePath, metaFilePathIn, metaFilePathOut):
    """
    Update ploidy information for multiple chromosomes.
    Expects ploidy file format: individual_name \t chrom1_ploidy \t chrom2_ploidy \t ...
    this returns a 'Corrected' metadatafile with the user-provided ploidy values.
    """
    
    # Read ploidy data
    df_ploidy = pd.read_csv(ploidyFilePath, sep='\t', header=0)
    individual_col = df_ploidy.columns[0]  # Individual names column
    
    # Read metadata
    df_meta = pd.read_csv(metaFilePathIn, sep='\t')

    # Initialize tracking sets
    chromosomes_not_in_meta = set()
    individuals_not_in_meta = set()
    individuals_not_in_ploidy = set()
    
    # Get sets for comparison
    meta_chromosomes = set(df_meta['#Chrom'])
    meta_individuals = set(df_meta.columns[7:])
    ploidy_individuals = set(df_ploidy[individual_col])
    ploidy_chromosomes = set(df_ploidy.columns[1:])

    # Check for individuals not in ploidy file
    individuals_not_in_ploidy = meta_individuals - ploidy_individuals
    individuals_not_in_meta = ploidy_individuals - meta_individuals

    
    # For each chromosome column in the ploidy file (except the individual names column)
    for chrom_col in df_ploidy.columns[1:]:

        if chrom_col not in meta_chromosomes:
            chromosomes_not_in_meta.add(chrom_col)
            continue

        # Create ploidy dictionary for this chromosome
        ploidy_dict = dict(zip(df_ploidy[individual_col], df_ploidy[chrom_col]))
        
        # Find the row for this chromosome in metadata
        target_row_mask = df_meta['#Chrom'] == chrom_col
        
        # Update ploidy for each individual
        for individual in df_meta.columns[7:]:
            if individual in ploidy_dict:
                df_meta.loc[target_row_mask, individual] = ploidy_dict[individual]
            # Note: individuals not in ploidy file are already tracked above
        
        print(f"Updated ploidy for {chrom_col}")

    # Report any issues found
    if chromosomes_not_in_meta:
        print(f"\nWarning: The following chromosomes from the ploidy file were not found in the metadata file: {sorted(chromosomes_not_in_meta)}")
        print("No ploidy updates were made for these chromosomes")
    if individuals_not_in_meta:
        print(f"\nWarning: The following individuals from the ploidy file were not found in the metadata file: {sorted(individuals_not_in_meta)}")
        print("No ploidy updates were made for these individuals")
    if individuals_not_in_ploidy:
        print(f"\nWarning: The following individuals from the metadata file were not found in the ploidy file: {sorted(individuals_not_in_ploidy)}")
        print("The ploidy for these individuals remains diploid for all chromosomes (the default value)")
    # Summary
    if not (chromosomes_not_in_meta or individuals_not_in_meta or individuals_not_in_ploidy):
        print("\nAll chromosomes and individuals matched successfully.")
    else:
        print("\nPloidy update completed with warnings as noted above. Please review your data before proceeding!!!")
    # Write updated metadata to output file
    df_meta.to_csv(metaFilePathOut, sep='\t', index=False)
    print(f"Updated metadata saved to {metaFilePathOut}")