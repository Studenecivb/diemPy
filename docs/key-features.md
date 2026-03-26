### Key Features
- **VCF Processing**: Direct conversion from VCF files to diem format
- **Genome Polarization**: Automated polarization of genetic markers using EM algorithm
- **Thresholding**: Tools to help decide on threshold for minium diagnostic index of markers to retain
- **Kernel Smoothing**: Spatial smoothing of genomic data along chromosomes
- **Obtaining Tract Length**: Detection and analysis of genomic regions with consistent ancestry
- **Parallel Processing**: Multi-core support for computationally intensive operations
- **Flexible I/O**: Support for various input/output formats including BED-like files

### Core Functionality

The package provides several main analysis workflows:

1. **Data Import and Processing**
   - Convert VCF files to diem format using [`vcf2diem`](src/diempy/vcf2diembeds.py)
   - Read diem BED format files with [`read_diem_bed`](src/diempy/io.py)
   - Handle masking of individuals and sites
   - Handling correct ploidy information, e.g. with regard to sex chromosomes

2. **Polarization Analysis**
   - Initialize polarization using random null
   - Run EM algorithm to optimize marker polarization
   - Calculate diagnostic indices and support values
   - Parallel and linear processing options available

3. **Post-Polarization Analysis**
   - Compute hybrid indices for individuals
   - Apply thresholding to filter less informative markers
   - Perform kernel smoothing across genomic windows
   - Generate tracts of contiguous ancestry and store them as contigs


