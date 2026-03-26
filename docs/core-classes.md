# Core Classes and Functions

There are three main classes in Diem.  Namely, 
- **DiemType** which is the main data structure that holes the core functions for polarizing the data
- **Contig** class which simply holds the sequence of intervals that form a contig
- **Intervals** a class which describes a single interval, i.e. a contiguous region of a single state in a given individual 

**DiemType** → **Contig** → **Interval**: DiemType contains the diem-formatted data as well as a matrix of Contigs (one per individual per chromosome), each Contig contains a list of Intervals representing contiguous ancestry tracts.

## DiemType Class
The central data structure that holds:
- **DMBC**: Diem Matrix By Chromosome (state matrices)
- **Genomic positions** and chromosome information
- **Individual metadata** including ploidy and exclusions
- **Polarization results** including polarity, diagnostic indices, and support values
- **Contigs** which are per-chromosome per-interval lists of tracts 

Key methods:
- [`polarize()`](src/diempy/diemtype.py): Run polarization analysis
- [`apply_threshold()`](src/diempy/diemtype.py): Filter markers by diagnostic index
- [`smooth()`](src/diempy/diemtype.py): Apply kernel smoothing
- [`sort()`](src/diempy/diemtype.py): Sort individuals by hybrid index
- ['create_contig_matrix()](src/diempy/diemtype.py): builds the contigs from the processed data

## Contig Class
The [`Contig`](src/diempy/contigs.py) class represents a collection of genomic intervals for a specific individual and chromosome, essentially describing the complete ancestry structure along that chromosome.

Key attributes:
- **chrName**: Chromosome name
- **indName**: Individual name
- **intervals**: List of Interval objects making up the contig
- **num_intervals**: Number of intervals in the contig

Key methods:
- [`printIntervals()`](src/diempy/contigs.py): Display interval information in a readable format
- [`get_my_intervals_of_state()`](src/diempy/contigs.py): Filter intervals by ancestry state

## Interval Class
The [`Interval`](src/diempy/contigs.py) class represents a contiguous genomic region with consistent ancestry state for a specific individual and chromosome.

Key attributes:
- **chrName**: Chromosome name
- **indName**: Individual name  
- **idxl, idxr**: Left and right indices (inclusive) in the state matrix
- **l, r**: Left and right physical positions
- **state**: Ancestry state of the interval (0=uncalled, 1-3=called states)

Key methods:
- [`span()`](src/diempy/contigs.py): Calculate and return the physical span of the interval
- [`mapSpan(chrLength)`](src/diempy/contigs.py): Calculate the relative span as fraction of chromosome length. chrLength must be provided
