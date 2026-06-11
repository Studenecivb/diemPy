# API Reference

This is the API

```{eval-rst}
.. currentmodule:: diempy
```

## diemtype

```{eval-rst}
.. automodule:: diempy.diemtype
    :members: save_DiemType, load_DiemType, flip_polarity
    :exclude-members: DiemType

.. autoclass:: diempy.diemtype.DiemType
   :members: sort,copy,polarize,apply_threshold,smooth,create_contig_matrix,get_intervals_of_state,intervals_to_bed
```


## contigs and intervals
```{eval-rst}
.. automodule:: diempy.contigs
    :members: Interval, Contig, export_contigs_to_ind_bed_files
```

<!--
## polarization submodule
```{eval-rst}
.. automodule:: diempy.polarize
    :members:
```
-->

<!--
## smoothing submodule
```{eval-rst}
.. automodule:: diempy.smooth
    :members:
```
-->

<!--
## tests submodule
```{eval-rst}
.. automodule:: diempy.tests
    :members:
```
-->

## plots submodule
```{eval-rst}
.. automodule:: diempy.plots
    :members: GenomeSummaryPlot, GenomeMultiSummaryPlot,GenomicDeFinettiPlot, GenomicMultiDeFinettiPlot, GenomicContributionsPlot, diemPairsPlot, diemMultiPairsPlot, diemPlotPrepFromDiemType, diemPlotPrepFromBedMeta, diemIrisFromDiemType, diemIrisFromPlotPrep, diemLongFromDiemType, diemLongFromPlotPrep
```

This is the end of the API documentation
