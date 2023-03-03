# SDCP-Calculation
A small CUDA script performing the same calculation as a provided python script, the calculation was used for the PhD thesis of a student at Aarhus University.
The problems resolved were 2-fold:

1. **Numerical Accuracy**: The program involved summing billions of small numbers into 1 accumulator, thus rounding error was a massive issue. In CUDA this problem was solved in 3 ways: Reduction, Small-numbers first, and Kahan Summation.
2. **Performance**: The python program took ~5hrs to compute the sum for a single file (of which there were 160), the CUDA script executed this in ~5 minutes (RTX 2080ti) for all 160 files. ~5000x speedup.  
