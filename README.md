# cuda-gen

<img src="https://github.com/VaradRengaraj/cuda-gen/blob/main/utils/image.png" width="500">

This project is a small offshoot from Gen_PES_Pred(https://github.com/VaradRengaraj/Gen_PES_Pred). In Gen_PES_Pred, as part of the dataset generation process, a large file containing the coordinate frames of the atoms is parsed and for each frame, a coulomb matrix[[1]](#1) is generated. This coulomb matrix is then sparsified and the resultant sparse coulomb matrix is divided into several dense smaller submatrices[[2]](#2). The eigenvalues for the submatrices are calculated and each of these eigenvalues vectors are stored in an HDF5 file. A single-threaded, serially running code executing this data generation process is time-consuming and hence a multithreaded Cuda-based approach is explored. 

### Kernels
coulomb      -- generates coulomb matrix<br/>
submatrix    -- sparse coulomb matrix is divided to smaller submatrices<br/>
jacobi-eigen -- calculates eigen values for a symmetric matrix 

## References
<a id="1">[1]</a> 
https://singroup.github.io/dscribe/0.3.x/tutorials/coulomb_matrix.html.<br/>
<a id="2">[2]</a> 
Lass, M.; Mohr, S.; Wiebeler, H.; Kühne, T.D.; Plessl, C.
"A Massively Parallel Algorithm for the Approximate Calculation of Inverse P-th Roots of Large Sparse Matrices".
In Proceedings of the Platform for Advanced Scientific Computing Conference, pp. 1-11. 2018.




