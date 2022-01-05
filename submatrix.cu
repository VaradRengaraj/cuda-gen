#include <stdio.h>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdlib.h>
#include <iostream>

#define NR 81

//#define NR 9
__device__ double* dataptr[NR];
__global__ void submatrix(double *col, int nx, int *submatsizes)
{
    unsigned int local[NR] = {0,};
    unsigned int i, j;
    unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int k,l;
    double *buf;
    k=0;
    l=0;

    if(id < nx)
    {

    for(i=id; i<=nx*nx; i+=nx)
    {
        if(*(col+i) != 0){
            local[l] = k+1;
            l++;
        }
        k++;
    }

   // allocate memory for the submatrix based on the size
    for(i=0; i<NR; i++)
    {
        if(local[i] == 0)
            break;
    }

    k=i;
    *(submatsizes+id) = k;
    //printf("\n k--id %d %d", k, id);
    buf = (double *)malloc(k*k*sizeof(double));
    l=0;

    for(i=0; i<k; i++)
    {
        for(j=0; j<k; j++)
        {
            *(buf+l) = *(col+(local[i]-1)*nx+(local[j]-1));
            l++;
        }
    }

    dataptr[id] = buf;
    __syncthreads();
    }
}


#if 1
__global__ void copysubmat(double *subm, int N, int num)
{
    unsigned int i;
    unsigned int id = num;//blockIdx.x*blockDim.x+threadIdx.x;
    double *bufptr = dataptr[id];

    for(i=0; i<N*N; i++)
        subm[i] = bufptr[i];
}
#endif


