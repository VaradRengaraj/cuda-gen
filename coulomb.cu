#include <stdio.h>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdlib.h>
#include <iostream>

__global__ void coulombMatrix(double *pos, double *col, int *chargeptr, int nx, int ny, int cutoff, double bc)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*nx + ix;
    double rd;

    if(ix < nx && iy < ny)
    {
        if(ix == iy){
            *(col+idx) = 0.5*pow(chargeptr[ix], 2.4);
        }
        else if(ix > iy){
            rd = pow(pow(*(pos+(iy*3))-*(pos+(ix*3))-bc*round((*(pos+(iy*3))-*(pos+(ix*3)))/bc),2)+
                pow(*(pos+(iy*3)+1)-*(pos+(ix*3)+1)-bc*round((*(pos+(iy*3)+1)-*(pos+(ix*3)+1))/bc),2)+
                pow(*(pos+(iy*3)+2)-*(pos+(ix*3)+2)-bc*round((*(pos+(iy*3)+2)-*(pos+(ix*3)+2))/bc),2),0.5);
            //printf("rd %lf\n", rd);
            if(rd >= cutoff){
                *(col+idx) = 0;
            }
            else{
                *(col+idx) = (chargeptr[ix]*chargeptr[iy])/rd;
            }
        }
        else{}
    }
}

__global__ void coulombMatrixLT(double *col, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*nx + ix;
    unsigned int idr = ix*nx + iy;

    if(ix < nx && iy < ny)
    {
        if(ix < iy)
        {
            *(col+idx) = *(col+idr);
        }
    }
}

