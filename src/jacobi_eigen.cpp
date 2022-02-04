#include <stdio.h>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdlib.h>
#include <iostream>

__global__ void jacobi(double *arr_ptr, int *pair_arr, int n, int *cont, double tolerance)
{
    unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
    double c, s;
    unsigned int k = 0;
    unsigned int i, j;
    unsigned int p, q, temp;
    unsigned int te = (n % 2 ? n+1 : n);

    //printf("\n te %d n %d", te, n);

    while(*cont != 0)
    {
        if(id == 0)
        {
            k += 1;
            //printf("k val %d", k);
        }
        __syncthreads();

        //for(in = 0; in < n/2; in++)
        //{
            p = *(pair_arr + id);
            q = *(pair_arr + id + te/2);

            //printf("\n p & q %d %d \n", p, q);

            if(p != 999 && q != 999){
            if(p > q)
            {
                temp = q;
                q = p;
                p = temp;
            }

            if(*(arr_ptr + n*p +q) != 0)
            {
                double torque, t;
                torque = ( *(arr_ptr + q * n + q) - *(arr_ptr + p * n + p))/(2*(*(arr_ptr + p * n + q)));
                if (torque >= 0)
                    t = 1/(torque + sqrt(1+torque*torque));
                else
                    t = -1/(-torque + sqrt(1+torque*torque));

                c = 1/sqrt(1+t*t);
                s = t*c;
            }
            else
            {
                c = 1;
                s = 0;
            }

            /* A = transpose(J)*A*J */
            for (i = 0; i < n; i++)
            {
                double Api = (*(arr_ptr + p * n + i))*c + (*(arr_ptr + q * n + i))*(-s);
                double Aqi = (*(arr_ptr + p * n + i))*s + (*(arr_ptr + q * n + i))*c;
                //__syncthreads();
                *(arr_ptr + p * n + i) = Api;
                *(arr_ptr + q * n + i) = Aqi;
            }

            for (i = 0; i < n; i++)
            {
                double Aip = (*(arr_ptr + i * n + p))*c + (*(arr_ptr + i * n + q))*(-s);
                double Aiq = (*(arr_ptr + i * n + p))*s + (*(arr_ptr + i * n + q))*c;
                //__syncthreads();
                *(arr_ptr + i * n + p) = Aip;
                *(arr_ptr + i * n + q) = Aiq;
            }

            //for(i = 0; i < n; i++)
            //{
            //    for(j = 0; j < n; j++)
            //    {
                   // if(i != j)
            //{
            //         printf("  %lf", *(arr_ptr + i*n + j));
            //}
            //    }
            //    printf(" \n");
            //}

            }

        //}
        __syncthreads();

        // chess reordering
        if(id == 0)
        {
            //unsigned int te = (n % 2 ? n-1 : n);
            //unsigned int temp;
            temp = *(pair_arr + te/2 - 1);

            for(i = te/2-1; i > 1; i--)
            {
                *(pair_arr + i) = *(pair_arr + i - 1);
            }

            *(pair_arr + 1) = *(pair_arr + te/2);

            for(i = te/2; i < te-1; i++)
            {
                *(pair_arr + i) = *(pair_arr + i + 1);
            }

            *(pair_arr + te - 1) = temp;

            //printf("\n k val %d", k);
           if(k == te)
           {
               double val = 0;
               for(i = 0; i < n; i++)
               {
                   for(j = 0; j < n; j++)
                   {
                       if(i != j)
                       {
                           val += pow(*(arr_ptr + n*i + j), 2);
                       }
                   }
               }
               //printf("sqrt(val) %lf", sqrt(val));
               if(sqrt(val) <= tolerance)
                   *cont = 0;
               else
                   *cont = 1;
               k = 0;
           }
           //break;
        }
        __syncthreads();
    }
}
