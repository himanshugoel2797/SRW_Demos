
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _OFFLOAD_GPU
#define _USE_CUDA
#define CUDA_BLOCK_SIZE_MAX 16
#define CUDA_GRID_SIZE_MAX 16


typedef void(*stokesFun)(float*, float*, float*, float*, float*, double);

template <int BLOCK_SIZE> __global__ void test_GPU(int *id)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Nx = blockDim.x;
	int Ny = blockDim.y;
	int idx = Nx*BLOCK_SIZE*BLOCK_SIZE*by+BLOCK_SIZE*BLOCK_SIZE*bx+BLOCK_SIZE*ty+tx;
	id[idx] = idx;
}

// mo = v1' v1 + v2' v2
// nxy: vector size in complex
template <int BLOCK_SIZE, stokesFun fun> __global__ void addVecOProdCUDA(float *mo, float *v1, float *v2, long nxy, double iter, long long tot_iter){
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Block dimension
    int Nx = blockDim.x;
    int Ny = blockDim.y;
    long vbsize = nxy/Nx;
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;	
    
    long bId = (Nx*by+bx);

    long offset_y = (long)((-1.+sqrt(1.+8.*bId))/2.0);
    if(bId==0) offset_y=0;
    long offset_x = bId - (long)((offset_y*(offset_y+1)+1)/2);
    while(offset_x < 0) { offset_y -= 1; offset_x = bId - (long)((offset_y*(offset_y+1)+1)/2); }
    while(offset_x > offset_y) { offset_y += 1; offset_x = bId - (long)((offset_y*(offset_y+1)+1)/2); }
    if(offset_y >= Ny || offset_y < 0) return;
    if(offset_x > offset_y || offset_x < 0) return;

    float *vb10 = v1 + 2*offset_x*vbsize; 
    float *vb1p0 = v1 + 2*offset_y*vbsize; 
    float *vb20 = v2 + 2*offset_x*vbsize; 
    float *vb2p0 = v2 + 2*offset_y*vbsize; 

    float *mb = mo + vbsize*((offset_y*vbsize+1)*offset_y+offset_x*2);

    __shared__ float V1p[2*BLOCK_SIZE], V2p[2*BLOCK_SIZE], V1[2*BLOCK_SIZE], V2[2*BLOCK_SIZE]; 

    for(long long n=0; n<tot_iter; n++)
    {
		float *vb1 = vb10 + 2*n*nxy;
		float *vb1p = vb1p0 + 2*n*nxy;
		float *vb2 = vb20 + 2*n*nxy;
		float *vb2p = vb2p0 + 2*n*nxy;
		for(long i=0; i<vbsize; i+=BLOCK_SIZE) 
		{
			if(true){ //tx==0
				V1p[2*ty] = vb1p[(i+ty)*2]; V1p[2*ty+1] = -vb1p[(i+ty)*2+1];
				V2p[2*ty] = vb2p[(i+ty)*2]; V2p[2*ty+1] = -vb2p[(i+ty)*2+1];
			}
			long jsize = vbsize;
			if(offset_x==offset_y) jsize=i+1;
			for(long j=0; j<jsize; j+=BLOCK_SIZE)
			{
				float *MB = 0x0;
				if((offset_x<offset_y) || (j+tx<=i+ty)) 
				{
					MB = mb + (2*offset_y*vbsize+i+ty+1)*(i+ty) + (j+tx)*2;
					if(true){//ty==0
						V1[2*tx] = vb1[(j+tx)*2]; V1[2*tx+1] = vb1[(j+tx)*2+1];
						V2[2*tx] = vb2[(j+tx)*2]; V2[2*tx+1] = vb2[(j+tx)*2+1];
					}
				}
				__syncthreads();   
				
				if((offset_x<offset_y) || (j+tx<=i+ty))
				{
					(*fun)(MB, V1+2*tx, V1p+2*ty, V2+2*tx, V2p+2*ty, iter);
#if 0
					MB[0] = (MB[0]*iter + V1p[2*ty]*V1[2*tx]-V1p[2*ty+1]*V1[2*tx+1] + V2p[2*ty]*V2[2*tx]-V2p[2*ty+1]*V2[2*tx+1])/(iter+1.);
					MB[1] = (MB[1]*iter + V1p[2*ty]*V1[2*tx+1]+V1p[2*ty+1]*V1[2*tx] + V2p[2*ty]*V2[2*tx+1]+V2p[2*ty+1]*V2[2*tx])/(iter+1.);
#endif	    
				}
				__syncthreads();
			}
		}
		__syncthreads();
		iter += 1.0;
    }
}


// s0
__global__ void StokesS0(float *mo, float *v1, float *v1p, float *v2, float *v2p, double iter){
	mo[0] = (mo[0]*iter + v1p[0]*v1[0]-v1p[1]*v1[1] + v2p[0]*v2[0]-v2p[1]*v2[1])/(iter+1.);
	mo[1] = (mo[1]*iter + v1p[0]*v1[1]+v1p[1]*v1[0] + v2p[0]*v2[1]+v2p[1]*v2[0])/(iter+1.);
}

// s1
__global__ void StokesS1(float *mo, float *v1, float *v1p, float *v2, float *v2p, double iter){
	mo[0] = (mo[0]*iter + v1p[0]*v1[0]-v1p[1]*v1[1] - v2p[0]*v2[0]+v2p[1]*v2[1])/(iter+1.);
	mo[1] = (mo[1]*iter + v1p[0]*v1[1]+v1p[1]*v1[0] - v2p[0]*v2[1]-v2p[1]*v2[0])/(iter+1.);
}

// Linear Horizontal Pol.
__global__ void StokesLH(float *mo, float *v1, float *v1p, float *v2, float *v2p, double iter){
	mo[0] = (mo[0]*iter + v1p[0]*v1[0]-v1p[1]*v1[1])/(iter+1.);
	mo[1] = (mo[1]*iter + v1p[0]*v1[1]+v1p[1]*v1[0])/(iter+1.);
}

// Linear Vertical Pol.
__global__ void StokesLV(float *mo, float *v1, float *v1p, float *v2, float *v2p, double iter){
	mo[0] = (mo[0]*iter + v2p[0]*v2[0]-v2p[1]*v2[1])/(iter+1.);
	mo[1] = (mo[1]*iter + v2p[0]*v2[1]+v2p[1]*v2[0])/(iter+1.);
}


void MutualIntensityComponentCUDA_RLi(float *mo, float *v1, float *v2, long nxy, double iter, long long tot_iter, int PloCom){
	int block_size=CUDA_BLOCK_SIZE_MAX, grid_size;
        grid_size = (int)(nxy/block_size);
	if(grid_size*block_size!=nxy){
		printf("MutualIntensityComponentCUDA error: Unsupported Wavefront mesh size %d\nHave to be devidable by %d\nChange either the mesh size or CUDA_BLOCK_SIZE_MAX (make sure it's supported by the CUDA device) and restart\n", nxy, block_size);
		return;
	}
        if(grid_size>CUDA_GRID_SIZE_MAX) grid_size = CUDA_GRID_SIZE_MAX;
        dim3 threads(block_size, block_size);
        dim3 grid(grid_size, grid_size);
	if(block_size==16) 
	switch(PloCom){
		case 0: // Linear H. Pol
			addVecOProdCUDA<16,&StokesLH><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
			break;
		case 1: // Linear V. Pol
			addVecOProdCUDA<16,&StokesLV><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
			break;
		case -1: // s0
			addVecOProdCUDA<16,&StokesS0><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
			break;
		case -2: // s1
			addVecOProdCUDA<16,&StokesS1><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
			break;
		default:
			addVecOProdCUDA<16,&StokesS0><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
	}
        if(block_size==8)
        switch(PloCom){
                case 0: // Linear H. Pol
                        addVecOProdCUDA<8,&StokesLH><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
                        break;
                case 1: // Linear V. Pol
                        addVecOProdCUDA<8,&StokesLV><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
                        break;
                case -1: // s0
                        addVecOProdCUDA<8,&StokesS0><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
                        break;
                case -2: // s1
                        addVecOProdCUDA<8,&StokesS1><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
                        break;
                default:
                        addVecOProdCUDA<8,&StokesS0><<<grid, threads,0>>>(mo, v1, v2, nxy, iter, tot_iter);
        }
}
