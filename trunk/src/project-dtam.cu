/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <string>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "CTensor.h"

using std::string;

#define BLOCK_SIZE 16
#define RADIUS     1

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

// kernel that does nothing
__global__ void dummy_kernel() { }


// kernel to do some simple stuff on an image
__global__ void test_kernel(float* in, float* out, int img_xsize, int img_ysize)
{
	__shared__ float shared_block[BLOCK_SIZE+2*RADIUS][BLOCK_SIZE+2*RADIUS];
	int xindex = blockDim.x * blockIdx.x + threadIdx.x;
	int yindex = blockDim.y * blockIdx.y + threadIdx.y;
	if (xindex >= img_xsize || yindex >= img_ysize) return;

	// load shared data
	shared_block[threadIdx.x+RADIUS][threadIdx.y+RADIUS] = in[xindex+yindex*img_xsize];
	// edge/corner cases, distributed such that no thread does more than 3 operations
	if (threadIdx.x < RADIUS) {
		shared_block[RADIUS-threadIdx.x-1][threadIdx.y+RADIUS] = shared_block[threadIdx.x+RADIUS][threadIdx.y+RADIUS];
		if (threadIdx.y < RADIUS) {
			shared_block[RADIUS-threadIdx.x-1][RADIUS-threadIdx.y-1] = shared_block[threadIdx.x+RADIUS][threadIdx.y+RADIUS];
		}
	}
	if (threadIdx.y < RADIUS) {
		shared_block[threadIdx.x+RADIUS][RADIUS-threadIdx.y-1] = shared_block[threadIdx.x+RADIUS][threadIdx.y+RADIUS];
		if (threadIdx.x >= BLOCK_SIZE-RADIUS) {
			shared_block[2*BLOCK_SIZE-threadIdx.x+RADIUS-1][RADIUS-threadIdx.y-1] = shared_block[threadIdx.x+RADIUS][threadIdx.y+RADIUS];
		}
	}
	if (threadIdx.x >= BLOCK_SIZE-RADIUS) {
		shared_block[2*BLOCK_SIZE-threadIdx.x+RADIUS-1][threadIdx.y+RADIUS] = shared_block[threadIdx.x+RADIUS][threadIdx.y+RADIUS];
		if (threadIdx.y >= BLOCK_SIZE-RADIUS) {
			shared_block[2*BLOCK_SIZE-threadIdx.x+RADIUS-1][2*BLOCK_SIZE-threadIdx.y+RADIUS-1] = shared_block[threadIdx.x+RADIUS][threadIdx.y+RADIUS];
		}
	}
	if (threadIdx.y >= BLOCK_SIZE-RADIUS) {
		shared_block[threadIdx.x+RADIUS][2*BLOCK_SIZE-threadIdx.y+RADIUS-1] = shared_block[threadIdx.x+RADIUS][threadIdx.y+RADIUS];
		if (threadIdx.x < RADIUS) {
			shared_block[RADIUS-threadIdx.x-1][2*BLOCK_SIZE-threadIdx.y+RADIUS-1] = shared_block[threadIdx.x+RADIUS][threadIdx.y+RADIUS];
		}
	}
	__syncthreads();

	// set pixel to average of RADIUS-neighborhood (anisotropic: box)
	float average = 0;
	for (int y = -RADIUS; y <= RADIUS; y++) {
		for (int x = -RADIUS; x <= RADIUS; x++) {
			if (x == 0 && y == 0) continue;
			average += shared_block[x+threadIdx.x+RADIUS][y+threadIdx.y+RADIUS] / ((2*RADIUS+1)*(2*RADIUS+1)-1);
		}
	}

	out[xindex+yindex*img_xsize] = average;

//	// CUDA 2.0+ (Fermi+) supports printf in __global__ (NOT ordered!)
//	if (threadIdx.x == 0 && threadIdx.y == 0)
//	{
//		printf("Block (%d,%d),\tpixels (%d,%d) \tto (%d,%d).\n",
//				blockIdx.x, blockIdx.y,
//				blockDim.x*blockIdx.x, blockDim.y*blockIdx.y,
//				blockDim.x*(blockIdx.x+1)-1, blockDim.y*(blockIdx.y+1)-1);
//	}
}


/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	string filename = "/home/nikolaus/cuda-workspace/project-dtam/images/lenaNoisy10.pgm";
	CTensor<float>* in_img = new CTensor<float>();
	in_img->readFromPGM(filename.c_str());

	int img_size = in_img->xSize() * in_img->ySize();
	int img_mem_size = sizeof(float) * img_size * in_img->zSize();
	float* mdata = in_img->data();

	clock_t start = clock();
	clock_t diff;

	float *d_in, *d_out;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_in, img_mem_size));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_out, img_mem_size));

	CUDA_CHECK_RETURN(cudaMemcpy(d_in, mdata, img_mem_size, cudaMemcpyHostToDevice));

	// invoke kernel:
	// 16x16 pixels per block
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((in_img->xSize()+BLOCK_SIZE-1)/BLOCK_SIZE, (in_img->ySize()+BLOCK_SIZE-1)/BLOCK_SIZE);

	// "dummy" kernel call to establish primary context
	dummy_kernel<<<1,1>>>();

	diff = clock() - start;
	int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Init took %d.%d seconds\n", msec/1000, msec%1000);


	for( int i = 0; i < 100; i++ )
		test_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, in_img->xSize(), in_img->ySize());

	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	diff = clock() - start;
	msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Kernel calls took %d.%d seconds\n", msec/1000, msec%1000);

	// did everything go OK?
	CUDA_CHECK_RETURN(cudaGetLastError());
	// retrieve result from GPU
	CUDA_CHECK_RETURN(cudaMemcpy(mdata, d_out, img_mem_size, cudaMemcpyDeviceToHost));

	string outfilename = "/home/nikolaus/cuda-workspace/project-dtam/images/out.pgm";
	in_img->writeToPGM(outfilename.c_str());

	// tidy up
	CUDA_CHECK_RETURN(cudaFree(d_in));
	CUDA_CHECK_RETURN(cudaFree(d_out));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}
