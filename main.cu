#include "readwriteImg.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "blur.h"

using namespace std;

#define VERIFY(err)				\
do 								\
{ 								\
	if (err != cudaSuccess)		\
    {							\
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));\
        exit(EXIT_FAILURE); 	\
    }							\
}while(0)


__global__ void BlurImage(const unsigned char* image,const int imgWidth,const int imgHeight,
						  const unsigned char component, unsigned char* outImage);


void LauchKernel(const unsigned char* imgData, const int& width, const int& height, 
				 const int& component,unsigned char* bluredImg)
{
	unsigned char* devImgDataIn;
	int realInputSize = (width+1)*(height+1)*3;
	int padingSize = (realInputSize+((1<<10)-1)) &(~((1<<10)-1)); //pad empty mem to make align with 1024
	cudaMalloc((void**)&devImgDataIn,padingSize);


	unsigned char* devImgDataOut;
	int realOutputSize = (width)*(height)*3;
	cudaMalloc((void**)&devImgDataOut,realOutputSize);

	cudaMemcpy(devImgDataIn,imgData,realInputSize,cudaMemcpyHostToDevice);
	//printf("LauchKernel\n");

	dim3 thread(1024,1,1);
	int blockNum = realOutputSize/1024;
	dim3 block(blockNum/1024,1024,1);

	BlurImage<<<block,thread>>>(devImgDataIn,width,height,component,devImgDataOut);
	
	cudaMemcpy(bluredImg,devImgDataOut,realOutputSize,cudaMemcpyDeviceToHost);

	cudaFree(devImgDataIn);
	cudaFree(devImgDataOut);
}

void CPUBlur(unsigned char* imgData,unsigned char* out,int size,int imgWidth,int imgHeight,int component)
{
    for(int y = 0; y<imgHeight; y++)
    {
    	for(int x=-KERNEL_RADIUS; x< imgWidth+KERNEL_RADIUS; x++)
    	{
    		float sumRed = 0;
    		float sumGreen = 0;
    		float sumBlue = 0;
    		for(int i=x;i<x+KERNEL_LENGTH;i++)
    		{
    			if(i<0 || i>=imgWidth)
    			{
    				sumRed += 0;
    				sumGreen += 0;
    				sumBlue += 0;
    			}
    			else
    			{
    				sumRed 	 += imgData[y*imgWidth*3 + i*3 ];
    				sumGreen += imgData[y*imgWidth*3 + i*3 + 1];
    				sumBlue  += imgData[y*imgWidth*3 + i*3 + 2];
    			}
    		}
    		out[3*(x-KERNEL_RADIUS)]   = sumRed/KERNEL_LENGTH;
			out[3*(x-KERNEL_RADIUS)+1] = sumGreen/KERNEL_LENGTH;
			out[3*(x-KERNEL_RADIUS)+2] = sumBlue/KERNEL_LENGTH;
    	}
    }
}
void ConstructRowKernel(unsigned char* imageInput,const unsigned int& width,
						const unsigned int& height,unsigned char* imageOutput)
{
	dim3 blocks(width/(ROW_PROC_PIXELS_EACH_THREAD*ROW_BLOCK_DIM_X),height/ROW_BLOCK_DIM_Y);
	dim3 threads(ROW_BLOCK_DIM_X,ROW_BLOCK_DIM_Y);

	BlurRow<<<blocks,threads>>>(imageInput,2048,2048,imageOutput);
	cudaDeviceSynchronize();
}
void OutputSampleTest(unsigned char* output)
{
	if(output == NULL)
		return;

	for(int i=0;i<200;i++)
	{
		printf("%d\n", output[i]);
	}
}
int main(int argc,char* argv[])
{

	unsigned char* data;
	int size = read_JPEG_file(argv[1],&data);
	//cudaDeviceReset();
	//printf("image size = %d\n", size);

	cudaError_t err = cudaSuccess;

	unsigned char* devInput;
	err = cudaMalloc((void**)&devInput,size);
	VERIFY(err);

	err = cudaMemcpy(devInput, data, size, cudaMemcpyHostToDevice);
	VERIFY(err);

	unsigned char* devOutput;
	err = cudaMalloc((void**)&devOutput,size);
	VERIFY(err);

	//warm up
	BlurRow<<<1,1>>>(devInput,2048,2048,devOutput);
	cudaDeviceSynchronize();
	
	StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    //TestKernel<<<1,1>>>(devInput);
	ConstructRowKernel(devInput,2048,2048,devOutput);
	// cudaMemcpy(devInput,devOutput,size,cudaMemcpyDeviceToDevice);
	// ConstructRowKernel(devInput,2048,2048,devOutput);
	unsigned char* bluredImg = new unsigned char[2048*2048*3];
	//memset(bluredImg,size,0);
	//CPUBlur(data,bluredImg,size,2048,2048,3);
	//LauchKernel(data,2048,2048,3,bluredImg);

	sdkStopTimer(&hTimer);
    double timer = 0.001 * sdkGetTimerValue(&hTimer);
    printf("Time = %.5f s\n\n", timer);

    
    cudaMemcpy(bluredImg,devOutput,size,cudaMemcpyDeviceToHost);
    cudaFree(devInput);
    cudaFree(devOutput);
    //OutputSampleTest(bluredImg);

	write_JPEG_file(argv[2],80,(JSAMPLE *)bluredImg,2048,2048,3,JCS_RGB);
	
	delete[] bluredImg;
	free(data);
	return 1;
}