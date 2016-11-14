#include <cuda_runtime.h>
#include <helper_cuda.h>

#define KERNEL_RADIUS 4

#define KERNEL_LENGTH (2*KERNEL_RADIUS+1)

#define BLOCK_WIDTH  16
#define BLOCK_HEIGHT 4

//each thread deal with 8 pixels, each pixels interval 48 
// 
#define IMAGE_COMPONENT					3
#define ROW_PROC_PIXELS_EACH_THREAD 	8					
#define ROW_BLOCK_DIM_X					16

#define ROW_THREAD_PIX_INTERVAL			ROW_BLOCK_DIM_X
#define ROW_BLOCK_DIM_Y					8
#define ROW_PADDING						1

__global__ void BlurRow(const unsigned char* image,const int imgWidth,const int imgHeight,
              			unsigned char* outImage);

__global__ void TestKernel(const unsigned char* inImage);
