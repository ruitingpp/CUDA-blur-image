
#include "blur.h"


__global__ void BlurImage(const unsigned char* image,const int imgWidth,const int imgHeight,
						  const unsigned char component, unsigned char* outImage)
{
	//const float kernel[9] = {1.0/16,2.0/16,1.0/16,2.0/16,4.0/16,2.0/16,1.0/16,2.0/16,1.0/16};

	  int bx = blockIdx.x;
    int by = blockIdx.y;


    int w = (imgWidth+1)*component;
    int h = (imgHeight+1);

    int oneDPosition = by*(gridDim.x*blockDim.x)+(bx*blockDim.x)+threadIdx.x;
    // Since we pad one row of 0s at first row of the image, so the real image data starts from row 1
    // instead of row 0, and similarly, we pad one column of 0s at first column, so the real image
    // data starts from column 3rd, instead of column 0. (RGB has 3 component)
    int y = oneDPosition/(imgWidth*component) +1;
    int x = oneDPosition%(imgWidth*component) +3;

    //printf("x=%d,y=%d\r\n",x,y);

    int oneComponent = (int)(image[ ((y-1)%h)*w + ((x-component)%w) ]  +
    		  				           image[ ((y-1)%h)*w + (x%w) ]*2         +
    		  				           image[ ((y-1)%h)*w + ((x+component)%w) ]  +
    		  				           image[ (y%h)*w    + ((x-component)%w) ]*2  +
              				       image[ (y%h)*w  + (x%w) ]*4          +
              				       image[ (y%h)*w  + ((x+component)%w) ]*2  +
              				       image[ ((y+1)%h)*w + ((x-component)%w) ]  +
              				       image[ ((y+1)%h)*w + ((x)%w) ]*2       +
              				       image[ ((y+1)%h)*w + ((x+component)%w) ])/16.0 ;
    outImage[(y-1)*(imgWidth*3)+(x-3)] = oneComponent;
}

__device__ void VoidOptimCodeTest(unsigned char* a, unsigned char* b, unsigned char* c,
                                  unsigned char* out)
{
  *a = 1;
  *b = 2;
  *c = 3;
  *out = *a+ *b+ *c;
}

__global__ void BlurRow(const unsigned char* inImage,const int imgWidth,const int imgHeight,
                        unsigned char* outImage)
{
  //share memory
  const int rowStride = imgWidth * IMAGE_COMPONENT;
  const int shareDateHeight = ROW_BLOCK_DIM_Y;
  const int shareDateWidth = (ROW_PROC_PIXELS_EACH_THREAD + 2 * ROW_PADDING) * ROW_BLOCK_DIM_X;
  __shared__ unsigned char shareRed[shareDateHeight][shareDateWidth];
  __shared__ unsigned char shareBlue[shareDateHeight][shareDateWidth];
  __shared__ unsigned char shareGreen[shareDateHeight][shareDateWidth];

  const int baseX = (blockIdx.x * ROW_PROC_PIXELS_EACH_THREAD - ROW_PADDING) * blockDim.x * IMAGE_COMPONENT 
                    + threadIdx.x*IMAGE_COMPONENT;
  const int baseY = blockIdx.y*blockDim.y + threadIdx.y;

  const int currentPos = baseY * rowStride + baseX;

  inImage += currentPos;
  outImage += currentPos;

//   // this code is just for caching the data, memory coalesced, cache all 64 bytes data 
//   // for 16 threads of halp warp. (I don't know the how many bytes GPU can cache, and how to cache.
//   // Since inImage is unsigned char type, which is 1 byte, so I don't know whether GPU can still cache
//   // 64 bytes or just 1*16 = 16 bytes. If it's later, then I have to use float type to read such that
//   // GPU can cache 4*16 =64 bytes).
//  float  a = ((float*)inImage)[ROW_BLOCK_DIM_X];

//   //Load main data; unroll the for loop to improve performance a little bit

  //volatile unsigned int *a = ((unsigned int*)(&inImage[3*ROW_BLOCK_DIM_X+1])); 

// #pragma unroll
  for(int i=ROW_PADDING; i< ROW_PROC_PIXELS_EACH_THREAD + ROW_PADDING; i++)
  {
    shareRed[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X]   = inImage[3*i * ROW_BLOCK_DIM_X];//(a&(0xff<<8))>>8 ;
    shareGreen[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X] = inImage[3*i * ROW_BLOCK_DIM_X+1];//(a&(0xff<<16))>>16;
    shareBlue[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X]  = inImage[3*i * ROW_BLOCK_DIM_X+2];//(a&(0xff<<24))>>24;
  }

//   //load left padding; unroll the for loop to improve performance a little bit
//#pragma unroll
  for(int i=0; i< ROW_PADDING; i++)
  {
    shareRed[threadIdx.y][threadIdx.x ] = 
          baseX >= -i*ROW_BLOCK_DIM_X*IMAGE_COMPONENT ? inImage[3*i * ROW_BLOCK_DIM_X] : 0;

   shareGreen[threadIdx.y][threadIdx.x ] = 
          baseX >= -i*ROW_BLOCK_DIM_X*IMAGE_COMPONENT ? inImage[3*i * ROW_BLOCK_DIM_X+1] : 0;

   shareBlue[threadIdx.y][threadIdx.x ]  = 
          baseX >= -i*ROW_BLOCK_DIM_X*IMAGE_COMPONENT ? inImage[3*i * ROW_BLOCK_DIM_X+2] : 0;

  }

//   //load left padding; unroll the for loop to improve performance a little bit
//#pragma unroll
  for(int i=ROW_PROC_PIXELS_EACH_THREAD + ROW_PADDING; 
          i<ROW_PROC_PIXELS_EACH_THREAD + ROW_PADDING + ROW_PADDING; i++)
  {
    shareRed[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X] = 
        rowStride - baseX > i*ROW_BLOCK_DIM_X*3 ? inImage[3*i * ROW_BLOCK_DIM_X]:0;

    shareGreen[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X] = 
        rowStride - baseX > i*ROW_BLOCK_DIM_X*3 ? inImage[3*i * ROW_BLOCK_DIM_X + 1]:0;

    shareBlue[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X] = 
        rowStride - baseX > i*ROW_BLOCK_DIM_X*3 ? inImage[3*i * ROW_BLOCK_DIM_X + 2]:0;
  }
  __syncthreads();

//   //start convolution
  float sumRed = 0.0;
  float sumGreen = 0.0;
  float sumBlue = 0.0;
  for(int i=ROW_PADDING; i< ROW_PROC_PIXELS_EACH_THREAD + ROW_PADDING; i++)
  {
    sumRed = 0.0;
    sumGreen = 0.0;
    sumBlue = 0.0;
    for(int j=-KERNEL_RADIUS; j<=KERNEL_RADIUS;j++)
    {
      sumRed   += shareRed[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X + j];
      sumGreen += shareGreen[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X + j];
      sumBlue  += shareBlue[threadIdx.y][threadIdx.x + i * ROW_BLOCK_DIM_X + j];
    }
    outImage[3*i * ROW_BLOCK_DIM_X]     = sumRed   / KERNEL_LENGTH;
    outImage[3*i * ROW_BLOCK_DIM_X + 1] = sumGreen / KERNEL_LENGTH;
    outImage[3*i * ROW_BLOCK_DIM_X + 2] = sumBlue  / KERNEL_LENGTH;
  }
  // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
  // {
  //   printf("imgWidth = %d, imageHeight = %d\n", imgWidth, imgHeight);
  //   printf("baseX = %d, baseY = %d\n", baseX, baseY);
  // }
}
__global__ void TestKernel(const unsigned char* inImage)
{
    unsigned int a = *((unsigned int*)(&inImage[0]));
    if(threadIdx.x ==0&&threadIdx.y ==0 && blockIdx.x==0 && blockIdx.y==0)
    {
        printf("%x\n",a);
        printf("%x,%x,%x,%x\n",a&0xFF, (a>>8)&0xFF,(a>>16)&0xFF,(a>>24)&0xFF);
    } 
}





