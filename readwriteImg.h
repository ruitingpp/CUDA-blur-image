#include <stdio.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <string.h>
#include <stdlib.h>
#include <setjmp.h>

extern GLOBAL(int) read_JPEG_file (const char * filename, unsigned char** imgData);
extern GLOBAL(void) write_JPEG_file (char * filename, int quality,JSAMPLE * imageBuffer,
                              const int& imageHeight, const int& imageWidth, const int& components,
                              J_COLOR_SPACE colorSpace);