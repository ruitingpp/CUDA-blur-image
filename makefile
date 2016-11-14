all:
	nvcc -O3 -g -o blur readwriteImg.cpp main.cu blur.cu  -I/Developer/NVIDIA/CUDA-8.0/samples/common/inc  -ljpeg
clean:
	rm blur 
	rm -r blur.dSYM