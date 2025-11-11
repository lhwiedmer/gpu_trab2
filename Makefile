all: t2.cu
	nvcc -arch sm_50 --allow-unsupported-compiler -ccbin /usr/bin/gcc-12 -lcudart t2.cu -o mppSort -lstdc++

clean:
	rm -f mppSort