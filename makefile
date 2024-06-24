all: lanczos gen

lanczos: lanczos.cpp structures.cu structures.h
	nvcc lanczos.cpp structures.cu -o lanczos -O3 -lcublas -lcusparse -lcusolver -lnivida-ml

gen: gen_sym_coo.cu
	nvcc gen_sym_coo.cu -o gen_sym_coo -O3

clean:
	rm -f lanczos gen_sym_coo
