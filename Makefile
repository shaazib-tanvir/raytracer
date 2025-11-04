OPTIMIZED:=-O3
DEBUG_MODE:=
DEBUG_LOG_LEVEL:=0
RELEASE_LOG_LEVEL:=1

all: debug/raytracer
release: release/raytracer

debug/raytracer: debug/main.o debug/gl.o debug/vector.o debug/base.o debug/matrix.o
	nvcc -g debug/gl.o debug/main.o debug/vector.o debug/base.o debug/matrix.o -lglfw -o debug/raytracer

debug/main.o: src/main.cu
	nvcc -std=c++11 -I include/ -c -dc -g -G -DLOG_LEVEL=$(DEBUG_LOG_LEVEL) -o debug/main.o src/main.cu

debug/vector.o: src/vector.cu include/vector.cuh
	nvcc -std=c++11 -I include/ -c -dc -g -G -DLOG_LEVEL=$(DEBUG_LOG_LEVEL) -o debug/vector.o src/vector.cu

debug/matrix.o: src/matrix.cu include/matrix.cuh
	nvcc -std=c++11 -I include/ -c -dc -g -G -DLOG_LEVEL=$(DEBUG_LOG_LEVEL) -o debug/matrix.o src/matrix.cu

debug/base.o: src/base.cu include/base.cuh
	nvcc -std=c++11 -I include/ -c -dc -g -G -DLOG_LEVEL=$(DEBUG_LOG_LEVEL) -o debug/base.o src/base.cu

debug/gl.o: src/gl.c
	clang -Wall -Werror -pedantic -std=c17 -I include/ -c -g -DLOG_LEVEL=$(DEBUG_LOG_LEVEL) -o debug/gl.o src/gl.c

release/raytracer: release/main.o release/gl.o release/vector.o release/base.o release/matrix.o
	nvcc -g $(OPTIMIZED) $(DEBUG_MODE) release/main.o release/vector.o release/gl.o release/base.o release/matrix.o -lglfw -o release/raytracer

release/main.o: src/main.cu
	nvcc -std=c++11 -I include/ -c -dc $(OPTIMIZED) $(DEBUG_MODE) -DLOG_LEVEL=$(RELEASE_LOG_LEVEL) -o release/main.o src/main.cu

release/vector.o: src/vector.cu include/vector.cuh
	nvcc -std=c++11 -I include/ -c -dc $(OPTIMIZED) $(DEBUG_MODE) -DLOG_LEVEL=$(RELEASE_LOG_LEVEL) -o release/vector.o src/vector.cu

release/matrix.o: src/matrix.cu include/matrix.cuh
	nvcc -std=c++11 -I include/ -c -dc $(OPTIMIZED) $(DEBUG_MODE) -DLOG_LEVEL=$(RELEASE_LOG_LEVEL) -o release/matrix.o src/matrix.cu

release/base.o: src/base.cu include/base.cuh
	nvcc -std=c++11 -I include/ -c -dc $(OPTIMIZED) $(DEBUG_MODE) -DLOG_LEVEL=$(RELEASE_LOG_LEVEL) -o release/base.o src/base.cu

release/gl.o: src/gl.c
	clang -Wall -Werror -pedantic -std=c17 -I include/ -c $(OPTIMIZED) $(DEBUG_MODE) -DLOG_LEVEL=$(RELEASE_LOG_LEVEL) -o release/gl.o src/gl.c

clean:
	rm -f debug/* release/*
