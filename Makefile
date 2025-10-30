OPTIMIZED:=-O3
DEBUG_MODE:=
DEBUG_LOG_LEVEL:=0
RELEASE_LOG_LEVEL:=1

all: debug/raytracer
release: release/raytracer

debug/raytracer: debug/main.o debug/gl.o
	nvcc --std=c++11 -g debug/gl.o debug/main.o -lglfw -DLOG_LEVEL=$(DEBUG_LOG_LEVEL) -o debug/raytracer

debug/main.o: src/main.cu
	nvcc --std=c++11 -I include/ -c -g -DLOG_LEVEL=$(DEBUG_LOG_LEVEL) -o debug/main.o src/main.cu

debug/gl.o: src/gl.c
	clang -Wall -Werror -pedantic --std=c17 -I include/ -c -g -DLOG_LEVEL=$(DEBUG_LOG_LEVEL) -o debug/gl.o src/gl.c

release/raytracer: release/main.o release/gl.o
	nvcc --std=c++11 $(OPTIMIZED) $(DEBUG_MODE) -DLOG_LEVEL=$(RELEASE_LOG_LEVEL) release/main.o release/gl.o -lglfw -o release/raytracer

release/main.o: src/main.cu
	nvcc --std=c++11 -I include/ -c $(OPTIMIZED) $(DEBUG_MODE) -DLOG_LEVEL=$(RELEASE_LOG_LEVEL) -o release/main.o src/main.cu

release/gl.o: src/gl.c
	clang -Wall -Werror -pedantic --std=c17 -I include/ -c $(OPTIMIZED) $(DEBUG_MODE) -DLOG_LEVEL=$(RELEASE_LOG_LEVEL) -o release/gl.o src/gl.c

clean:
	rm -f debug/* release/*
