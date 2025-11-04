#ifndef BASE_CUH
#define BASE_CUH

template<typename T, size_t N>
struct Array {
	T data[N];

	__device__
	T& operator[](size_t i);
};

enum class LogLevel {
	DEBUG,
	INFO,
	WARN,
	ERROR
};

template<typename T, size_t N>
__device__
T& Array<T, N>::operator[](size_t i) {
	return data[i];
}

void log(LogLevel level, char const* format, ...);
void panic(const char* message, ...);
void panic_err(cudaError_t error);

#endif
