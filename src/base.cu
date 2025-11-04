#include <base.cuh>
#include <cstdarg>
#include <cstdio>

void log(LogLevel level, char const* format, ...) {
	switch (level) {
		case LogLevel::DEBUG:
			{
#if LOG_LEVEL <= 0
			printf("[DEBUG]: ");
			va_list args;
			va_start(args, format);
			std::vprintf(format, args);
			va_end(args);
#endif
			break;
			}
		case LogLevel::INFO:
			{
#if LOG_LEVEL <= 1
			printf("[INFO]: ");
			std::va_list args;
			va_start(args, format);
			std::vprintf(format, args);
			va_end(args);
#endif

			break;
			}
		case LogLevel::WARN:
			{
#if LOG_LEVEL <= 2
			printf("[WARN]: ");
			std::va_list args;
			va_start(args, format);
			std::vprintf(format, args);
			va_end(args);

#endif
			break;
			}
		case LogLevel::ERROR:
			{
#if LOG_LEVEL <= 3

			printf("[ERROR]: ");
			std::va_list args;
			va_start(args, format);
			std::vprintf(format, args);
			va_end(args);
#endif

			break;
			}
	}

}

void panic(const char* message, ...) {
	std::va_list args;
	va_start(args, message);
	std::vprintf(message, args);
	va_end(args);
	exit(1);
}

void panic_err(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("error: %s\n", cudaGetErrorString(error));
		exit(1);
	}
}
