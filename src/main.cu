#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <array>
#include <types.hpp>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

struct WindowData {
	int framebuffer_width;
	int framebuffer_height;
};

enum class LogLevel {
	DEBUG,
	INFO,
	WARN,
	ERROR
};

void log(LogLevel level, char const* format, ...);
void panic(char* message, ...);
void panic_err(cudaError_t error);
void print_device_info();

void log(LogLevel level, char const* format, ...) {
	switch (level) {
		case LogLevel::DEBUG:
			{
#if LOG_LEVEL <= 0
			printf("[DEBUG]: ");
			std::va_list args;
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

void print_device_info() {
	int device_count;
	panic_err(cudaGetDeviceCount(&device_count));
	for (int i = 0; i < device_count; i++) {
		cudaDeviceProp prop;
		panic_err(cudaGetDeviceProperties(&prop, i));
		printf("=============================================\n");
		printf("Device Name: %s\n", prop.name);
		printf("Total Constant Memory: %zuB\n", prop.totalConstMem);
		printf("Total Global Memory: %zuB\n", prop.totalGlobalMem);
		printf("Bus Width: %db\n", prop.memoryBusWidth);
		printf("L2 Cache: %dB\n", prop.l2CacheSize);
	}
	printf("=============================================\n");
}

__device__
static float3 add_float3(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__
static float3 sub_float3(float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__
static float3 scalar_mul(f32 x, float3 a) {
	return make_float3(a.x * x, a.y * x, a.z * x);
}

__device__
static f32 dot_float3(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__
static f32 norm2(float3 a) {
	return dot_float3(a, a);
}

__device__
static float3 normalize(float3 a) {
	f32 factor = rsqrtf(norm2(a));
	return scalar_mul(factor, a);
}

__host__
__device__
struct Ray {
	float3 origin;
	float3 direction;
};

__host__
__device__
struct Material {
	f32 refractive_index;
	float3 color;
};

__host__
__device__
struct IntersectionResult {
	bool intersected;
	float3 normal;
	f32 t;
	Material material;
};

__host__
__device__
struct Plane {
	float3 point;
	float3 normal;
	Material material;

	__device__
	IntersectionResult intersect(Ray ray);
};

__host__
__device__
struct Sphere {
	float3 center;
	f32 radius;
	Material material;

	__device__
	std::array<IntersectionResult, 2> intersect(Ray ray);
	__device__
	float3 calc_normal(float3 position);
};

__device__
IntersectionResult Plane::intersect(Ray ray) {
	IntersectionResult result {};
	f32 t = dot_float3(add_float3(ray.origin, point), normal) / dot_float3(ray.direction,  normal);
	result.t = t;
	result.normal = normal;
	result.intersected = t > 0.0f;
	result.material = material;

	return result;
}

__device__
float3 Sphere::calc_normal(float3 position) {
	float3 displacement = sub_float3(position, center);
	return normalize(displacement);
}

__device__
std::array<IntersectionResult, 2> Sphere::intersect(Ray ray) {
	IntersectionResult result0 {};
	IntersectionResult result1 {};
	f32 a = 1.0f;
	f32 b = 2.0f*dot_float3(ray.direction, sub_float3(ray.origin, center));
	f32 c = norm2(sub_float3(ray.origin, center)) - radius*radius;

	if (b*b-4.0f*a*c < 0.0f) {
		result0.intersected = false;
		result1.intersected = false;
	} else {
		result0.intersected = true;
		result0.material = material;
		f32 t0 = (-b + sqrtf(b*b-4*a*c)) / (2.0f * a);
		result0.t = t0;
		float3 position0 = add_float3(ray.origin, scalar_mul(t0, ray.direction));
		result0.normal = calc_normal(position0);

		result1.intersected = true;
		result1.material = material;
		f32 t1 = (-b - sqrtf(b*b-4*a*c)) / (2.0f * a);
		result1.t = t1;
		float3 position1 = add_float3(ray.origin, scalar_mul(t1, ray.direction));
		result1.normal = calc_normal(position1);
	}

	std::array<IntersectionResult, 2> result {result0, result1};
	return result;
}

constexpr u32 SAMPLES_PER_FRAME = 64;

__global__
void draw(float* radiance) {
}

static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
	// WindowData* window_data = (WindowData*) glfwGetWindowUserPointer(window);
}

int main() {
	constexpr int WIDTH = 1280;
	constexpr int HEIGHT = 720;

	print_device_info();
	if (!glfwInit()) {
		panic("error: failed to initialize glfw");
	}

	int device;
	cudaDeviceProp prop {};
	prop.major = 5;
	prop.minor = 0;
	cudaChooseDevice(&device, &prop);

	cudaSetDevice(device);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Cuda Sandbox", nullptr, nullptr);
	assert(window != nullptr);

	WindowData window_data;
	glfwGetFramebufferSize(window, &window_data.framebuffer_width, &window_data.framebuffer_height);
	glfwSetWindowUserPointer(window, &window_data);
	glfwSetCursorPosCallback(window, cursor_pos_callback);

	glfwMakeContextCurrent(window);
	assert(gladLoadGL(glfwGetProcAddress) != 0);

	GLuint pbo;
	int pbo_size = window_data.framebuffer_width * window_data.framebuffer_height * 3 * sizeof(f32);
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, pbo_size, nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	cudaGraphicsResource_t pbo_resource;
	panic_err(cudaGraphicsGLRegisterBuffer(&pbo_resource, pbo, cudaGraphicsRegisterFlagsNone));

	GLuint framebuffer;
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window_data.framebuffer_width, window_data.framebuffer_height, 0, GL_RGB, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	f32 delta = 0.0f;
	while (!glfwWindowShouldClose(window)) {
		auto start = glfwGetTime();
		glfwPollEvents();

		panic_err(cudaGraphicsMapResources(1, &pbo_resource, 0));

		f32* pixel_buffer;
		size_t resource_size;
		panic_err(cudaGraphicsResourceGetMappedPointer((void**)&pixel_buffer, &resource_size, pbo_resource));

		panic_err(cudaGraphicsUnmapResources(1, &pbo_resource, 0));

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glBindTexture(GL_TEXTURE_2D, texture);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glActiveTexture(texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, window_data.framebuffer_width, window_data.framebuffer_height, GL_RGB, GL_FLOAT, nullptr);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glBindTexture(GL_TEXTURE_2D, 0);

		glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glBlitFramebuffer(0, 0, window_data.framebuffer_width, window_data.framebuffer_height, 0, 0, window_data.framebuffer_width, window_data.framebuffer_height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

		glfwSwapBuffers(window);
		delta = glfwGetTime() - start;
		log(LogLevel::DEBUG, "%f\n", delta);
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
