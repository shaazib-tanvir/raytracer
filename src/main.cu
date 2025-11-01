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
#include <curand_kernel.h>

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
struct Matrix4 {
	float4 x;
	float4 y;
	float4 z;
	float4 w;
};

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
static f32 dot_float4(float4 a, float4 b) {
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

__device__
static float4 to_projective(float3 a) {
	return make_float4(a.x, a.y, a.z, 1.0f);
}

__device__
static float3 from_projective(float4 a) {
	return make_float3(a.x / a.w, a.y / a.w, a.z / a.w);
}

__device__
static float4 apply_mat4(Matrix4 mat, float4 vector) {
	return make_float4(dot_float4(mat.x, vector),
			dot_float4(mat.y, vector),
			dot_float4(mat.z, vector),
			dot_float4(mat.w, vector));
}

__device__
static Matrix4 mat_mul4(Matrix4 mat0, Matrix4 mat1) {
	Matrix4 result{}; 
	result.x.x = dot_float4(mat0.x, make_float4(mat1.x.x, mat1.y.x, mat1.z.x, mat1.w.x));
	result.x.y = dot_float4(mat0.x, make_float4(mat1.y.x, mat1.y.y, mat1.y.z, mat1.y.w));
	result.x.z = dot_float4(mat0.x, make_float4(mat1.z.x, mat1.z.y, mat1.z.z, mat1.z.w));
	result.x.w = dot_float4(mat0.x, make_float4(mat1.w.x, mat1.w.y, mat1.w.z, mat1.w.w));

	result.y.x = dot_float4(mat0.y, make_float4(mat1.x.x, mat1.y.x, mat1.z.x, mat1.w.x));
	result.y.y = dot_float4(mat0.y, make_float4(mat1.y.x, mat1.y.y, mat1.y.z, mat1.y.w));
	result.y.z = dot_float4(mat0.y, make_float4(mat1.z.x, mat1.z.y, mat1.z.z, mat1.z.w));
	result.y.w = dot_float4(mat0.y, make_float4(mat1.w.x, mat1.w.y, mat1.w.z, mat1.w.w));

	result.z.x = dot_float4(mat0.z, make_float4(mat1.x.x, mat1.y.x, mat1.z.x, mat1.w.x));
	result.z.y = dot_float4(mat0.z, make_float4(mat1.y.x, mat1.y.y, mat1.y.z, mat1.y.w));
	result.z.z = dot_float4(mat0.z, make_float4(mat1.z.x, mat1.z.y, mat1.z.z, mat1.z.w));
	result.z.w = dot_float4(mat0.z, make_float4(mat1.w.x, mat1.w.y, mat1.w.z, mat1.w.w));

	result.w.x = dot_float4(mat0.w, make_float4(mat1.x.x, mat1.y.x, mat1.z.x, mat1.w.x));
	result.w.y = dot_float4(mat0.w, make_float4(mat1.y.x, mat1.y.y, mat1.y.z, mat1.y.w));
	result.w.z = dot_float4(mat0.w, make_float4(mat1.z.x, mat1.z.y, mat1.z.z, mat1.z.w));
	result.w.w = dot_float4(mat0.w, make_float4(mat1.w.x, mat1.w.y, mat1.w.z, mat1.w.w));

	return result;
}

__device__
static Matrix4 scale_mat4(f32 scale) {
	Matrix4 result {};
	result.x.x = scale;
	result.y.y = scale;
	result.z.z = scale;
	result.w.w = 1.0f;
	return result;
}

__device__
static Matrix4 rotation_mat4(f32 yaw, f32 pitch, f32 roll) {
	Matrix4 result {};
	result.x.x = __cosf(pitch) * __cosf(yaw);
	result.x.y = -__sinf(pitch);
	result.x.z = __cosf(pitch) * __sinf(yaw);
	result.w.w = 1.0f;
	return result;
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
constexpr f32 SCALE = 1.0f;
constexpr u64 SEED = 926831508;

__global__
void draw(float* framebuffer, i32 width, i32 height, f32 samples_count) {
	curandStateXORWOW_t state;
	curand_init(SEED + threadIdx.x + blockIdx.x * blockDim.x, 0, (u32) samples_count, &state);

	for (u32 i = threadIdx.x + blockIdx.x * blockDim.x; i < width * height; i += blockDim.x * gridDim.x) {
		i32 x_idx = i % width;
		i32 y_idx = i / width;
		f32 screen_x = (f32) x_idx / width - .5;
		f32 screen_y = (f32) y_idx / width - (f32) .5 * height / width;
		// for (u32 i = 0; i < SAMPLES_PER_FRAME; i++) {
		// }

		// framebuffer[3*i] = curand_uniform(&state);
		// framebuffer[3*i+1] = curand_uniform(&state);
		// framebuffer[3*i+2] = curand_uniform(&state);
	}
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
	f32 samples_count = 0.0f;
	while (!glfwWindowShouldClose(window)) {
		auto start = glfwGetTime();
		glfwPollEvents();

		panic_err(cudaGraphicsMapResources(1, &pbo_resource, 0));

		f32* pixel_buffer;
		size_t resource_size;
		panic_err(cudaGraphicsResourceGetMappedPointer((void**)&pixel_buffer, &resource_size, pbo_resource));
		draw<<<32, 256>>>(pixel_buffer, window_data.framebuffer_width, window_data.framebuffer_height, samples_count);

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
		log(LogLevel::DEBUG, "%fms\n", 1e3*delta);
		samples_count += SAMPLES_PER_FRAME;
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
