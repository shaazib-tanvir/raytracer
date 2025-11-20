#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <types.hpp>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <vector.cuh>
#include <base.cuh>
#include <matrix.cuh>

struct WindowData {
	int framebuffer_width;
	int framebuffer_height;
};

static void print_device_info();

static void print_device_info() {
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

struct Ray {
	Vector<3> origin;
	Vector<3> direction;

	__device__
	static Ray init(Vector<3> origin, Vector<3> direction);
};

__device__
Ray Ray::init(Vector<3> origin, Vector<3> direction) {
	Ray ray;
	ray.origin = origin;
	ray.direction = direction;
	return ray;
}

struct Material {
	f32 emissivity;
	f32 albedo;
	Vector<3> color;

	static Material init(f32 emissivity, f32 albedo, Vector<3> color);
};

struct IntersectionResult {
	Vector<3> position;
	Vector<3> normal;
	Material mat;
	f32 t;
	bool intersected;
};

Material Material::init(f32 emissivity, f32 albedo, Vector<3> color) {
	Material material;
	material.color = color;
	material.emissivity = emissivity;
	material.albedo = albedo;

	return material;
}

constexpr f32 THRESHOLD = 1e-4f;

struct Sphere {
	Vector<3> center;
	f32 radius;
	Material material;

	static Sphere init(Vector<3> center, f32 radius, Material material);
	__device__
	Array<IntersectionResult, 2> intersect(Ray ray);
};

Sphere Sphere::init(Vector<3> center, f32 radius, Material material) {
	Sphere sphere;
	sphere.center = center;
	sphere.radius = radius;
	sphere.material = material;

	return sphere;
}

struct Plane {
	Vector<3> point;
	Vector<3> normal;
	Material material;

	static Plane init(Vector<3> point, Vector<3> normal, Material material);
	__device__
	IntersectionResult intersect(Ray ray);
};

Plane Plane::init(Vector<3> point, Vector<3> normal, Material material) {
	Plane plane;
	plane.normal = normal;
	plane.material = material;
	plane.point = point;

	return plane;
}

__device__
IntersectionResult Plane::intersect(Ray ray) {
	IntersectionResult result;
	result.t = dot(normal, point - ray.origin) / dot(normal, ray.direction);
	result.normal = normal;
	result.position = ray.origin + result.t * ray.direction;
	result.mat = material;
	result.intersected = result.t > THRESHOLD;

	return result;
}

__device__
Array<IntersectionResult, 2> Sphere::intersect(Ray ray) {
	Array<IntersectionResult, 2> result;
	result[0] = IntersectionResult{};
	result[1] = IntersectionResult{};

	f32 a = 1.0f;
	f32 b = 2.0f * dot(ray.direction, ray.origin - center);
	f32 c = norm2(ray.origin - center) - radius*radius;

	if (b*b - 4.0f*a*c >= 0.0f) {
		f32 d = sqrtf(b*b-4*a*c);
		result[0].t = (-b+d) / (2.0f*a);
		result[0].intersected = result[0].t >= THRESHOLD;
		result[0].position = ray.origin + result[0].t * ray.direction;
		result[0].normal = (result[0].position - center).normalize();
		result[0].mat = material;

		result[1].t = (-b-d) / (2.0f*a);
		result[1].intersected = result[1].t >= THRESHOLD;
		result[1].position = ray.origin + result[1].t * ray.direction;
		result[1].normal = (result[1].position - center).normalize();
		result[1].mat = material;
	}

	return result;
}

template<u32 SPHERE_COUNT, u32 PLANE_COUNT>
struct Scene {
	Sphere spheres[SPHERE_COUNT];
	Plane planes[PLANE_COUNT];
};

struct Camera {
	Vector<3> focal_point;
	f32 focal_length;
	Matrix<4> transform;

	static Camera init(f32 focal_length, Vector<3> position, f32 yaw, f32 pitch, f32 roll, f32 scale);
};

Camera Camera::init(f32 focal_length, Vector<3> position, f32 yaw, f32 pitch, f32 roll, f32 scale) {
	Camera camera;
	camera.transform = translation_matrix(position) * rotation_matrix(yaw, pitch, roll) * scale_matrix(scale);
	camera.focal_point = from_projective(camera.transform * to_projective(Vector<3>::init({0., 0., 0.})));
	camera.focal_length = focal_length;
	return camera;
}

constexpr u64 SEED = 10;
constexpr u32 SAMPLES_PER_PIXEL = 1000;
constexpr u32 SPHERE_COUNT = 2;
constexpr u32 PLANE_COUNT = 1;
constexpr u32 MAX_DEPTH = 10;
constexpr f32 PI = 3.141592653589793;

__constant__ Camera camera;
__constant__ Scene<SPHERE_COUNT, PLANE_COUNT> scene;

__device__
Vector<3> path_trace(Ray const& ray, curandStateXORWOW_t* state) {
	Ray r = ray;
	Vector<3> result = Vector<3>::init({0.f, 0.f, 0.f});
	Vector<3> factor = Vector<3>::init({1.f, 1.f, 1.f});
	for (u32 i = 0; i < MAX_DEPTH; i++) {
		IntersectionResult min_int{};
		for (u32 i = 0; i < SPHERE_COUNT; i++) {
			auto int_result = scene.spheres[i].intersect(r);
			if (int_result[0].intersected && (!min_int.intersected || int_result[0].t < min_int.t)) {
				min_int = int_result[0];
			}

			if (int_result[1].intersected && (!min_int.intersected || int_result[1].t < min_int.t)) {
				min_int = int_result[1];
			}
		}

		for (u32 i = 0; i < PLANE_COUNT; i++) {
			auto int_result = scene.planes[i].intersect(r);
			if (int_result.intersected && (!min_int.intersected || int_result.t < min_int.t)) {
				min_int = int_result;
			}
		}

		if (!min_int.intersected) {
			return result;
		}

		result = result + min_int.mat.emissivity * factor * min_int.mat.color;
		f32 theta = (PI / 2.f) * curand_uniform(state);
		f32 phi = (2.0f * PI) * curand_uniform(state);
		Vector<3> incoming = get_spherical_at(theta, phi, min_int.normal);
		Vector<3> brdf = (1.0f / PI) * min_int.mat.color;
		f32 cos_theta = __saturatef(dot(incoming, min_int.normal));
		factor = 2.0f * PI * min_int.mat.albedo * cos_theta * brdf * factor;
		r = Ray::init(min_int.position, incoming);
	}

	return result;
}

__global__
void draw(float* __restrict__ framebuffer, i32 width, i32 height) {
	curandStateXORWOW_t state;
	curand_init(SEED + threadIdx.x + blockIdx.x * blockDim.x, 0, 0, &state);

	for (u32 i = threadIdx.x + blockIdx.x * blockDim.x; i < width * height; i += blockDim.x * gridDim.x) {
		i32 x_idx = i % width;
		i32 y_idx = i / width;
		f32 screen_x = (f32) x_idx / width - .5;
		f32 screen_y = (f32) y_idx / width - (f32) .5 * height / width;
		Vector<3> screen_vector = Vector<3>::init({screen_x, screen_y, camera.focal_length});
		screen_vector = from_projective(camera.transform * to_projective(screen_vector));

		Vector<3> direction = (screen_vector - camera.focal_point).normalize();
		Ray ray = Ray::init(camera.focal_point, direction);
		Vector<3> color = Vector<3>::init({0.f, 0.f, 0.f});
		for (u32 j = 0; j < SAMPLES_PER_PIXEL; j++) {
			color = color + path_trace(ray, &state);
		}

		framebuffer[3*i] = (framebuffer[3*i] + color[0]) / SAMPLES_PER_PIXEL;
		framebuffer[3*i+1] = (framebuffer[3*i+1] + color[1]) / SAMPLES_PER_PIXEL;
		framebuffer[3*i+2] = (framebuffer[3*i+2] + color[2]) / SAMPLES_PER_PIXEL;
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

	{
		Camera camera_cpu = Camera::init(1.f, Vector<3>{0.f, 0.f, -3.f}, 0.1f, 0.f, 0.f, 1.f);
		cudaMemcpyToSymbol(camera, &camera_cpu, sizeof(camera_cpu));

		Scene<SPHERE_COUNT, PLANE_COUNT> scene_data;
		scene_data.spheres[0] = Sphere::init(Vector<3>::init({0.f, 0.f, 4.0f}), 1.f, Material::init(0.f, .9f, Vector<3>::init({1.f, 1.f, 1.f})));
		scene_data.spheres[1] = Sphere::init(Vector<3>::init({5.5f, 4.5f, 4.f}), 1.2f, Material::init(60.f, 0.f, Vector<3>::init({1.f, 1.f, 1.f})));
		scene_data.planes[0] = Plane::init(Vector<3>::init({0.f, -1.5f, 0.f}), Vector<3>::init({0.f, 1.f, 0.f}), Material::init(0.f, .5f, Vector<3>::init({1.f, 1.f, 1.f})));
		cudaMemcpyToSymbol(scene, &scene_data, sizeof(scene_data));
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Raytracer", nullptr, nullptr);
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

	panic_err(cudaGraphicsMapResources(1, &pbo_resource, 0));

	f32* pixel_buffer;
	size_t resource_size;
	panic_err(cudaGraphicsResourceGetMappedPointer((void**)&pixel_buffer, &resource_size, pbo_resource));
	draw<<<64, 256>>>(pixel_buffer, window_data.framebuffer_width, window_data.framebuffer_height);

	panic_err(cudaGraphicsUnmapResources(1, &pbo_resource, 0));

	while (!glfwWindowShouldClose(window)) {
		auto start = glfwGetTime();
		glfwPollEvents();

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
		log(LogLevel::INFO, "%fms\n", 1e3*delta);
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
