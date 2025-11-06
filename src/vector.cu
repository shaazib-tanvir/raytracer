#include <types.hpp>
#include <vector.cuh>

__host__
__device__
Vector<3> from_projective(Vector<4> const& v4) {
	return Vector<3>{v4.data[0] / v4.data[3], v4.data[1] / v4.data[3], v4.data[2] / v4.data[3]};
}

__host__
__device__
Vector<4> to_projective(Vector<3> const& v3) {
	return Vector<4>{v3.data[0], v3.data[1], v3.data[2], 1.0f};
}

__device__
Vector<3> get_spherical_at(f32 theta, f32 phi, Vector<3> const& normal) {
	Vector<3> n = normal;
	if (normal.data[2] < 0.f) {
		n = -n;
	}

	Vector<3> result;
	f32 sincos = __sinf(theta) * __cosf(phi);
	f32 sinsin = __sinf(theta) * __sinf(phi);
	f32 cos = __cosf(theta);
	f32 nx2 = n[0]*n[0];
	f32 ny2 = n[1]*n[1];
	f32 mnxny = -n[0] * n[1];
	f32 factor = 1.f / (1.f + n[2]);

	result[0] = n[2] * sincos + n[0] * cos + (ny2 * sincos + mnxny * sinsin) * factor;
	result[1] = n[2] * sinsin + n[1] * cos + (nx2 * sinsin + mnxny * sincos) * factor;
	result[2] = n[2] * cos - n[0] * sincos - n[1] * sinsin;

	// return result;
	return normal.data[2] < 0.f ? -result : result;
}
