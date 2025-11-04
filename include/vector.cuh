#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <types.hpp>

template<u32 N>
struct Vector {
	f32 data[N];

	__host__
	__device__
	static Vector<N> init(std::initializer_list<f32> elements);

	__device__
	Vector<N> normalize() {
		return rsqrtf(norm2(*this)) * (*this);
	}

	__host__
	__device__
	f32& operator[](u32 i) {
		return data[i];
	}

	__device__
	Vector<N> operator-(void) {
		Vector<N> result{};
		for (u32 i = 0; i < N; i++) {
			result[i] = -data[i];
		}

		return result;
	}

	__device__
	Vector<N> operator+(Vector<N> other) {
		Vector<N> result{};
		for (u32 i = 0; i < N; i++) {
			result[i] = data[i] + other[i];
		}

		return result;
	}

	__device__
	Vector<N> operator-(Vector<N> other) {
		Vector<N> result{};
		for (u32 i = 0; i < N; i++) {
			result[i] = data[i] - other[i];
		}

		return result;
	}

	template<u32 M>
	__device__
	friend Vector<M> operator*(f32 x, Vector<M> v);
};

template<u32 N>
__device__
f32 dot(Vector<N> v, Vector<N> w);

template<u32 N>
__device__
f32 norm2(Vector<N> v);

__host__
__device__
Vector<3> from_projective(Vector<4> const& v4);
__host__
__device__
Vector<4> to_projective(Vector<3> const& v3);

template<u32 N>
__host__
__device__
Vector<N> Vector<N>::init(std::initializer_list<f32> elements) {
	Vector<N> result;
	auto it = elements.begin();
	for (u32 i = 0; i < N; i++) {
		result.data[i] = (*it);
		it++;
	}

	return result;
}

template<u32 N>
__device__
Vector<N> operator*(f32 x, Vector<N> v) {
	Vector<N> result{};
	for (u32 i = 0; i < N; i++) {
		result[i] = x*v[i];
	}

	return result;
}

template<u32 N>
__device__
f32 dot(Vector<N> v, Vector<N> w) {
	f32 result {};
	for (u32 i = 0; i < N; i++) {
		result += v[i] * w[i];
	}

	return result;
}

template<u32 N>
__device__
f32 norm2(Vector<N> v) {
	return dot(v, v);
}

__device__
Vector<3> get_spherical_at(f32 theta, f32 phi, Vector<3> const& normal);
#endif
