#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cassert>
#include <types.hpp>
#include <vector.cuh>

template<u32 N>
struct Matrix {
	f32 data[N][N];

	__host__
	__device__
	static Matrix<N> init(std::initializer_list<f32> elements);

	__host__
	__device__
	Vector<N> operator*(Vector<N> const& other) {
		Vector<N> result{};
		for (u32 i = 0; i < N; i++) {
			for (u32 j = 0; j < N; j++) {
				result.data[i] += data[i][j] * other.data[j];
			}
		}

		return result;
	}
	__device__
	Matrix<N> operator+(Matrix<N> const& other);
	__host__
	__device__
	Matrix<N> operator*(Matrix<N> const& other) {
		Matrix<N> result{};
		for (u32 i = 0; i < N; i++) {
			for (u32 j = 0; j < N; j++) {
				for (u32 k = 0; k < N; k++) {
					result.data[i][j] += data[i][k] * other.data[k][j];
				}
			}
		}

		return result;
	}
};

template<u32 N>
__host__
__device__
Matrix<N> Matrix<N>::init(std::initializer_list<f32> elements) {
	Matrix<N> result;
	u32 i = 0;
	for (auto element = elements.begin(); element != elements.end(); element++) {
		u32 j = i % N;
		u32 k = i / N;
		result.data[k][j] = (*element);
		i += 1;
	}

	return result;
}

template<u32 N>
__device__
Matrix<N> Matrix<N>::operator+(Matrix<N> const& other) {
	Matrix<N> result{};
	for (u32 i = 0; i < N; i++) {
		for (u32 j = 0; j < N; j++) {
			result.data[i][j] = data[i][j] + other.data[i][j];
		}
	}
}

Matrix<4> rotation_matrix(f32 yaw, f32 pitch, f32 roll);
Matrix<4> scale_matrix(f32 scale);
Matrix<4> translation_matrix(Vector<3> displacement);

#endif
