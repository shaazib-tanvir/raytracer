#include <matrix.cuh>
#include <math.h>

Matrix<4> rotation_matrix(f32 yaw, f32 pitch, f32 roll) {
	return Matrix<4>::init({cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll), 0.f,
					sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll), 0.f,
					sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll), 0.f,
					0.f, 0.f, 0.f, 1.f});
}

Matrix<4> scale_matrix(f32 scale) {
	return Matrix<4>::init({scale, 0.f, 0.f, 0.f,
					0.f, scale, 0.f, 0.f,
					0.f, 0.f, scale, 0.f,
					0.f, 0.f, 0.f, 1.f});
}

Matrix<4> translation_matrix(Vector<3> displacement) {
	return Matrix<4>::init({1.f, 0.f, 0.f, displacement[0],
					0.f, 1.f, 0.f, displacement[1],
					0.f, 0.f, 1.f, displacement[2],
					0.f, 0.f, 0.f, 1.f});
}
