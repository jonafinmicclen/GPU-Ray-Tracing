#pragma once

#include "Vec3.cuh"

static void rotateVec3(Vec3& vec, double angleRadians, const Vec3 axis) {

    double cosAngle = cos(angleRadians);
    double sinAngle = sin(angleRadians);

    double oneMinusCos = 1.0 - cosAngle;

    double axisX = axis.x;
    double axisY = axis.y;
    double axisZ = axis.z;

    double newX = (cosAngle + oneMinusCos * axisX * axisX) * vec.x +
        (oneMinusCos * axisX * axisY - sinAngle * axisZ) * vec.y +
        (oneMinusCos * axisX * axisZ + sinAngle * axisY) * vec.z;

    double newY = (oneMinusCos * axisY * axisX + sinAngle * axisZ) * vec.x +
        (cosAngle + oneMinusCos * axisY * axisY) * vec.y +
        (oneMinusCos * axisY * axisZ - sinAngle * axisX) * vec.z;

    double newZ = (oneMinusCos * axisZ * axisX - sinAngle * axisY) * vec.x +
        (oneMinusCos * axisZ * axisY + sinAngle * axisX) * vec.y +
        (cosAngle + oneMinusCos * axisZ * axisZ) * vec.z;

    vec.x = newX;
    vec.y = newY;
    vec.z = newZ;
}

static void translate(Vec3& vec, const Vec3& translation) {
    vec.x += translation.x;
    vec.y += translation.y;
    vec.z += translation.z;
}

static void rotateVec3AroundPoint(Vec3& vec, double angleRadians, const Vec3& axis, const Vec3& point) {
    translate(vec, { -point.x, -point.y, -point.z });
    rotateVec3(vec, angleRadians, axis);
    translate(vec, { point.x, point.y, point.z });
}