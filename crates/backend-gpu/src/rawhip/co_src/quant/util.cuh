#pragma once
// plans/136 P1-1: fork util.cuh의 GEMV TU 필요 부분만 — cublas/bf16 플럼빙 제거판
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "compat.cuh"

union half_uint16 {
    __half h;
    __half as_half;
    uint16_t as_uint16;
    __device__ half_uint16() {}
    __device__ half_uint16(uint16_t u) : as_uint16(u) {}
};

union half2_uint32 {
    __half2 h2;
    __half2 as_half2;
    uint32_t as_uint32;
    __device__ half2_uint32() {}
    __device__ half2_uint32(uint32_t u) : as_uint32(u) {}
};
