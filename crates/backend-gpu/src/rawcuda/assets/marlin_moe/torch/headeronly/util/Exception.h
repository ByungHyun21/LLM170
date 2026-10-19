#pragma once
// [2026-10-18] torch headeronly 예외 헤더 최소 심 — 마린 벤더링 컴파일 전용.
// 원본(torch/headeronly/util/Exception.h·BSD-3)은 macros/Export.h·Macros.h의
// 네임스페이스 기계에 의존하므로, 여기선 STD_TORCH_CHECK 의미(std::runtime_error
// throw + 메시지 결합)만 동일하게 제공한다. 호스트 constexpr 경로 전용 — 커널
// 본체는 이 매크로를 쓰지 않는다.
#include <sstream>
#include <stdexcept>
#include <string>

template <typename... Args>
inline std::string llm170_std_check_msg(const Args&... args) {
    std::ostringstream oss;
    ((oss << args), ...);
    return oss.str();
}

#define STD_TORCH_CHECK(cond, ...)                                              \
    ((cond) ? (void)0                                                           \
            : throw std::runtime_error(std::string("CHECK failed: ") +          \
                                       llm170_std_check_msg(__VA_ARGS__)))
#define STD_TORCH_CHECK_MSG(cond, type, ...) STD_TORCH_CHECK(cond, __VA_ARGS__)
#define TORCH_CHECK(cond, ...) STD_TORCH_CHECK(cond, __VA_ARGS__)
