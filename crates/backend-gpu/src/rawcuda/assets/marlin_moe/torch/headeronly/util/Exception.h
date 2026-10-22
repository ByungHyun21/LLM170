#pragma once
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
