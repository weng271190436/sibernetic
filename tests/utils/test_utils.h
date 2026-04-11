#pragma once

#include <fstream>
#include <iostream>
#include <sstream> // IWYU pragma: keep
#include <stdexcept>
#include <string>

#define SIB_ANSI_RESET "\x1b[0m"
#define SIB_ANSI_GREEN "\x1b[32m"

#define SIB_TEST_LOG_COLOR(color, label, msg_stream)                           \
  do {                                                                         \
    std::ostringstream sib_test_log_oss;                                       \
    sib_test_log_oss << msg_stream;                                            \
    std::cerr << (color) << "[ " << (label) << " ]"                            \
              << ((color)[0] != '\0' ? SIB_ANSI_RESET : "") << " "             \
              << sib_test_log_oss.str() << "\n"                                \
              << std::flush;                                                   \
  } while (0)

#define SIB_TEST_LOG(label, msg_stream)                                        \
  SIB_TEST_LOG_COLOR("", label, msg_stream)

namespace SiberneticTest {

inline std::string readTextFile(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

} // namespace SiberneticTest