#pragma once

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace alaya {
class AlayaException : public std::exception {
  public:
  explicit AlayaException(const std::string& msg);

  AlayaException(
    const std::string& msg,
    const char* funcName,
    const char* file,
    int line);

  /// from std::exception
  const char* what() const noexcept override;

  std::string msg;
};

}