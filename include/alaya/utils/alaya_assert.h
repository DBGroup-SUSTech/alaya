/**
 * @file alaya_assert.h
 * @brief Header file containing the ALAYA_ASSERT_MSG macro for assertion with custom error message.
 */

#pragma once

#include <cstdio>
#include <string>

/**
 * @def ALAYA_ASSERT_MSG(X, MSG)
 * @brief Macro for assertion with custom error message.
 * @param X The expression to be evaluated.
 * @param MSG The custom error message to be displayed if the assertion fails.
 *
 * This macro checks if the expression X evaluates to false. If it does, it prints an error message
 * containing the failed expression, the function name, the file name, the line number, and the custom
 * error message. It then aborts the program.
 */
#define ALAYA_ASSERT_MSG(X, MSG)                    \
  do {                                             \
    if (!(X)) {                                    \
      fprintf(stderr,                              \
              "Alaya assertion '%s' failed in %s " \
              "at %s:%d; details: " MSG "\n",      \
              #X,                                  \
              __PRETTY_FUNCTION__,                 \
              __FILE__,                            \
              __LINE__);                           \
      abort();                                     \
    }                                              \
  } while (false)
