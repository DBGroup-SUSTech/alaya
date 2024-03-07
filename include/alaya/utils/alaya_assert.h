#pragma once

#include <cstdio>
#include <string>

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
