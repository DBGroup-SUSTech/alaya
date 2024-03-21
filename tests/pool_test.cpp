#include "../include/alaya/utils/heap.h"

int main() {
  int a[10]{0, 2, 8, 9, 3, 7, 4, 6, 5, 1};
  float b[10]{0.1, 2.1, 8.1, 9.1, 3.1, 0.05, 4.1, 6.1, 5.1, 0.3};
  alaya::ResultPool<int, float> res(10, 6, 6);

  for (int i = 0; i < 10; i++) {
    res.Insert(a[i], b[i]);
  }

  for (int i = 0; i < 6; i++) {
    std::cout << res.result_.pool_[i].id_ << " " << res.result_.pool_[i].dis_ << std::endl;
  }
}