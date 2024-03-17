//
// Created by yujun on 3/15/24.
//

#include <cassert>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <alaya/utils/heap.h>
#include <alaya/utils/alaya_assert.h>
template <typename dist_t = float> struct Neighbor {
  int id;
  dist_t distance;

  Neighbor() = default;
  Neighbor(int id, dist_t distance) : id(id), distance(distance) {}

  inline friend bool operator<(const Neighbor &lhs, const Neighbor &rhs) {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.id < rhs.id);
  }
  inline friend bool operator>(const Neighbor &lhs, const Neighbor &rhs) {
    return !(lhs < rhs);
  }
};
template <typename dist_t> struct MaxHeap {
  explicit MaxHeap(int capacity) : capacity(capacity), pool(capacity) {}
  void push(int u, dist_t dist) {
    if (size < capacity) {
      pool[size] = {u, dist};
      std::push_heap(pool.begin(), pool.begin() + ++size);
    } else if (dist < pool[0].distance) {
      sift_down(0, u, dist);
    }
  }
  int pop() {
    std::pop_heap(pool.begin(), pool.begin() + size--);
    return pool[size].id;
  }
  bool empty() {
    return size == 0;
  }
  void sift_down(int i, int u, dist_t dist) {
    pool[0] = {u, dist};
    for (; 2 * i + 1 < size;) {
      int j = i;
      int l = 2 * i + 1, r = 2 * i + 2;
      if (pool[l].distance > dist) {
        j = l;
      }
      if (r < size && pool[r].distance > std::max(pool[l].distance, dist)) {
        j = r;
      }
      if (i == j) {
        break;
      }
      pool[i] = pool[j];
      i = j;
    }
    pool[i] = {u, dist};
  }
  int size = 0, capacity;
  std::vector<Neighbor<dist_t>> pool;
};

template <typename dist_t> struct MinMaxHeap {
  explicit MinMaxHeap(int capacity) : capacity(capacity), pool(capacity) {}
  bool push(int u, dist_t dist) {
    if (cur == capacity) {
      if (dist >= pool[0].distance) {
        return false;
      }
      if (pool[0].id >= 0) {
        size--;
      }
      std::pop_heap(pool.begin(), pool.begin() + cur--);
    }
    pool[cur] = {u, dist};
    std::push_heap(pool.begin(), pool.begin() + ++cur);
    size++;
    return true;
  }
  dist_t max() { return pool[0].distance; }
  void clear() { size = cur = 0; }

  int pop_min() {
    int i = cur - 1;
    for (; i >= 0 && pool[i].id == -1; --i)
      ;
    if (i == -1) {
      return -1;
    }
    int imin = i;
    dist_t vmin = pool[i].distance;
    for (; --i >= 0;) {
      if (pool[i].id != -1 && pool[i].distance < vmin) {
        vmin = pool[i].distance;
        imin = i;
      }
    }
    int ret = pool[imin].id;
    pool[imin].id = -1;
    --size;
    return ret;
  }

  int size = 0, cur = 0, capacity;
  std::vector<Neighbor<dist_t>> pool;
};

int GenRandInt(int min, int max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(min, max);
  return dis(gen);
}

void test_Bitset() {
  std::random_device rd;
  std::mt19937 gen(rd());
  int n = 1000;
  alaya::Bitset bs(n);
  std::vector<bool> vb(n);
  auto print = [&]() -> void {
    for (int i = 0; i < n; i++)
      std::putchar('0' + bs.Get(i));
    std::putchar('\n');
  };
  for (int T = 1; T <= 10; T++) {
    int Q = 100000;
    while (Q--) {
      int i = gen() % n, op = gen() % 3;
      // printf("%d %d\n", i, op);
      if (op == 0) {
        vb[i] = 0;
        bs.Reset(i);
      } else if (op == 1) {
        vb[i] = 1;
        bs.Set(i);
      } else {
        for (int j = n - 1; j > i; j--)
          vb[j] = vb[j - 1];
        vb[i] = 0;
        bs.Up(i);
      }
      bool flag = true;
      for (int i = 0; i < n; i++) {
        if (vb[i] != bs.Get(i)) {
          flag = false;
          break;
        }
      }
      // print();
      // for (int i = 0; i < n; i++)
      //   std::putchar('0' + vb[i]);
      // std::putchar('\n');
      ALAYA_ASSERT_MSG(flag, "");
    }
  }
  printf("Correctness tests passed.\n");
}
void test_MaxHeap() {
  puts("Test MaxHeap");
  // test for correctness:
  std::random_device rd;
  std::mt19937 gen(rd());
  {
    const int N = 1e9;
    for (int T = 1; T <= 100; T++) {
      int Q = 100000, M = gen() % 1000 + 1;
      printf("Run test %d with %d operations and capacity %d.\n", T, Q, M);
      alaya::MaxHeap<int, int> a(M);
      MaxHeap<int> b(M);
      while (Q--) {
        int op = gen() % 3;
        if (op == 0) {
          ALAYA_ASSERT_MSG(a.Size() == b.size, "");
          if (!a.Empty()) {
            ALAYA_ASSERT_MSG(a.Pop() == b.pop(), "");
          }
        } else {
          int id = gen() % N + 1, dis = gen() % N + 1;
          a.Push(id, dis);
          b.push(id, dis);
        }
      }
    }
    printf("Correctness tests passed.\n");
  }
  {
    // test for speed:
    int N = 1000000, M = 500;
    std::vector<int> p(N);
    std::iota(p.begin(), p.end(), 1);
    shuffle(p.begin(), p.end(), gen);
    auto start = clock();
    for (int T = 1; T <= 100; T++) {
      MaxHeap<int> hp(M);
      for (int i : p) {
        hp.push(i, i);
      }
      while (!hp.empty()) {
        // printf("%d\n", hp.pop());
        hp.pop();
      }
    }
    auto finish = clock();
    printf("%0.4lfs\n", (double)(finish - start) / CLOCKS_PER_SEC);
    start = clock();
    for (int T = 1; T <= 100; T++) {
      alaya::MaxHeap<int, int> hp(M);
      for (int i : p) {
        hp.Push(i, i);
      }
      while (!hp.Empty()) {
        // printf("%d\n", zk.pop());
        hp.Pop();
      }
    }
    finish = clock();
    printf("%0.4lfs\n", (double)(finish - start) / CLOCKS_PER_SEC);
  }
}
void test_MinMaxHeap() {
  puts("Test MinMaxHeap");
  std::random_device rd;
  std::mt19937 gen(rd());
  {
    const int N = 1e9;
    for (int T = 1; T <= 100; T++) {
      int Q = 1000000, M = gen() % 100 + 1;
      printf("Run test %d with %d operations and capacity %d.\n", T, Q, M);
      alaya::MinMaxHeap<int, int> a(M);
      MinMaxHeap<int> b(M);
      while (Q--) {
        int op = gen() % 2;
        if (op == 0) {
          ALAYA_ASSERT_MSG(a.Size() == b.size, "");
          if (a.Size()) {
            int x = a.PopMin();
            // printf("%d\n", x);
            ALAYA_ASSERT_MSG(x == b.pop_min(), "");
          }
        } else {
          int id = gen() % N + 1, dis = gen() % N + 1;
          a.Push(id, dis);
          b.push(id, dis);
        }
      }
    }
    printf("Correctness tests passed.\n");
  }
  {
    // test for speed:
    int N = 1000000, M = 50;
    std::vector<int> p(N);
    std::iota(p.begin(), p.end(), 1);
    shuffle(p.begin(), p.end(), gen);
    auto start = clock();
    for (int T = 1; T <= 100; T++) {
      MinMaxHeap<int> hp(M);
      int cnt = 0;
      for (int i : p) {
        hp.push(i, i);
        if (++cnt == 10) {
          hp.pop_min();
          cnt = 0;
        }
      }
    }
    auto finish = clock();
    printf("%0.4lfs\n", (double)(finish - start) / CLOCKS_PER_SEC);
    start = clock();
    for (int T = 1; T <= 100; T++) {
      alaya::MinMaxHeap<int, int> hp(M);
      int cnt = 0;
      for (int i : p) {
        hp.Push(i, i);
        if (++cnt == 10) {
          hp.PopMin();
          cnt = 0;
        }
      }
    }
    finish = clock();
    printf("%0.4lfs\n", (double)(finish - start) / CLOCKS_PER_SEC);
  }
}
template <typename dist_t> struct LinearPool {
  LinearPool(int capacity)
      : capacity_(capacity), data_(capacity_ + 1) {}
  int find_bsearch(dist_t dist) {
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (data_[mid].distance > dist) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  bool insert(int u, dist_t dist) {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo],
                 (size_ - lo) * sizeof(Neighbor<dist_t>));
    data_[lo] = {u, dist};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
    return true;
  }

  int pop() {
    if (cur_ >= size_) {
      return -1;
    }
    set_checked(data_[cur_].id);
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id)) {
      cur_++;
    }
    return get_id(data_[pre].id);
  }

  bool has_next() const { return cur_ < size_; }
  int id(int i) const { return get_id(data_[i].id); }
  int size() const { return size_; }
  int capacity() const { return capacity_; }

  constexpr static int kMask = 2147483647;
  int get_id(int id) const { return id & kMask; }
  void set_checked(int &id) { id |= 1 << 31; }
  bool is_checked(int id) { return id >> 31 & 1; }

  int nb, size_ = 0, cur_ = 0, capacity_;
  std::vector<Neighbor<dist_t>> data_;
};

void test_LinearPool() {
  puts("Test LinearPool");
  //std::random_device rd;
  std::mt19937 gen(19260817);
  {
    const int N = 10000000;
    for (int T = 1; T <= 100; T++) {
      int Q = 100000, M = gen() % 1000 + 1;
      printf("Run test %d with %d operations and capacity %d.\n", T, Q, M);
      alaya::LinearPool<int, int> a(M);
      LinearPool<int> b(M);
      while (Q--) {
        int op = gen() % 3;
        // printf("%d\n", Q);
        if (op == 0) {
          // printf("%d %d\n", a.size, b.size());
          ALAYA_ASSERT_MSG(a.size_ == b.size(), "");
          if (a.size_) {
            int x = a.Pop(), y = b.pop();
            // printf("%d %d\n", x, y);
            // for (int i = 0; i < a.size; i++)
            //   printf("%d ", a.pool[i].id);
            // putchar('\n');
            // for (int i = 0; i < b.size_; i++)
            //   printf("%d ", b.data_[i].id & b.kMask);
            // putchar('\n');
            ALAYA_ASSERT_MSG(x == y, "");
          }
        } else {
          int id = gen() % N + 1, dis = gen() % N + 1;
          ALAYA_ASSERT_MSG(a.size_ == b.size(), "");
          bool x = a.Insert(id, dis), y = b.insert(id, dis);
          // if (x != y) {
          //   printf("%d %d\n", x, y);
          //   for (int i = 0; i < a.size; i++)
          //     printf("%d ", a.pool[i].id);
          //   putchar('\n');
          //   for (int i = 0; i < b.size_; i++)
          //     printf("%d ", b.data_[i].id & b.kMask);
          //   putchar('\n');
          // }
          ALAYA_ASSERT_MSG(x == y, "");
        }
      }
    }
    printf("Correctness tests passed.\n");
  }
  {
    // test for speed:
    int N = 1000000, M = 50;
    std::vector<int> p(N);
    std::iota(p.begin(), p.end(), 1);
    shuffle(p.begin(), p.end(), gen);
    auto start = clock();
    for (int T = 1; T <= 100; T++) {
      LinearPool<int> hp(M);
      int cnt = 0;
      for (int i : p) {
        hp.insert(i, i);
        if (++cnt == 10) {
          hp.pop();
          cnt = 0;
        }
      }
    }
    auto finish = clock();
    printf("%0.4lfs\n", (double)(finish - start) / CLOCKS_PER_SEC);
    start = clock();
    for (int T = 1; T <= 100; T++) {
      alaya::LinearPool<int, int> hp(M);
      int cnt = 0;
      for (int i : p) {
        hp.Insert(i, i);
        if (++cnt == 10) {
          hp.Pop();
          cnt = 0;
        }
      }
    }
    finish = clock();
    printf("%0.4lfs\n", (double)(finish - start) / CLOCKS_PER_SEC);
  }
}
int main() {
  test_Bitset();
  test_MaxHeap();
  test_MinMaxHeap();
  test_LinearPool();
  return 0;
}