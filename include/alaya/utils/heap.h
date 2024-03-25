#pragma once

#include <iostream>
#include <alaya/utils/memory.h>

namespace alaya {
/**
 * @brief 
 * 
 * @param
 */
struct Bitset {
  explicit Bitset(int n) : 
    kBsCnt((n + 63) / 64), kBytes(kBsCnt * 8), 
    bs_((uint64_t*)Alloc64B(kBytes)) {
    memset(bs_, 0, kBytes);
  }
  ~Bitset() { free(bs_); }
  void Set(int i) {
    bs_[i >> 6] |= uint64_t(1) << (i & 63);
  }
  /**
   * @brief 
   * 
   * @param i 
   */
  void Reset(int i) {
    bs_[i >> 6] &= uint64_t(-1) ^ (uint64_t(1) << (i & 63));
  }
  bool Get(int i) {
    return bs_[i >> 6] >> (i & 63) & 1;
  }
  void Up(int i) {
    int p = i >> 6;
    int q = i & 63;
    uint64_t b = bs_[p] >> 63;
    uint64_t nb = 0;
    if (q != 63) {
      bs_[p] = ((bs_[p] >> q) << (q + 1)) | (bs_[p] & ((uint64_t(1) << q) - 1));
    } else {
      bs_[p] &= (uint64_t(1) << q) - 1;
    }
    for (int j = p + 1; j < kBsCnt; j++) {
      nb = bs_[j] >> 63;
      bs_[j] = (bs_[j] << 1) | b;
      b = nb;
    }
  }
  const int kBsCnt;
  const int kBytes;
  uint64_t *bs_;
};

template<typename IDType, typename DistType>
struct Node {
  IDType id_;
  DistType dis_;
  Node() = default;
  Node(IDType id_, DistType dis_) : id_(id_), dis_(dis_) {}
  inline friend bool operator < (const Node &l, const Node &r) {
    return l.dis_ < r.dis_ || (l.dis_ == r.dis_ && l.id_ < r.id_);
  }
};

template<typename IDType, typename DistType>
struct MaxHeap {
  explicit MaxHeap(int capacity_) : size_(0), capacity_(capacity_) {
    pool_ = new Node<IDType, DistType>[capacity_];
  }
  ~MaxHeap () { delete[] pool_; }
  void Push(IDType u, DistType dis) {
    if (size_ < capacity_) {
      pool_[size_++] = {u, dis};
      int now = size_ - 1;
      while (now > 0) {
        int nxt = (now - 1) / 2;
        if (pool_[nxt] < pool_[now])
          std::swap(pool_[now], pool_[nxt]);
        now = nxt;
      }
    } else if (dis < pool_[0].dis_) {
      pool_[0] = {u, dis};
      Down();
    }
  }
  IDType Pop() {
    std::swap(pool_[0], pool_[--size_]);
    Down();
    return pool_[size_].id_;
  }
  bool Empty() {
    return size_ == 0;
  }
  int Size() {
    return size_;
  }
  void Down(int now = 0) {
    while (true) {
      int ls = now * 2 + 1;
      int rs = now * 2 + 2;
      int mx = ls;
      if (ls >= size_) {
        break;
      }
      if (rs < size_) {
        if (pool_[mx] < pool_[rs]) {
          mx = rs;
        }
      }
      if (pool_[now] < pool_[mx]) {
        std::swap(pool_[now], pool_[mx]);
        now = mx;
      } else {
        break;
      }
    }
  }
  int size_, capacity_;
  Node<IDType, DistType> *pool_;
};

template<typename IDType, typename DistType>
struct MinMaxHeap {
  explicit MinMaxHeap(int capacity) : capacity_(capacity) {
    pool_ = new Node<IDType, DistType>[capacity_];
  }
  ~MinMaxHeap () { delete[] pool_; }
  bool Push(IDType u, DistType dis) {
    if (pos_ < capacity_) {
      pool_[pos_++] = {u, dis};
      int now = pos_ - 1;
      while (now > 0) {
        int nxt = (now - 1) / 2;
        if (pool_[nxt] < pool_[now])
          std::swap(pool_[now], pool_[nxt]);
        now = nxt;
      }
      size_++;
      return true;
    } else if (dis < pool_[0].dis_) {
      if (pool_[0].id_ == -1) {
        size_++;
      }
      pool_[0] = {u, dis};
      Down();
      return true;
    }
    return false;
  }
  IDType PopMin() {
    int p = -1;
    DistType dis_ = 0;
    for (int i = 0; i < pos_; i++) {
      if (p == -1 || (pool_[i].id_ != -1 && pool_[i].dis_ < dis_)) {
        dis_ = pool_[i].dis_;
        p = i;
      }
    }
    if (p == -1) {
      return -1;
    }
    IDType res = pool_[p].id_;
    pool_[p].id_ = -1;
    size_--;
    return res;
  }
  bool Empty() {
    return size_ == 0;
  }
  int Size() {
    return size_;
  }
  void Down(int now = 0) {
    while (true) {
      int ls = now * 2 + 1;
      int rs = now * 2 + 2;
      int mx = ls;
      if (ls >= pos_) {
        break;
      }
      if (rs < pos_) {
        if (pool_[mx] < pool_[rs]) {
          mx = rs;
        }
      }
      if (pool_[now] < pool_[mx]) {
        std::swap(pool_[now], pool_[mx]);
        now = mx;
      } else {
        break;
      }
    }
  }
  int pos_ = 0, size_ = 0, capacity_;
  Node<IDType, DistType> *pool_;
};

template<typename IDType, typename DistType>
struct LinearPool {
  explicit LinearPool(int capacity) : pos_(0), size_(0), capacity_(capacity), del_(capacity), vis_(capacity) {
    pool_ = new Node<IDType, DistType>[capacity + 1];
  }
  ~LinearPool() { delete[] pool_; }
  int UpperBound(DistType dis) {
    int l = 0;
    int r = size_ - 1;
    int res = size_;
    while (l <= r) {
      int mid = (l + r) / 2;
      if (pool_[mid].dis_ > dis) {
        res = mid;
        r = mid - 1;
      } else {
        l = mid + 1;
      }
    }
    return res;
  }
  bool Insert(IDType u, DistType dis) {
    if (size_ == capacity_ && dis >= pool_[size_ - 1].dis_) {
      return false;
    }
    int p = UpperBound(dis);
    del_.Up(p);
    std::memmove(pool_ + p + 1, pool_ + p, sizeof(Node<IDType, DistType>) * (size_ - p));
    pool_[p] = {u, dis};
    if (size_ < capacity_) {
      size_++;
    }
    if (p < pos_) {
      pos_ = p;
    }
    return true;
  }
  IDType Pop() {
    if (pos_ >= size_) {
      return -1;
    }
    del_.Set(pos_);
    IDType res = pool_[pos_].id_;
    while (pos_ < size_ && del_.Get(pos_)) {
      pos_++;
    }
    return res;
  }
  int pos_, size_, capacity_;
  Node<IDType, DistType> *pool_;
  Bitset del_, vis_;
};

template<typename IDType, typename DistType>
struct ResultPool {
  ResultPool(int n, int capacity, int k) : 
    n_(n), capacity_(capacity), k_(k), 
    result_(k), candidate_(capacity), vis_(n) {}
  bool Insert(IDType u, DistType dis) {
    result_.Push(u, dis);
    return candidate_.Push(u, dis);
  }
  IDType Pop() { return candidate_.PopMin(); }
  bool HasNext() const { return candidate_.size_ > 0; }
  IDType ID(int i) const { return result_.pool_[i].id_; }
  int Capacity() const { return capacity_; }
  int n_, capacity_, k_;
  MaxHeap<IDType, DistType> result_;
  MinMaxHeap<IDType, DistType> candidate_;
  Bitset vis_;
};

}