// #pragma once
#include <iostream>
#include <map>
#include <unordered_set>
#include <vector>
// indices of mergerd list elements
using MergedItemIndices = std::vector<int>;

// 自定义哈希函数，用于计算 MergedItemIndices 的哈希值
struct MergedItemIndicesHash {
  size_t operator()(const MergedItemIndices& vec) const {
    size_t seed = vec.size();
    for (const int& i : vec) {
      // 使用 C++ 标准库提供的哈希函数来计算每个元素的哈希值，并将其组合
      seed ^= std::hash<int>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
template <typename DataType, typename IDType>
class OrderedListMerger {
 public:
  OrderedListMerger() {}
  ~OrderedListMerger() {}
  const std::vector<int*>* order_ptr;
  const std::vector<DataType*>* centroids_dist_ptr;
  int bucket_num_;
  // std::vector<int>, MergedItemIndicesHash 自定义哈希函数，用于计算 MergedItemIndices 的哈希值
  // 用于判断是否已经遍历过这个cell
  std::unordered_set<MergedItemIndices, MergedItemIndicesHash> traversed_;

  /**
   *  函数接收一个坐标，根据centroids_dist_ptr计算出距离和，然后插入到heap中
   */
  void InsertMergedItemIndicesInHeap(const MergedItemIndices& merged_item_indices) {
    float sum = 0;
    for (int list_index = 0; list_index < order_ptr->size(); ++list_index) {
      // 根据坐标系坐标得到真实坐标，计算真实坐标的距离和
      sum += centroids_dist_ptr->at(list_index)[merged_item_indices[list_index]];
    }
    // 将距离和 以及 对应的二维坐标插入到heap中
    heap_.insert(std::make_pair(sum, merged_item_indices));
  }

  void SetLists(const std::vector<int*>& order, const std::vector<DataType*>& centroids_dist,
                int bucket_num) {
    order_ptr = &order;
    centroids_dist_ptr = &centroids_dist;

    bucket_num_ = bucket_num;
    heap_.clear();
    std::vector<int> first_item_indices(order.size());
    for (int list_index = 0; list_index < order.size(); ++list_index) {
      first_item_indices[list_index] = 0;
    }
    // 初始化时先插入坐标为0, 0的元素
    InsertMergedItemIndicesInHeap(first_item_indices);
  }
  /**
   * This function tries to update priority queue after yielding
   * @param merged_item_indices new indices we should try to push in priority queue
   */
  void UpdatePrioirityQueue(MergedItemIndices& merged_item_indices) {
    for (int list_index = 0; list_index < order_ptr->size(); ++list_index) {
      if (merged_item_indices[list_index] >= bucket_num_) {
        return;
      }
      int current_index = merged_item_indices[list_index];
      merged_item_indices[list_index] -= 1;
      if (current_index > 0 &&
          !TraversedJudgement(&merged_item_indices)) {  // 在这里判断是不是他的上级没有被pop
        merged_item_indices[list_index] += 1;
        return;
      } else {
        merged_item_indices[list_index] += 1;
      }
    }
    // 如果通过判断，将新的坐标插入到heap中
    InsertMergedItemIndicesInHeap(merged_item_indices);
  }
  // next_merged_item_indices 下一个要merge的item的坐标
  inline bool GetNextMergedItemIndices(MergedItemIndices* next_merged_item_indices) {
    if (heap_.empty()) {
      return false;
    }
    // 从heap的第一个开始，先取出坐标，放到 next_merged_item_indices 中
    *next_merged_item_indices = heap_.begin()->second;
    traversed_.insert(*next_merged_item_indices);  // 加入traversed 中。
    // 几个subspace 就往几个方向扩展
    for (int list_index = 0; list_index < order_ptr->size(); ++list_index) {
      // 然后向各个方向扩展  更新优先队列
      // 先把第一个维度上的坐标+1
      next_merged_item_indices->at(list_index) += 1;
      UpdatePrioirityQueue(*next_merged_item_indices);
      next_merged_item_indices->at(list_index) -= 1;
    }
    heap_.erase(heap_.begin());
    return true;
  }
  /**
   *  Proirity queue for multilist algorithm
   */
  std::multimap<float, MergedItemIndices> heap_;

  inline bool TraversedJudgement(MergedItemIndices* current_merge_item) {
    auto it = traversed_.find(*current_merge_item);
    if (it != traversed_.end()) {
      return true;
    }
    return false;
  }
};
