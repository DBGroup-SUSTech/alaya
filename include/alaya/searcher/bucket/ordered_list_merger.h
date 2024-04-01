// #pragma once
#include <iostream>
#include <map>
#include <unordered_set>
#include <vector>
// indices of mergerd list element
typedef std::vector<int> MergedItemIndices;

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

class OrderedListMerger {
 public:
  OrderedListMerger() {}
  /**将传进来的lists进行merge操作
   */
  // void setLists(const std::vector<std::vector<std::pair<float, int> > >& lists);
  /**
  从priority queue中拿出距离最小的cell的坐标
  */
  // inline bool GetNextMergedItemIndices(MergedItemIndices* merged_item_indices);
  /**
   * Pointer to input lists
   */
  const std::vector<std::vector<std::pair<float, int> > >* lists_ptr;
  // std::vector<int>, MergedItemIndicesHash 自定义哈希函数，用于计算 MergedItemIndices 的哈希值
  // 应该用于判断是否已经遍历过这个cell
  std::unordered_set<MergedItemIndices, MergedItemIndicesHash> traversed_;

  void print_heap() {
    for (auto it = heap_.begin(); it != heap_.end(); ++it) {
      // 打印第二个元素（vector<int>）
      std::cout << "Second element of pair: ";
      for (int element : it->second) {
        std::cout << element << " ";
      }
      std::cout << std::endl;
    }
  }
  /**
   *  This function pushes new item into priority queue
   * @param merged_item_indices indices of item to add
   */
  void InsertMergedItemIndicesInHeap(const MergedItemIndices& merged_item_indices) {
    float sum = 0;
    for (int list_index = 0; list_index < lists_ptr->size(); ++list_index) {
      // sum 是dist的sum
      sum += lists_ptr->at(list_index)[merged_item_indices[list_index]].first;
    }
    // float  和  vector<int> 作为一个pair插入到heap中
    heap_.insert(std::make_pair(sum, merged_item_indices));
  }

  void setLists(const std::vector<std::vector<std::pair<float, int> > >& lists) {
    lists_ptr = &lists;
    heap_.clear();
    // std::vector<int> MergedItemIndices;
    MergedItemIndices first_item_indices(lists.size());
    for (int list_index = 0; list_index < lists.size(); ++list_index) {
      first_item_indices[list_index] = 0;
    }
    // todo 这里需要给traversed_ 初始化
    InsertMergedItemIndicesInHeap(first_item_indices);
  }
  /**
   * This function tries to update priority queue after yielding
   * @param merged_item_indices new indices we should try to push in priority queue
   */
  void UpdatePrioirityQueue(MergedItemIndices& merged_item_indices) {
    for (int list_index = 0; list_index < lists_ptr->size(); ++list_index) {
      if (merged_item_indices[list_index] >= lists_ptr->at(list_index).size()) {
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
    InsertMergedItemIndicesInHeap(merged_item_indices);
  }
  inline bool GetNextMergedItemIndices(MergedItemIndices* next_merged_item_indices) {
    if (heap_.empty()) {
      return false;
    }
    // std::multimap<float, MergedItemIndices> heap_;
    *next_merged_item_indices = heap_.begin()->second;
    traversed_.insert(*next_merged_item_indices);  // 加入traversed 中。
    //   yielded_items_indices_.SetValue(1, *next_merged_item_indices);
    // 几个subspace 就往几个方向扩展
    for (int list_index = 0; list_index < lists_ptr->size(); ++list_index) {
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
