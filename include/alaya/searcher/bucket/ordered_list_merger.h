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
template <typename DataType, typename IDType>
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
  const std::vector<int*>* order_ptr;
  const std::vector<DataType*>* centroids_dist_ptr;
  int bucket_num_;
  // std::vector<int>, MergedItemIndicesHash 自定义哈希函数，用于计算 MergedItemIndices 的哈希值
  // 应该用于判断是否已经遍历过这个cell
  std::unordered_set<MergedItemIndices, MergedItemIndicesHash> traversed_;

  void print_heap() {
    for (auto it = heap_.begin(); it != heap_.end(); ++it) {
      // 打印第二个元素（vector<int>）
      std::cout << "first and second element of pair: ";
      std::cout << it->first << " ";
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
    for (int list_index = 0; list_index < order_ptr->size(); ++list_index) {
      // sum 是dist的sum
      sum += centroids_dist_ptr->at(list_index)[merged_item_indices[list_index]];
    }
    // 把距离和  和  两个对应的坐标插入到heap中
    std::cout << "dist " << sum << "  ";
    for (int i = 0; i < merged_item_indices.size(); ++i) {
      std::cout << "pair " << merged_item_indices[i] << " ";
    }
    heap_.insert(std::make_pair(sum, merged_item_indices));
  }

  void setLists(const std::vector<int*>& order, const std::vector<DataType*>& centroids_dist,
                int bucket_num) {
    order_ptr = &order;
    centroids_dist_ptr = &centroids_dist;

    bucket_num_ = bucket_num;
    heap_.clear();
    // std::vector<int> MergedItemIndices;  其实就是一个二维数组而已,也就是两个维度下的索引
    std::vector<int> first_item_indices(order.size());
    for (int list_index = 0; list_index < order.size(); ++list_index) {
      first_item_indices[list_index] = 0;
    }
    // todo 这里需要给traversed_ 初始化
    // 先插入一个0，0
    InsertMergedItemIndicesInHeap(first_item_indices);
  }
  /**
   * This function tries to update priority queue after yielding
   * @param merged_item_indices new indices we should try to push in priority queue
   */
  void UpdatePrioirityQueue(MergedItemIndices& merged_item_indices) {
    for (int list_index = 0; list_index < order_ptr->size(); ++list_index) {
      // TODO 这个地方 需要拿到bucket_num_
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
    InsertMergedItemIndicesInHeap(merged_item_indices);
  }
  // next_merged_item_indices 下一个要merge的item的坐标
  inline bool GetNextMergedItemIndices(MergedItemIndices* next_merged_item_indices) {
    if (heap_.empty()) {
      return false;
    }
    // 从heap的第一个开始，先取出坐标，放到 next_merged_item_indices 中
    // 然后把这个坐标加入到traversed_中
    // std::multimap<float, MergedItemIndices> heap_;
    *next_merged_item_indices = heap_.begin()->second;
    traversed_.insert(*next_merged_item_indices);  // 加入traversed 中。
    //   yielded_items_indices_.SetValue(1, *next_merged_item_indices);
    // 几个subspace 就往几个方向扩展
    for (int list_index = 0; list_index < order_ptr->size(); ++list_index) {
      // 然后向各个方向扩展  更新优先队列
      // 先把第一个维度上的坐标+1
      next_merged_item_indices->at(list_index) += 1;
      UpdatePrioirityQueue(*next_merged_item_indices);
      next_merged_item_indices->at(list_index) -= 1;
    }
    for (int i = 0; i < (*next_merged_item_indices).size(); ++i) {
      std::cout << "-.-.-.-.-.-..-.-.--..-.-.-.-.-.-.-.-.-" << std::endl;
      std::cout << (*next_merged_item_indices)[i] << std::endl;
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
