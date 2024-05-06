#include <string>
#include "sqlite3.h"

namespace alaya {
template <typename DataType>
struct Table
{
    std::string table_name_; ///< Name of the table
    int *ids_;               ///< Array of data IDs
    DataType* data_;         ///< Array of data                               //这个报错不知道怎么改
    //Index *index_;           ///< Pointer to an data index object
    //Searcher *searcher_;     ///< Pointer to a data searcher object
    int dim_;                ///< Dimension of the vectors
    int num_;                ///< Number of vectors in the table

    Table(std::string table_name, int dim);
    ~Table();
    void rename_table(std::string old_name, std::string new_name);
    void drop_table(std::string table_name);
    bool has_table(std::string table_name);
    bool insert_data(int num, DataType* data);
    bool insert_data(int num, int *ids, DataType* data);
    void create_index(char *des);
    void search(int k, int query_num, DataType *query_data, DataType* distance, int64_t* labels);
    
};

} // namespace alaya