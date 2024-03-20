/**
 * @brief Abstract structure of the table schema, mainly to store the meta data of the tables for the database to
 * manage.
 *
 */
#pragma once

#include <string>

namespace alaya {

struct TableSchema
{
    std::string table_name_; ///< Name of the table
    int dim_;                ///< Dimension of the vectors
    char *des_;              ///< Description of the table
    char *index_path_;       ///< Path to the index file
    int db_id_;              ///< id of father db

    /**
     * @brief Construct a new Table Schema object
     *
     * @param table_name Name of the table
     * @param dim Dimension of the vectors
     * @param des Description of the table
     * @param db_id father database of the table
     */
    TableSchema(std::string table_name, int dim, char *des = NULL, int db_id) : table_name_(table_name), dim_(dim), des_(des), db_id_(db_id);
    ~TableSchema() {}
    /**
     * @brief Function to add an index to the table
     *
     */
    void add_index() {}
};

} // namespace alaya