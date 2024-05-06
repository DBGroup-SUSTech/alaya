/**
 * @brief Abstract structure of the database, mainly used to create a database and add tables in it.
 *
 */

#pragma once

#include <alaya/Table/Table.h>
#include <alaya/TableSchema/TableSchema.h>
#include <sstream>
#include <sqlite3.h>
#include <map>
#include <vector>
#include <cstdio>
#include <string.h>
#include <string>

namespace alaya {

template <typename DataType>
struct Database
{
    std::string db_name_;                          ///< Name of the database
    std::map<std::string, Table<DataType>> table_map_; ///< Mapping of table names to table objects
    std::vector<TableSchema> table_list_;          ///< List of the table objects

    /**
     * @brief Construct a new db object
     *
     * @param db_name Name of the database
     * @param cnct Database connections
     */
    Database(std::string db_name) : db_name_(db_name) {}

    ~Database() { drop_database(db_name_); }

    /**
     * @brief Function to delete the database, to be implemented
     *
     * @param db_name Name of the database
     */
    void drop_database(std::string db_name);

    /**
     * @brief Create a table object, add the map between the table name and the table object.
     *
     * @param table_name Name of the table
     * @param dim dimension of the vector
     */
    void create_table(std::string table_name, int dim, char *des);

    /**
     * @brief list all the tables in the database
     *
     */
    void show_tables();

    /**
     * @brief Function to retrieve the table object using the specified table name
     *
     * @param table_name Name of the table
     * @return Table* returned table object
     */
    Table<DataType> *use_table(std::string table_name);
};

}