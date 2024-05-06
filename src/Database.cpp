#include <alaya/Database/Database.h>
#include <sstream>
#include <sqlite3.h>
#include <string.h>
#include <map>
#include <vector>
#include <cstdio>
#include <iostream>

namespace alaya {

/**
    * @brief Function to delete the database, to be implemented
    *
    * @param db_name Name of the database
    */
template <typename DataType>
void Database<DataType>::drop_database(std::string db_name) {
    db_name+=".db";
    if (std::remove(db_name.c_str()) == 0) {
        std::cout << "Successfully drop database" << std::endl;
    }
    else {
        std::cerr << "Failed to drop database" << std::endl;
    }
    return ;
}

/**
    * @brief Create a table object, add the map between the table name and the table object.
    *
    * @param table_name Name of the table
    * @param dim dimension of the vector
    */

template <typename DataType>
void Database<DataType>::create_table(std::string table_name, int dim, char *des)
{
    Table<DataType> table = new alaya::Table<DataType>(table_name, dim);
    Database<DataType>::table_map_.append(table_name, table);

    TableSchema table_schema = new alaya::TableSchema(table_name, dim, des, 0);
    Database<DataType>::table_list_.push_back(table_schema);
}

/**
    * @brief list all the tables in the database
    *
    */
template <typename DataType>
void Database<DataType>::show_tables() {
    for(auto &t : Database<DataType>::table_map_){
        std::cout<<t.first<<"\t";
    }
}

/**
    * @brief Function to retrieve the table object using the specified table name
    *
    * @param table_name Name of the table
    * @return Table* returned table object
    */
   
template <typename DataType>
Table<DataType>* Database<DataType>::use_table(std::string table_name) { return *(Database<DataType>::table_map_[table_name]); }

}