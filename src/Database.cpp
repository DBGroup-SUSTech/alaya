#include <alaya/Database/Database.h>
#include <alaya/Table/Table.h>
#include <sstream>
#include <sqlite3.h>
#include <cstring>
#include <map>
#include <vector>
#include <cstdio>

namespace alaya {

/**
    * @brief Function to delete the database, to be implemented
    *
    * @param db_name Name of the database
    */
void Database::drop_database(std::string db_name) {
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
void create_table(std::string table_name, int dim, char *des = NULL)
{
    alaya::Table table = new alaya::Table(table_name, dim);
    table_map_.append(table_name, table);

    alaya::TableSchema table_schema = new alaya::TableSchema(table_name, dim, des, 0);
    table_list_.append(table_schema);
}

/**
    * @brief list all the tables in the database
    *
    */
void show_tables() {
    for(auto &t : table_map_){
        cout<<t.first<<"\t";
    }
}

/**
    * @brief Function to retrieve the table object using the specified table name
    *
    * @param table_name Name of the table
    * @return Table* returned table object
    */
Table *use_table(string table_name) { return *(table_map_[table_name]); }

}