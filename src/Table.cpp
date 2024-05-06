#include <sstream>
#include <sqlite3.h>
#include <cstring>
#include <alaya/Table/Table.h>
namespace alaya
{
template <typename DataType>
Table<DataType>::Table(std::string table_name, int dim){
    //this->dim_=dim;
    //this->table_name_=table_name;               //不知道要不要变, 关键就在怎么设置局部变量db_,如果不设置，怎么传参
    sqlite3 *db_;
    const char *dir = "sqluser.db";
    int rc = 0;
    rc = sqlite3_open_v2(dir, &db_, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE, NULL);
    this->num_=0;

    std::string create_table_sql = "CREATE TABLE IF NOT EXISTS " + table_name + " (id INT";
    for (int i = 1; i <= dim; i++) {
        create_table_sql += ", v" + std::to_string(i) + " BLOB";
    }
    create_table_sql += ")";
    const char* sql = create_table_sql.c_str();

    rc = sqlite3_exec(db_, sql, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK)
        fprintf(stderr, "error create %s\n", sqlite3_errmsg(db_));
    else
        fprintf(stdout, "create done\n");
    
    rc = sqlite3_close(db_);

}
/*
Table<DataType>::~Table(){
}*/

template <typename DataType>
void Table<DataType>::rename_table(std::string old_name, std::string new_name){
    //this->table_name_=new_name;

    sqlite3 *db_;
    const char *dir = "sqluser.db";
    int rc = 0;
    rc = sqlite3_open_v2(dir, &db_, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE, NULL);
    this->num_=0;

    std::string rename_table_sql = "ALTER TABLE " + old_name + " RENAME TO " + new_name;
    const char* sql = rename_table_sql.c_str();
    rc = sqlite3_exec(db_, sql, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK)
        fprintf(stderr, "error rename %s\n", sqlite3_errmsg(db_));
    else
        fprintf(stdout, "rename done\n");
    rc = sqlite3_close(db_);
}



template <typename DataType>
void Table<DataType>::drop_table(std::string table_name){      

    sqlite3 *db_;
    const char *dir = "sqluser.db";
    int rc = 0;
    rc = sqlite3_open_v2(dir, &db_, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE, NULL);
    this->num_=0;

    std::string drop_table_sql = "DROP TABLE IF EXISTS " + table_name;
    const char* sql = drop_table_sql.c_str();
    rc = sqlite3_exec(db_, sql, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK)
        fprintf(stderr, "error drop %s\n", sqlite3_errmsg(db_));
    else
        fprintf(stdout, "drop done\n");
    rc = sqlite3_close(db_);
}


template <typename DataType>
bool Table<DataType>::has_table(std::string table_name){

    sqlite3 *db_;
    const char *dir = "sqluser.db";
    int rc = 0;
    rc = sqlite3_open_v2(dir, &db_, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE, NULL);
    this->num_=0;

    std::string check_table_sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='" + table_name + "'";
    const char* sql = check_table_sql.c_str();

    rc = sqlite3_exec(db_, sql, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK){
        fprintf(stderr, "error has_table %s\n", sqlite3_errmsg(db_));
        rc = sqlite3_close(db_);
        return false;
    }
    else
        fprintf(stdout, "has_table done\n");

    rc = sqlite3_close(db_);
    return true;
    
}


template <typename DataType> 
bool Table<DataType>::insert_data(int num, DataType* data){       

    sqlite3 *db_;
    const char *dir = "sqluser.db";
    int rc = 0;
    rc = sqlite3_open_v2(dir, &db_, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE, NULL);
    this->num_=0;

    std::string insert_sql = "INSERT INTO " + table_name_ + " (id";
    std::string value_placeholders = "VALUES (?";
    for (int i = 1; i <= dim_; ++i) {
        insert_sql += ", v" + std::to_string(i);
        value_placeholders += ", ?";
    }
    insert_sql += ") " + value_placeholders + ")";

    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(db_, insert_sql.c_str(), -1, &stmt, nullptr);


    for(int i = 0; i<num; i++){
        num_++;                                            //编译器没有告诉我这一步的正确性
        rc = sqlite3_bind_int(stmt, 1, num_);

        for (int j = 0; j < dim_; ++j) {
            rc = bind_data<DataType>(stmt, j + 2, data[i*dim_+j]);
        }
    
        rc = sqlite3_step(stmt);
        rc = sqlite3_reset(stmt);
        

    }

    if (rc != SQLITE_OK) {
        printf("Failed to insert: %s\n", sqlite3_errmsg(db_));
        sqlite3_finalize(stmt);
        rc = sqlite3_close(db_);
        return false;
    }
    sqlite3_finalize(stmt);
    printf("Success to insert\n");
    rc = sqlite3_close(db_);
    return true;

}



template <typename DataType> 
bool Table<DataType>::insert_data(int num, int* ids, DataType* data){    

    sqlite3 *db_;
    const char *dir = "sqluser.db";
    int rc = 0;
    rc = sqlite3_open_v2(dir, &db_, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE, NULL);
    this->num_=0;

    std::string insert_sql = "INSERT INTO " + table_name_ + " (id";
    std::string value_placeholders = "VALUES (?";
    for (int i = 1; i <= dim_; ++i) {
        insert_sql += ", v" + std::to_string(i);
        value_placeholders += ", ?";
    }
    insert_sql += ") " + value_placeholders + ")";

    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(db_, insert_sql.c_str(), -1, &stmt, nullptr);

    for(int i = 0; i<num; i++){
        rc = sqlite3_bind_int(stmt, 1, ids[i]);
        
        for (int j = 0; j < dim_; ++j) {
            rc = bind_data<DataType>(stmt, j + 2, data[i*dim_+j]);
        }
    
        rc = sqlite3_step(stmt);
        rc = sqlite3_reset(stmt);
    }


    if (rc != SQLITE_OK) {
        printf("Failed to insert: %s\n", sqlite3_errmsg(db_));
        sqlite3_finalize(stmt);
        rc = sqlite3_close(db_);
        return false;
    }
    sqlite3_finalize(stmt);
    printf("Success to insert\n");
    rc = sqlite3_close(db_);
    return true;

}

} // namespace alaya