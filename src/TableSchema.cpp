#include <alaya/TableSchema/TableSchema.h>
#include <sstream>

namespace alaya {

/**
 * @brief Construct a new Table Schema object
 *
 * @param table_name Name of the table
 * @param dim Dimension of the vectors
 * @param des Description of the table
 * @param db_id father database of the table
 */
TableSchema(std::string table_name, int dim, char *des = NULL, int db_id) : table_name_(table_name), dim_(dim), des_(des), db_id_(db_id){
    sqlite3 *db;
    const char *dir = "sqluser.db";
    int rc = 0;
    rc = sqlite3_open_v2(dir, &db, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE, NULL);
    
    char *sql;
    sql = "INSERT INTO TableSchema (db_id, table_name, dim, description, index_path)"
            "VALUES(?, ?, ?, ?, '');";
    sqlite3_stmt* stmt;
    
    sqlite3_prepare(db, sql, strlen(sql), &stmt, 0);

    sqlite3_bind_int(stmt, 1, db_id);
    sqlite3_bind_text(stmt, 2, table_name, strlen(table_name), NULL);
    sqlite3_bind_int(stmt, 3, dim);
    sqlite3_bind_text(stmt, 4, des, strlen(des), NULL);

    sqlite3_step(stmt);


    if (rc != SQLITE_OK)
        fprintf(stderr, "error insert %s\n", sqlite3_errmsg(db));
    else
        fprintf(stdout, "insert done\n");
    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

~TableSchema() {}

/**
 * @brief Function to add an index to the table
 *
 */
void add_index() {}

}  // namespace alaya 