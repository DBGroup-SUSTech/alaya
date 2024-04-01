#include <alaya/TableSchema/TableSchema.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h>
#include <string.h>
#include <sstream>
using namespace std;

static int callback(void* NotUsed, int argc, char** argv, char** azColName) {  
    EXPECT_STREQ(argv[1], "1");
    EXPECT_STREQ(argv[2], "test");
    EXPECT_STREQ(argv[3], "3");
    EXPECT_STREQ(argv[4], "abcde");
    return 0;
}

TEST(TableschemaTest, Construct) {
    sqlite3 *db;
    const char *dir = "sqluser.db";
    int rc = 0;
    rc = sqlite3_open_v2(dir, &db, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE, NULL);
    
    const char *tbName = "TableSchema("
                "id             INTEGER     PRIMARY KEY     AUTOINCREMENT,"
                "db_id 	        INTEGER,"
                "table_name 	TEXT,"
                "dim         	INTEGER,"
                "description 	TEXT,"
                "index_path 	TEXT);";

    string tablePrefix = "CREATE TABLE IF NOT EXISTS \t";
    ostringstream str1;
    str1 << tablePrefix << tbName;
    string sql0 = str1.str();

    EXPECT_FLOAT_EQ(32.0, 32.0);

    char *msgErr;
    rc = sqlite3_exec(db, sql0.c_str(), NULL, 0, &msgErr);
    if (rc != SQLITE_OK)
    {
        fprintf(stderr, "Error Create Table %s\n", sqlite3_errmsg(db));
        sqlite3_free(msgErr);
    }
    else
        fprintf(stderr, "Table created successfully \n");


    int db_id = 1;
    int dim = 3;
    std::string table_name = "test";
    char arr[233]="abcde";
    char *des = arr;

    alaya::TableSchema test_ts(table_name, dim, des, db_id);

    char *sql;
    sql = "SELECT * FROM TableSchema;";
    rc = sqlite3_exec(db, sql, callback, 0, &msgErr); 

    sqlite3_close(db);
  
}
