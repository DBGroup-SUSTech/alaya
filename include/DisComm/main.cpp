#include "main.hpp"
#include "mpi_env.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    Env::initEnv(argc, argv);
    std::cout << "Hello, from DisComm!\n";
    if (serverId() == 0) {
        
    }

    Env::endEnv();
}
