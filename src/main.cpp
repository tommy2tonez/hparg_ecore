#define DEBUG_MODE_FLAG true
#include "network_fileio_unified_x.h"
#include <iostream>
#include "network_kernel_mailbox_impl1.h"
#include <iostream>
#include "stdx.h"

int main(){
    
    std::string inp = "Hello World!";
    std::string out = std::string(inp.size(), ' '); 

    std::string root_filename = "/home/tommy2tonez/dg_projects/dg_polyobjects/test.txt";
    std::vector<std::string> datapath_vec = {"/home/tommy2tonez/dg_projects/dg_polyobjects/test1.txt", "/home/tommy2tonez/dg_projects/dg_polyobjects/test2.txt"};

    dg::network_fileio_unified_x::dg_create_cbinary(root_filename.c_str(), datapath_vec, 5);
    exception_t err1 = dg::network_fileio_unified_x::dg_write_binary(root_filename.c_str(), inp.data(), inp.size());
    exception_t err = dg::network_fileio_unified_x::dg_read_binary(root_filename.c_str(), out.data(), out.size());
    dg::network_fileio_unified_x::dg_remove(root_filename.c_str());

    std::cout << static_cast<size_t>(err1) << "<>" << out << std::endl;
}