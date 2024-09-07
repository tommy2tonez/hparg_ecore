#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <iostream>

int main(){

    int fd = open("/home/tommy2tonez/dg_projects/dg_polyobjects/src/network_virtual_device.h", O_RDONLY, S_IRWXU);
    int sz = lseek(fd, 0L, SEEK_END);

    std::cout << sz << std::endl;
}