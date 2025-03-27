#include <variant>
#include <iostream>
#include "network_compact_serializer.h"

struct Foo{

    int a;
    std::vector<size_t> b;
    std::unordered_map<uint32_t, uint32_t> c;
    std::pair<uint32_t, uint64_t> d;

    template <class Reflector>
    void dg_reflect(const Reflector& reflector) const noexcept{
        reflector(a, b, c, d);
    }

    template <class Reflector>
    void dg_reflect(const Reflector& reflector){
        reflector(a, b, c, d);
    }
};

struct Bar{

    Foo foo;
    std::string b;
    std::vector<uint64_t> c; 
    std::array<size_t, 2> d;

    template <class Reflector>
    void dg_reflect(const Reflector& reflector) const noexcept{
        reflector(foo);
    }

    template <class Reflector>
    void dg_reflect(const Reflector& reflector){
        reflector(foo);
    }
};

int main(){

    int i = 0;
    Bar bar{Foo{1, {2}, {{3, 3}}, {4, 4}}, std::string{"qwerty"}, {1, 2, 3}, {1, 2}};

    std::string buf     = dg::network_compact_serializer::integrity_serialize<std::string>(bar);
    Bar bar2            = dg::network_compact_serializer::integrity_deserialize<Bar>(buf);
    std::string buf2    = dg::network_compact_serializer::integrity_serialize<std::string>(bar2);

    std::cout << (buf == buf2);
}