#include "dg_map_variants.h"

int main(){

    dg::map_variants::unordered_unstable_map<size_t, int> map_container{};
    map_container.insert({1, 1});
}
