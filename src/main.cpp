#include <array>
#include <type_traits>

int main(){

    static_assert(std::has_unique_object_representations_v<std::array<char , 2>>);

}