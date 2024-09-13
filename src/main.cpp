#include <tuple>
#include <functional>

int main(){

    // std::pair<>
    int a = 0;
    int b = 0;
    
    std::pair<int&, int&> c{a, b};
    std::pair<int, int> d = c;
    
     // std::pair<std::reference_wrapper<int>, std::reference_wrapper<int>> c{std::reference_wrapper<int>(a), std::reference_wrapper<int>(b)};
    // c = c;


}