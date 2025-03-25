#include <variant>
#include <iostream>

int main(){

    std::variant<int, float> x;

    std::cout << x.index() << "<>" << std::holds_alternative<int>(x) << "<>" << std::holds_alternative<float>(x);
}