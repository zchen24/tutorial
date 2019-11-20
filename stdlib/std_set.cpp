#include <iostream>
#include <set>

int main()
{
    std::set<int> my_set{1, 3, 5, 7, 9};
    std::cout << "Hello std_set\n"
              << "my_set = " << my_set.size() << "\n";
    for (auto i : my_set) {
        std::cout << i << "\t";
    }
}
