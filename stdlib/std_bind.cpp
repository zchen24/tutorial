/**
 * From:
 * C++ Weekly - Ep 15 Using `std::bind`
 *
 * Date: 2021-08-12
 *
 * BSD License
 */

#include <iostream>
#include <functional>

void print_1(const int &i)
{
    std::cout << "print_1: " << i << "\n";
}

void print_2(int i, const std::string &s)
{
    std::cout << "print_2: " << i << " " << s << "\n";
}

int main(int, char**)
{
    // basic bind a param
    auto f_print_1_5 = std::bind(&print_1, 5);
    f_print_1_5();

    // swallow extra args (not great)
    f_print_1_5(1, 2, 3);

    int i = 5;
    auto f_print_1_i = std::bind(&print_1, i);
    // expect 5
    f_print_1_i();
    i = 6;
    // expect 5, as i is a value
    f_print_1_i();

    // now reference
    auto f_print_1_i_ref = std::bind(&print_1, std::ref(i));
    // expect 6
    f_print_1_i_ref();
    i = 7;
    // expect 7, now i is a reference
    f_print_1_i_ref();

    // using placeholder
    auto f_print_2_placeholder = std::bind(&print_2, std::ref(i), std::placeholders::_1);
    f_print_2_placeholder("hello world");

    // reorder parameters
    auto f_print_2_reorder = std::bind(&print_2, std::placeholders::_2, std::placeholders::_1);
    f_print_2_reorder("hello world", 8);

    // std::function (strong type)
    std::function<void (const std::string&, int)> f_std(f_print_2_reorder);
    f_std("str_1", 1);

    // compile error
    // f_std("str_1", 1, 2, 3);
}
