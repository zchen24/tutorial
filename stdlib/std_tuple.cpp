/**
 * Shows how to use std::tuple
 *
 * Author: Zihan Chen
 * Date: 2020-01-30
 *
 * BSD License
 */

#include <iostream>
#include <tuple>


int main(int, char**)
{
    std::tuple<char, int, float> my_tuple;
    my_tuple = std::make_tuple('m', 1, 1.5);
    std::cout << "Tuple value: "
              << std::get<0>(my_tuple) << "\t"
              << std::get<1>(my_tuple) << "\t"
              << std::get<2>(my_tuple) << "\n";

    // change value in tuple
    std::get<1>(my_tuple) = 10;
    std::cout << "Tuple value: "
              << std::get<0>(my_tuple) << "\t"
              << std::get<1>(my_tuple) << "\t"
              << std::get<2>(my_tuple) << "\n";

    return 0;
}
