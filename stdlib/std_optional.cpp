/**
 * Shows how to use std::optional
 *
 * Reference: https://en.cppreference.com/w/cpp/utility/optional
 *
 * Author: Zihan Chen
 * Date: 2022-12-30
 *
 */

#include <optional>
#include <iostream>

// optional can be used as the return type of a factory that may fail
std::optional<std::string> create(bool b) {
    if (b)
        return "Godzilla";
    return {};
}

// std::nullopt can be used to create any (empty) std::optional
auto create2(bool b) {
    return b ? std::optional<std::string>{"Godzilla"} : std::nullopt;
}
int main()
{
    std::cout << "create(false) returned "
              << create(false).value_or("empty") << '\n';

    // optional-returning factory functions are usable as conditions of while and if
    if (auto str = create2(true)) {
        std::cout << "create2(true) returned " << *str << '\n';
    }

    auto val = create(false);
    if (val) {
        std::cout << "returned val is not empty\n";
    }
    else {
        std::cout << "returned val is empty\n";
    }
}