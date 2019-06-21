#include <iostream>
#include <deque>

//! @brief Sample code showing std::deque
//! @author Zihan Chen
//! @date 2019-06-20
//! @ref https://en.cppreference.com/w/cpp/container/deque


int main(int argc, char** argv)
{
	std::deque<int> dq = {4, 3, 2, 1};
	std::cout << "dq.size() = " << dq.size() << "\n";

    // push
	dq.push_front(10);
	dq.push_back(20);
	for (auto i : dq) { std::cout << i << "  "; }
	std::cout << "\n";

    // pop
    std::cout << "front = " << dq.front() << "\n";
    dq.pop_front();
    for (auto i : dq) { std::cout << i << "  "; }
    std::cout << "\n";

    std::cout << "back  = " << dq.back() << "\n";
    dq.pop_back();
    for (auto i : dq) { std::cout << i << "  "; }
    std::cout << "\n";
}
