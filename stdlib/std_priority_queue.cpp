// std priority queue
// Date  : 2023-01-29
// Reference:
// https://en.cppreference.com/w/cpp/container/priority_queue

#include <iostream>
#include <queue>

using namespace std;

template<typename T>
void print(std::string_view name, T const& q)
{
    std::cout << name << ": \t";
    for (auto const& n : q)
        std::cout << n << ' ';
    std::cout << '\n';
}

template<typename Q>
void print_queue(std::string_view name, Q q)
{
    // NB: q is passed by value because there is no way to traverse
    // priority_queue's content without erasing the queue.
    for (std::cout << name << ": \t"; !q.empty(); q.pop())
        std::cout << q.top() << ' ';
    std::cout << '\n';
}

struct MyStruct
{
    int priority{255}; // 0-255, 0 highest priority

    explicit MyStruct(int priority)
            : priority(priority) {
    };
    friend ostream& operator<<(ostream& os, const MyStruct& dt);
};

ostream& operator<<(ostream& os, const MyStruct& dt)
{
    os << "p:" << dt.priority << " ";
    return os;
}

bool operator<(const MyStruct& p1, const MyStruct& p2)
{
    return p1.priority > p2.priority;
}

int main()
{
    const auto data = {1, 8, 5, 6, 3, 4, 0, 9, 7, 2};
    print("data", data);

    // max priority queue
    std::priority_queue<int> q1;
    for (int n : data)
        q1.push(n);
    print_queue("q1", q1);

    // min priority queue, specify compare function
    std::priority_queue q2(data.begin(), data.end(), std::greater<int>());
    print_queue("q2", q2);

    // using struct
    std::priority_queue<MyStruct> q4;
    for (int n : data) {
        q4.push(MyStruct(n));
    }
    print_queue("q4", q4);

    // using lambda to compare elements.
    auto cmp = [](int left, int right) { return (left ^ 1) < (right ^ 1); };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> q5(cmp);
    for (int n : data)
        q5.push(n);
    print_queue("q5", q5);
}
