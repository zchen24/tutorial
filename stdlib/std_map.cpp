/**
 * Shows how to use std::map
 *
 * Author: Zihan Chen
 * Date: 2020-01-30
 *
 * BSD License
 */

#include <iostream>
#include <map>


int main(int, char**)
{
    std::map<int, int> imap;
    imap[0] = 0;
    imap[1] = 2;
    imap[2] = 4;
    imap[3] = 6;
    imap[4] = 8;
    imap[5] = 10;

    imap.erase(6);

    for (auto & it : imap) {
        std::cout << "map <" << it.first << ", " << it.second << ">\n";
    }
    std::cout << "\n";

    for (auto it = imap.begin(); it != imap.end(); it++) {
        if (it->first == 2) {
            imap.erase(it);
        }

    }

    for (auto it:imap) {
        std::cout << "map <" << it.first << ", " << it.second << ">\n";
    }
    std::cout << "\n";

    return 0;
}
