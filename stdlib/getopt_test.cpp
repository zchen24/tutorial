#include <iostream>
#include <stdlib.h>
#include <unistd.h>


//! @brief Sample code shows how to use geopt
//! @author Zihan Chen
//! @date 2013-08-30


void print_usage()
{
    std::cout << "Useage: getopt_tset [-h] [-a arg]" << std::endl
              << "   -h:  for help" << std::endl
              << "   -a:  set a argument" << std::endl;
}

int main(int argc, char** argv)
{
    // option string
    // e.g.
    //   h     option h
    //   a:    option h, requires arg
    //   b::   option h, optional arg
    const char short_options[] = "ha:";

    // declare flag and arg for option a
    bool a_flag = false;
    int a_arg;

    int next_option;
    do {
        next_option = getopt(argc, argv, short_options);
        std::cout << "optind = " << optind
                  << "  nopt = " << next_option << std::endl;

        switch(next_option)
        {
        case 'h':  // opt h, print usage
            print_usage();
            break;
        case 'a':  // opt a, with arg
            a_flag = true;
            a_arg = atoi(optarg);
            break;
        case '?':  // invalid arg
            return EXIT_FAILURE;
            break;
        case -1:  // parsing complete
            break;
        default:
            break;
        }
    }
    while(next_option != -1);

    // use a_flag & a_arg here
    if (a_flag) {
        std::cout << "a opt, arg = " << a_arg << std::endl;
    }

    return EXIT_SUCCESS;
}
