#include <pybind11/pybind11.h>


// enumeration
enum ShapeType {
    NONE = 1,
    ROUND = 2,
    TRIANGLE = 3
};


class BasicMath {
public:
    BasicMath() = default;
    int add(int a, int b) {return (a + b);}
    int sub(int a, int b) {return (a - b);}

public:
    int a_property = 10;
};



int add(int i, int j) {
    return i + j;
}

int add_many_args(int arg0, int arg1, int arg2)
{
    return (arg0 + arg1 + arg2);
}

namespace py = pybind11;

PYBIND11_MODULE(example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    // Anonymous functions
    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

    // Arguments:
    //    1) how to specify arg name
    //    2) how to provide default args
    m.def("add_many_args", &add_many_args,
            py::arg("arg1"), py::arg("arg2"), py::arg("arg3") = 5);

    // Enums: how to export enum type
    py::enum_<ShapeType>(m, "ShapeType")
            .value("NONE", ShapeType::NONE)
            .value("ROUND", ShapeType::ROUND)
            .value("TRIANGLE", ShapeType::TRIANGLE)
            .export_values();

    // Class:
    //    1) how to warp a class
    py::class_<BasicMath>(m, "BasicMath")
            .def(py::init<>())
            .def("add", &BasicMath::add)
            .def("sub", &BasicMath::sub)
            .def_readwrite("a_property", &BasicMath::a_property);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
