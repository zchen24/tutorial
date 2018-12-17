## About
Examples based on Kurt Smith's Cython tutorial.

## Build 

```bash
# -i: inplace build 
# -f: force clean & build
python setup_hello.py build_ext -if

# To test
$ ipython
>>> import hello
>>> hello.say_hello_to('a_name')
Hello a_name!
```




## Reference: 
1. https://github.com/kwmsmith/scipy-2015-cython-tutorial
2. https://www.youtube.com/watch?v=gMvkiQ-gOW8&t=3687s

