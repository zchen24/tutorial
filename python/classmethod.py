#!/usr/bin/env python

"""
Python OOP Tutorial 3: classmethods and staticmethods
https://www.youtube.com/watch?v=rq8cL2XMM5M

Code snippet from the above video

Three types of methods:
1. regular instance method
2. class method (can be used as alternative constructors)
3. static method (don't operate on instances nor the class)
"""

from __future__ import print_function
import datetime


class Employee(object):
    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)

    @classmethod
    def set_raise_amt(cls, amount):
        cls.raise_amt = amount

    @classmethod
    def from_string(cls, emp_str):
        """Create Employee from string"""
        first, last, pay = emp_str.split('-')
        return cls(first, last, pay)

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        else:
            return True


if __name__ == '__main__':
    emp_1 = Employee('Corey', 'Shafer', 50000)
    emp_2 = Employee('Test', 'Employee', 60000)

    print('\n======= before raise =======')
    print('Class raise_amt = {}'.format(Employee.raise_amt))
    print(emp_1.fullname() + ' ' + str(emp_1.raise_amt))
    print(emp_2.fullname() + ' ' + str(emp_2.raise_amt))

    Employee.set_raise_amt(1.06)
    print('\n======= after raise =======')
    print('Class raise_amt = {}'.format(Employee.raise_amt))
    print(emp_1.fullname() + ' ' + str(emp_1.raise_amt))
    print(emp_2.fullname() + ' ' + str(emp_2.raise_amt))

    print('\n======= after raise emp_1 =======')
    emp_1.set_raise_amt(1.08)
    print('Class raise_amt = {}'.format(Employee.raise_amt))
    print(emp_1.fullname() + ' ' + str(emp_1.raise_amt))
    print(emp_2.fullname() + ' ' + str(emp_2.raise_amt))

    print('\n======= employee from string =======')
    emp_from_str = Employee.from_string('John-Doe-90000')
    print(emp_from_str.fullname() + ' created')

    today = datetime.date.today()
    print('Is workday today? = ' + str(Employee.is_workday(today)))
