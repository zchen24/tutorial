#!/usr/bin/env python3

"""
A script that numbers receipts from 01, 02 => NN

Author: Zihan Chen
Date: 2024-07-01
"""

# a folder with 4 sub folders
# 01 Hotels, 02 Travel, 03 Meals, 04 Others
# for receipts in each folder, prefix each filename from 1 to n, number in 2 digits
# for example, 01 Hotels/01 20240311 MyHotel.pdf

import os
from tkinter import filedialog
from tkinter import Tk
import copy

def number_receipts(is_dryrun = False):

    root = Tk()
    root.withdraw()

    my_dir = filedialog.askdirectory()
    print('File: {}'.format(my_dir))
    if my_dir == '':
        print('No file selected')
        return

    # get the list of folders in the current working directory
    folders = [os.path.join(my_dir, f) for f in os.listdir(my_dir) if os.path.isdir(os.path.join(my_dir, f))]
    print("folders: " + str(folders))

    # sort the folders
    folders.sort()

    i = 0

    # loop through each folder
    for folder in folders:
        # get the list of files in the folder
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        # sort the files
        files.sort()

        # loop through each file
        for file in files:

            if 'DS_Store' in file:
                continue

            old_file = copy.copy(file)
            # check if the first 3 characters are 2 digits integer followed by a space
            # if yes, remove the first 3 characters
            if file[:2].isdigit() and file[2] == ' ':
                print(f"Renaming {file} to {file[3:]}")
                # os.rename(f"{folder}/{file}", f"{folder}/{file[3:]}")
                file = file[3:]

            # get the file extension
            ext = os.path.splitext(file)[1]
            # get the new file name
            new_file = f"{folder}/{i+1:02d} {file}"

            print(f"Renaming {old_file} to {new_file}")
            # rename the file
            if not is_dryrun:
                os.rename(f"{folder}/{old_file}", new_file)

            i = i + 1


if __name__ == "__main__":

    # accept argument
    # if -d or --dry is passed, do dry run

    import argparse

    parser = argparse.ArgumentParser(description='Number Receipts')
    parser.add_argument('-d', '--dry', action='store_true', default=False, help='Dry run')

    args = parser.parse_args()
    number_receipts(args.dry)