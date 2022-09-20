#!/usr/bin/env python

# COnvert all notebooks to .rst format and save them
# in the ../doc folder

from __future__ import print_function
import glob
import subprocess
import shutil
import os

notebooks = glob.glob("*.ipynb")

for nb in notebooks:
    
    root = nb.split(".")[0]

    cmd_line = f'ipython nbconvert --to rst {nb}'

    print(cmd_line)

    subprocess.check_call(cmd_line,shell=True)

    # Now move the .rst file and the directory with the data
    # under ../doc

    try:
        
        os.remove(f"../doc/{root}.rst")

    except:

        pass

    files_dir = f"{root}_files"

    try:
        
        shutil.rmtree(f"../doc/{files_dir}")

    except:

        pass

    shutil.move(f"{root}.rst", "../doc")

    if os.path.exists(files_dir):

        shutil.move(files_dir, "../doc")
