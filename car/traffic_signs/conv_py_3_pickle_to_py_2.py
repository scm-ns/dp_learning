#!/usr/bin/env python3
import os
import numpy
import pickle

# find all the file names ending with .pkl

file_end = "pkl3"

pickled_files = [f for f in os.listdir(".") if f.endswith(file_end)]

print("Converting following files into py 2 pickle format")
print(pickled_files)

for pf in pickled_files:
    with open(pf , "rb") as f:
        pkl  = pickle.load(f)

    pickle.dump(pkl , open(pf.split(".")[0] + ".pkl2" , "wb") , protocol = 2)
