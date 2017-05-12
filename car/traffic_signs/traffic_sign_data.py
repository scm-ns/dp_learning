import pickle
import os
import csv 

train_data_file = "data/train.pkl2"
test_data_file = "data/test.pkl2"

with open(train_data_file , mode ="rb") as f:
    train = pickle.load(f)
with open(test_data_file  , mode ="rb") as f:
    test = pickle.load(f)

x_train , y_train = train["features"] , train["labels"]
y_test , y_test =   test["features"] , test["labels"]

with open("signnames.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader , None) # skip header
    # create dict from reader
    sign_names = dict((int(n) , label) for n , label in reader)

# convert the key:value pairs into seperate varaible
cls_number , cls_names = zip(*sign_names.items())

print(cls_number)
print(cls_names)

