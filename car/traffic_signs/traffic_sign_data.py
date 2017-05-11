import pickle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import csv 

train_data_file = "./data/train.p"
test_data_file = "./data/test.p"

with open(train_data_file , mode ="rb") as f:
    train = pickle.load(f)
with open(test_data_file  , mode ="rb") as f:
    test = pickle.load(f)

x_train , y_train = train["features"] , train["labels"]
y_test , y_test =   test["features"] , test["labels"]

with open("signnames.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    print(reader)

    

