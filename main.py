import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

train_path = 'data/train.csv'
test_path = 'data/test.csv'


def create_train_test_files():
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print('Creating Train and Test files')
        # create train test split
        titanic = pd.read_csv('data/titanic.csv')
        train, test = train_test_split(titanic, test_size=0.2, random_state=42)
        train.to_csv('data/train.csv', index=False)
        test.to_csv('data/test.csv', index=False)
    else:
        print('Train and Test files already exist')


def main():
    create_train_test_files()
    pass


if __name__ == "__main__":
    main()
    pass
