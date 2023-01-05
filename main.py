import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2


def main():
    pass


if __name__ == "__main__":
    # create train test split
    titanic = pd.read_csv('data/titanic.csv')
    train, test = train_test_split(titanic, test_size=0.2, random_state=42)
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)

    pass
