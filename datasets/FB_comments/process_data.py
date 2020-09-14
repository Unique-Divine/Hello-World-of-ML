# standard DS stack
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
# embed static images in the ipynb
get_ipython().run_line_magic('matplotlib', 'inline')

train_df = pd.read_csv("training.csv", header=None)
test_df = pd.read_csv("testing.csv", header=None)
print("Original dataset shapes\n"
     +f"Training set:{train_df.shape}, Testing set:{test_df.shape}")
train_df.head()


# def integer_check(vec):
#     """Args: vec (np.ndarray, 1D): a vector."""
#     if np.all(((vec % 1) == 0)) == True:
#         print("This vector contains only integers")
#     else:
#         print("This vector contains non-integer values")

# integer_check(vec=np.array(train_df.iloc[:,-1]))

train = np.array(train)
test = np.array(test)
X_train, Y_train = train[:,:-1], train[:,-1] 
X_test, Y_test = test[:,:-1], test[:,-1]

from sklearn.preprocessing import StandardScaler
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

scale_features(X_train, X_test)

reduction_method = "pca"
reduction_method = "kbest"

if reduction_method == "pca":
    # Principal component analysis (PCA) feature reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

elif reduction_method == "kbest":
    # SelectKBest feature selection
    from sklearn.feature_selection import SelectKBest, f_regression
    X_train = SelectKBest(f_regression, k=10).fit_transform(X_train, Y_train)

rng = np.random.RandomState(5)
def random_shrink(X, Y, shrink=0.5):
    """Shrinks the dataset size.

    Args:
        X (np.ndarray): feature matrix
        Y (np.ndarray): target matrix
        shrink (float, optional): Percentage of samples desired.
            Defaults to 0.5, i.e. a 50% reduction in the number of samples.

    Returns:
        X_small, Y_small : Random samples of the input sets
    """
    n_samples = X.shape[0] 
    sorted_indices = np.arange(n_samples)
    random_indices = rng.choice(sorted_indices, int(shrink * n_samples))
    X_small = X[random_indices]
    Y_small = Y[random_indices]
    return X_small, Y_small

X_train, Y_train = random_shrink(X_train, Y_train, shrink=0.25)
X_test, Y_test = random_shrink(X_test, Y_test)
print("Dataset shapes after PCA and random sampling\n"
     +f"X_train.shape:{X_train.shape}, Y_train.shape:{Y_train.shape}\n"
     +f"X_test.shape:{X_test.shape}, Y_test.shape:{Y_test.shape}")

def present_data:
    return X_train, Y_train, X_test, Y_test

present_data()

