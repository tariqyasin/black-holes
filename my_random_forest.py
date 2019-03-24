""" An implementation of the random forest regressor. See my_random_forest.ipynb for example use, testing and development."""

import numpy as np
import pandas as pd

class MyForest:
    """ Creates a random forest using bootstrapping.
    
    Args:
        ntrees (int): number of decision trees.
        leaf_size (int): leaf size at which to stop splitting.
        x (pandas dataframe): a table of parameter values.
        y (numpy array): the variable to be predicted.
        sample_size (int): the number of samples in each tree.
    
    Attributes:
        trees (list): a list of decision tree objects (see the class Tree).
    
    """
    def __init__(self, ntrees, leaf_size, x, y, sample_size):
        self.ntrees = ntrees
        self.leaf_size = leaf_size
        self.x = x
        self.y = y
        self.sample_size = sample_size
        self.trees = self.tree_ensemble()
        
    def tree_ensemble(self):
            idxs = [np.random.permutation(len(self.y))[:self.sample_size] for i in range(self.ntrees)]
            return [Tree(idxs[i], self.x, self.y, self.leaf_size) for i in range(self.ntrees)]
    
    def predict(self, x_test):
        """Predict y given a sample a table of parameter values."""
        return np.mean([t.predict(x_test) for t in self.trees], axis=0)
    
    def __repr__(self):
            return f'ntrees {self.ntrees}, leaf size {self.leaf_size}, sample size {self.sample_size}'

class Tree:
    """ Creates a decision tree, chosing the split point to minimize the mean squared error.
    
    Args:
        idx (numpy array): The indices of the samples still remaining in the tree branch.
        x (pandas dataframe): the table of parameter values.
        y (numpy array): the variable to be predicted.
        leaf_size (int): leaf size at which to stop splitting.
    
    Attributes:
       split_var (int):  The index of the split parameter for the node.
       split_val (float): The value of the split parameter.
       lhs (class): The subtree corresponding to samples with parameter values greater than split_val.
       rhs (class): The subtree corresponding to samples with parameter values less than or equal to split_val.
       is_leaf (boolean): True if it is a tree leaf. Otherwise equal to number of samples at node.
       score (float): the score of the best split for the node.
        
    """
    def __init__(self, idx, x, y, leaf_size):
        self.idx = idx
        self.x, self.y = x, y
        self.leaf_size = leaf_size
        self.isleaf = len(idx)
        self.score = float('inf')
        
        if len(self.idx) > self.leaf_size:
            self.split()
        else:
            self.isleaf = True
            self.mean = np.sum(y[idx] / len(idx))
            
    def split(self):
        """Recursive. Find optimal split point and then split."""
        for i in range(self.x.shape[1]):
            self.find_split(i)
        split_col = self.x.values[self.idx, self.split_var]
        idx_lhs = np.nonzero(split_col <= self.split_val)[0]
        idx_rhs = np.nonzero(split_col > self.split_val)[0]
        self.lhs = Tree(self.idx[idx_lhs], self.x, self.y, self.leaf_size)
        self.rhs = Tree(self.idx[idx_rhs], self.x, self.y, self.leaf_size)
    
    def find_split(self, var_idx):    
        """Find the best split for the variable with index var_idx. If it is better
        than all other splits tested so far assign it to class attributes."""

        x = self.x.values[self.idx, var_idx]
        y = self.y[self.idx]
        #sortidx = np.argsort(x)
        #x, y, idx_sorted = x[sortidx], y[sortidx], self.idx[sortidx]

        for j in range(len(self.idx)):
            lhs = y[x<=x[j]]
            rhs = y[x>x[j]]

            if not (len(lhs) == 0 or len(rhs) == 0):
                mean_lhs = np.sum(lhs) / len(lhs)
                mean_rhs = np.sum(rhs) / len(rhs)     
                new_score = np.sum((lhs - mean_lhs)**2) + np.sum((rhs - mean_rhs)**2)
                if new_score < self.score:
                    self.score = new_score
                    self.split_var = var_idx
                    self.split_val = (x[j] + np.min(x[x>x[j]])) / 2

    def predict_row(self, row):
        """Predict y for a 1D array of x values."""
        if self.isleaf == True:
            return self.mean
        else:
            if row[self.split_var]  > self.split_val:
                return self.rhs.predict_row(row)
            else:
                return self.lhs.predict_row(row)
    
    def predict(self, x_test):
        """Predict y for a pandas dataframe of x values."""
        data = x_test.values.tolist()
        predictions = []
        for row in data:
            predictions.append(self.predict_row(row))
        return predictions

    def __repr__(self):
        try:
            self.split_var
            return f'Variable to index split on: {self.split_var} . Value to split on:{self.split_val}'
        except:
            return f'Leaf. Mean is: {self.mean}'

if __name__ == "__main__":

    # Check it works

    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    def make_data(n):
        x1 = np.random.uniform(0, 1, n)
        x2 = np.random.uniform(0,100, n)

        y = x1 + x2/100 + np.random.normal(0, 0.2, n)

        #np.random.shuffle(x1)
        #np.random.shuffle(x2)

        x1 = x1[...,None]
        x2 = x2[...,None]
        x = np.concatenate((x1, x2), axis=1)

        x = pd.DataFrame(x)
        #x.columns = ['var1', 'var2']

        #display(pd.concat([x,pd.DataFrame(y)], axis=1))
        return x, y

    x, y = make_data(200)
    a = MyForest(10, 1, x, y, 100)
    m = RandomForestRegressor(n_estimators=10, bootstrap=True)
    m.fit(x,y)
    
    x_test, y_test = make_data(50)
    np.std(a.predict(x_test) - y_test)

    plt.figure()
    plt.scatter(y_test, a.predict(x_test), color ='b', label='my version')
    plt.scatter(y_test, m.predict(x_test), color ='r', label='SK Learn')
    plt.xlabel('Actual value')
    plt.ylabel('Prediction')
    plt.xlim(0,2)
    plt.ylim(0,2)
    plt.legend()
    plt.show()