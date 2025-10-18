from abc import ABC, abstractmethod

import numpy as np


class IScalar(ABC):
    """
    Template of a scalar class
    """
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass


class ZscoreScaler(IScalar):
    def __init__(self) -> None:
        self.mean_ = None
        self.std_ = None
        self.feature_names_ = None

    def fit(self, X):
        """
        Learn the mean and std from training data.

        Args:
            X: pandas DataFrame or numpy array
        """
        if hasattr(X, 'values'):  # pandas DataFrame
            self.mean_ = X.mean()
            self.std_ = X.std()
            self.feature_names_ = X.columns.tolist()
        else:  # numpy array
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """
        Apply the learned transformation to new data.

        Args:
            X: pandas DataFrame or numpy array

        Returns:
            Transformed data with same type as input
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before transform")

        if hasattr(X, 'values'):  # pandas DataFrame
            # Ensure same columns as training data
            if self.feature_names_ is not None:
                X = X[self.feature_names_]
            return (X - self.mean_) / self.std_
        else:  # numpy array
            return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        Fit the scaler and transform the data in one step.

        Args:
            X: pandas DataFrame or numpy array

        Returns:
            Transformed data with same type as input
        """
        return self.fit(X).transform(X)
