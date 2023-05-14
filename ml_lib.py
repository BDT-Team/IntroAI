import numpy as np
from scipy import sparse
from collections import Counter
from sklearn.base import BaseEstimator

def sigmoid(x):
    return 1 / (1+np.exp(-x))


class LogisticRegression(BaseEstimator):
    
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
        
            dw = (1/n_samples) * np.dot(X.transpose, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
            
    def predict(self, X, thresh=0.5):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y_pred < thresh else 1 for y in y_pred]
        return class_pred
    

class SoftmaxRegression(BaseEstimator):
    
    def __init__(self, lr=0.001, max_epochs=2000):
        self.lr = lr
        self.max_epochs = max_epochs
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        Y = self.convert_labels(y, n_classes)
        print(Y.shape)
        
        W_init = np.random.randn(n_features, n_classes)
        W = [W_init] 
        
        count = 0
        while count < self.max_epochs:
            for i in range(n_samples):
                xi = X[i, :].reshape(n_features, 1)
                yi = Y[:, i].reshape(n_classes, 1)
                ai = self.softmax_stable(np.dot(W[-1].T, xi))
                W_new = W[-1] + self.lr*xi.dot((yi - ai).T)
                count += 1

                W.append(W_new)
        self.weights = W[-1]
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        """
        predict output of each columns of X
        Class of each x_i is determined by location of max probability
        Note that class are indexed by [0, 1, 2, ...., C-1]
        """
        A = self.softmax_stable(self.weights.T.dot(x.T))
        return self._classes[np.argmax(A)]
        
        
    def convert_labels(self, y, C):
        """
        convert 1d label to a matrix label: each column of this 
        matrix coresponding to 1 element in y. In i-th column of Y, 
        only one non-zeros element located in the y[i]-th position, 
        and = 1 ex: y = [0, 2, 1, 0], and 3 classes then return

                [[1, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0]]
        """
        Y = sparse.coo_matrix((np.ones_like(y), 
            (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
        return Y 
    
    def softmax_stable(self, Z):
        """
        Compute softmax values for each sets of scores in Z.
        each column of Z is a set of score.    
        """
        e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
        A = e_Z / e_Z.sum(axis = 0)
        return A
        

class GaussianNaiveBayes(BaseEstimator):
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        # Calculate mean, variance, prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        
        for id, c in enumerate(self._classes):
            print(id)
            X_c = X[y==c]
            self._mean[id, :] = X_c.mean(axis=0)
            self._var[id, :] = np.var(X_c)
            self._priors[id] = X_c.shape[0] / float(n_samples)
        
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        
        # Calculate posterior probability for each class
        # y = argmax(sum(log(P(xi|y)))+log(P(y)))
        for id, c in enumerate(self._classes):
            prior = np.log(self._priors[id])
            posterior = np.sum(np.log(self._pdf(id, x))) + prior
            posteriors.append(posterior)
            
        # Return class with the highest posterior probability
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_id, x):
        mean = self._mean[class_id]
        var = self._var[class_id]
        
        exponent = np.exp(-((x-mean)**2) / (2 * var))
        return exponent / np.sqrt(2 * np.pi * var)
    

class MultinomialNaiveBayes(BaseEstimator):
    def __init__(self, alpha=1):
        self.alpha = alpha
        
    def fit(self, X, y):        
        """
        Calculates the following things- 
            class_priors_ is list of priors for each y.
            N_yi: 2D array. Contains for each class in y, the number of time each feature i appears under y.
            N_y: 1D array. Contains for each class in y, the number of all features appear under y.
            
        params
        ------
        X: 2D array. shape(n_samples, n_features)
            Multinomial data
        y: 1D array. shape(n_samples,). Labels must be encoded to integers.
        """
        self.y = y
        self.n_samples, self.n_features = X.shape
        self._classes = np.unique(y)
        self.n_classes = self._classes.shape[0]
        self._priors = np.zeros(self.n_classes, dtype=np.float64)
        
        for id, c in enumerate(self._classes):
            X_c = X[y==c]
            self._priors[id] = X_c.shape[0] / float(self.n_samples)
            
        # distinct values in each features
        self.uniques = []
        for i in range(self.n_features):
            tmp = np.unique(X[:,i])
            self.uniques.append( tmp )
            
        self.N_yi = np.zeros((self.n_classes, self.n_features)) # feature count
        self.N_y = np.zeros((self.n_classes)) # total count 
        for i in self._classes: # x axis
            indices = np.argwhere(self.y==i).flatten()
            columnwise_sum = []
            for j in range(self.n_features): # y axis
                columnwise_sum.append(np.sum(X[indices,j]))
                
            self.N_yi[i] = columnwise_sum # 2d
            self.N_y[i] = np.sum(columnwise_sum) # 1d
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        
        # Calculate posterior probability for each class
        for id, c in enumerate(self._classes):
            prior = self._priors[id]
            posterior = self._likelyhood(x, c) + np.log(prior)
            posteriors.append(posterior)
         
        # Return class with the highest posterior probability
        return self._classes[np.argmax(posteriors)]
    
    def _theta(self, x_i, i, h):
        """
        Calculates theta_yi. aka P(xi | y) using eqn(1) in the notebook.
        
        params
        ------
        x_i: int. 
            feature x_i
            
        i: int.
            feature index. 
            
        h: int or string.
            a class in y
        
        returns
        -------
        theta_yi: P(xi | y)
        """
        
        Nyi = self.N_yi[h,i]
        Ny  = self.N_y[h]
        
        numerator = Nyi + self.alpha
        denominator = Ny + (self.alpha * self.n_features)
        
        return  (numerator / denominator)**x_i
    
    def _likelyhood(self, x, h):
        """
        Calculates P(E|H) = P(E1|H) * P(E2|H) .. * P(En|H).
        
        params
        ------
        x: array. shape(n_features,)
            a row of data.
        h: int. 
            a class in y
        """
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(self._theta(x[i], i,h))
        
        return np.sum(np.log(tmp))
    
def distance(x1, x2):
    d = np.sqrt(np.sum((x1-x2)**2))
    return d

class KNN():
    def __init__(self, k=5):
        self.k = k
        
    def fit(self, X, y):
        self.X = X
        self.Y = y
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        distances = [distance(x, i) for i in self.X]
        
        # Get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y[i] for i in k_indices]
        
        # Majority vote   
        vote = Counter(k_nearest_labels).most_common()
        return vote 
    
    
class MulticlassSVM(BaseEstimator):
    def __init__(self, reg=0.1, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-6, max_epochs=2000) -> None:
        self.reg = reg
        self.lr = lr
        self.max_epochs = max_epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
    def fit(self, X, y, print_every=100):
        self.m = 0
        self.v = 0
        
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        self.W = np.random.randn(n_features, n_classes)
        
        for i in range(self.max_epochs):
            delta = 1.0
            # Compute for the scores
            scores = X.dot(self.W)
            # Record the score of the example's correct class
            correct_class_score = scores[np.arange(n_samples), y]

            # Compute for the margin by getting the max between 0 and the computed expression
            margins = np.maximum(0, scores - correct_class_score[:,np.newaxis] + delta)
            margins[np.arange(n_samples), y] = 0

            # Add all the losses together
            loss = np.sum(margins)

            # Divide the loss all over the number of training examples
            loss /= n_samples

            # Regularize
            loss += 0.5 * self.reg * np.sum(self.W * self.W)
            
            # This mask can flag the examples in which their margin is greater than 0
            X_mask = np.zeros(margins.shape)
            X_mask[margins > 0] = 1

            # As usual, we count the number of these examples where margin > 0
            count = np.sum(X_mask,axis=1)
            X_mask[np.arange(n_samples),y] = -count

            dW = X.T.dot(X_mask)

            # Divide the gradient all over the number of training examples
            dW /= n_samples

            # Regularize
            dW += self.reg * self.W
            
            # Update
            # self.W -= self.lr * dW
            self.W -= self.adam(dW, i+1)
            
            # Print
            if (i % print_every == 1):
                print(f"it {i}/{self.max_epochs}, loss = {loss}")

    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        A = self.W.T.dot(x)
        return self._classes[np.argmax(A)]
        
    def svm_loss_vectorized(self, X, y): 
        N = X.shape[0]
        loss = 0 
        
        Z = X.dot(self.W)   
        print(Z.shape)  
        
        # correct_class_score = np.choose(y, Z).reshape(N,1).T  
        correct_class_score = Z[np.arange(N), y]   
        margins = np.maximum(0, Z - correct_class_score[:, np.newaxis] + 1) 
        margins[y, np.arange(margins.shape[1])] = 0
        loss = np.sum(margins, axis = (0, 1))
        loss /= N 
        loss += 0.5 * self.reg * np.sum(self.W * self.W)
        
        F = (margins > 0).astype(int)
        F[y, np.arange(F.shape[1])] = np.sum(-F, axis = 0)
        dW = X.dot(F.T)/N + self.reg*self.W
        return loss, dW
    
    def adam(self, dW, t):
        self.m = self.beta1*self.m + (1.0-self.beta1)*dW
        self.v = self.beta2*self.v + (1.0-self.beta2)*(dW**2)
        
        mhat = self.m / (1.0 - self.beta1**t)
        vhat = self.v / (1.0 - self.beta2**t)
        
        return self.lr * mhat / (np.sqrt(vhat) + self.eps)
        
        
    
    
class LazyEnsemble():
    
    def __init__(self, nb_alpha=0.5, sr_lr=0.4, sr_max_epochs=8000):
        self.nb_alpha = nb_alpha
        self.sr_lr = sr_lr
        self.sr_max_epochs = sr_max_epochs
        
    def fit(self, X, y):
        self.nb = MultinomialNaiveBayes(alpha=self.nb_alpha)
        self.nb.fit(X, y)
        
        self.sr = SoftmaxRegression(lr=self.sr_lr, max_epochs=self.sr_max_epochs)
        self.sr.fit(X, y)
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        pass