import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl



class Priors:
    def __init__(self):
        
        self.priors = {}
        
    def calcPriors(self, y):
        
        classes, counts = np.unique(y, return_counts=True)
        
        for c in classes:
            self.priors[int(c)] = counts[int(c)]/y.size
            
    def getPriors(self):
            
        return self.priors
        

class GaussianDistribution:
    def __init__(self):
        
        self.gaussian = {}
        
    def gaussianParams(self, x1, x2, y):
        
        for c in np.unique(y):
            
            values = []
            
            values.append(x1[np.where(y==c)[0]].mean())
            values.append(x2[np.where(y==c)[0]].mean())
            values.append(x1[np.where(y==c)[0]].var())
            values.append(x2[np.where(y==c)[0]].var())
            
            self.gaussian[int(c)] = values
        
    def getParams(self):
        return self.gaussian
        
        
        
class BernoulliDistribution:
    def __init__(self):
        
        self.bernoulli = {}
        
    def bernoulliParams(self, x3, x4, y):

        for c in np.unique(y):
            
            values = []
            
            values.append(x3[np.where(y==c)[0]].mean())
            values.append(x4[np.where(y==c)[0]].mean())
            
            self.bernoulli[int(c)] = values
        
    def getParams(self):
        return self.bernoulli
    
    

class ExponentialDistribution:
    def __init__(self):
        
        self.exponential = {}
        
    def exponentialParams(self, x7, x8, y):

        for c in np.unique(y):
            
            values = []
            
            values.append(1/(x7[np.where(y==c)[0]].mean()))
            values.append(1/(x8[np.where(y==c)[0]].mean()))
            
            self.exponential[int(c)] = values
        
    def getParams(self):
        return self.exponential
    
    
class LaplaceDistribution:
    def __init__(self):
        
        self.laplace = {}
        
    def laplaceParams(self, x5, x6, y):

        for c in np.unique(y):
            
            values = []
            
            values.append(x5[np.where(y==c)[0]].mean())
            values.append(x6[np.where(y==c)[0]].mean())
            values.append(np.sqrt((x5[np.where(y==c)[0]].var())/2))
            values.append(np.sqrt((x6[np.where(y==c)[0]].var())/2))
            
            self.laplace[int(c)] = values
        
     
    def getParams(self):
        return self.laplace
    
    
class MultinomialDistribution:
    def __init__(self):
        
        self.multinomial = {}
        self.multi_counts = {}
        
    def multinomialParams(self, x9, x10, y):

        for c in np.unique(y):
            
            values = []
            counts = []
            
            x9_uniques, x9_counts = np.unique(x9, return_counts=True)
            x10_uniques, x10_counts = np.unique(x10, return_counts=True)
            
            values.append(x9_counts/y.shape[0])
            values.append(x10_counts/y.shape[0])
            
            counts.append(x9_counts)
            counts.append(x10_counts)
            
            self.multinomial[int(c)] = values
            self.multi_counts[int(c)] = counts
        
    def getParams(self):
        return (self.multinomial, self.multi_counts)



p = Priors()
g = GaussianDistribution()
b = BernoulliDistribution()
l = LaplaceDistribution()
e = ExponentialDistribution()
m = MultinomialDistribution()


class NaiveBayes:
    def fit(self, X, y):

        """Start of your code."""
        """
        X : np.array of shape (n,10)
        y : np.array of shape (n,)
        Create a variable to store number of unique classes in the dataset.
        Assume Prior for each class to be ratio of number of data points in that class to total number of data points.
        Fit a distribution for each feature for each class.
        Store the parameters of the distribution in suitable data structure, for example you could create a class for each distribution and store the parameters in the class object.
        You can create a separate function for fitting each distribution in its and call it here.
        """

        p.calcPriors(y)
        g.gaussianParams(X[:,0], X[:,1], y)
        b.bernoulliParams(X[:,2], X[:,3], y)
        l.laplaceParams(X[:,4], X[:,5], y)
        e.exponentialParams(X[:,6], X[:,7], y)
        m.multinomialParams(X[:,8], X[:,9], y)


        """End of your code."""

    def predict(self, X):
        """Start of your code."""
        """
        X : np.array of shape (n,10)

        Calculate the posterior probability using the parameters of the distribution calculated in fit function.
        Take care of underflow errors suitably (Hint: Take log of probabilities)
        Return an np.array() of predictions where predictions[i] is the predicted class for ith data point in X.
        It is implied that prediction[i] is the class that maximizes posterior probability for ith data point in X.
        You can create a separate function for calculating posterior probability and call it here.
        """
        
        predictions = []
        
        def posterior(x):
            
            probs = []
            
            priors = p.getPriors()
            gaussian = g.getParams()
            bernoulli = b.getParams()
            laplace = l.getParams()
            exponential = e.getParams()
            multinomial, counts = m.getParams()
            
            for i in range(3):
                
                prob = 0
                
                px1_y = np.log((np.exp(-np.square(x[0]-gaussian[i][0])/(2*gaussian[i][2])))/(np.sqrt(2*(np.pi)*gaussian[i][2])))
                px2_y = np.log((np.exp(-np.square(x[1]-gaussian[i][1])/(2*gaussian[i][3])))/(np.sqrt(2*(np.pi)*gaussian[i][3])))

                px3_y = np.log((bernoulli[i][0]**x[2])*(1-bernoulli[i][0])**(1-x[2]))
                px4_y = np.log((bernoulli[i][1]**x[3])*(1-bernoulli[i][1])**(1-x[3]))

                px5_y = np.log((np.exp(-abs(x[4]-laplace[i][0])/laplace[i][2]))/(2*laplace[i][2]))
                px6_y = np.log((np.exp(-abs(x[5]-laplace[i][1])/laplace[i][3]))/(2*laplace[i][3]))

                px7_y = np.log(exponential[i][0]*np.exp(-exponential[i][0]*x[6]))
                px8_y = np.log(exponential[i][1]*np.exp(-exponential[i][1]*x[7]))
                
                px9_y = 0
                
                for j in range(len(counts[i][0])):
                    px9_y += counts[i][0][j]*np.log(multinomial[i][0][j])
                    
                px10_y = 0
                
                for j in range(len(counts[i][1])):
                    px10_y += counts[i][1][j]*np.log(multinomial[i][1][j])
                
                prob = np.log(priors[i]) + px1_y + px2_y + px3_y + px4_y + px5_y + px6_y + px7_y + px8_y + px9_y + px10_y
            
                probs.append(prob)
                
            return probs
        
        for i in range(X.shape[0]):
            
            posteriors = posterior(X[i])
            predictions.append(posteriors.index(max(posteriors)))
        
        return (np.array(predictions))


        """End of your code."""

    def getParams(self):
        """
        Return your calculated priors and parameters for all the classes in the form of dictionary that will be used for evaluation
        Please don't change the dictionary names
        Here is what the output would look like:
        priors = {"0":0.2,"1":0.3,"2":0.5}
        gaussian = {"0":[mean_x1,mean_x2,var_x1,var_x2],"1":[mean_x1,mean_x2,var_x1,var_x2],"2":[mean_x1,mean_x2,var_x1,var_x2]}
        bernoulli = {"0":[p_x3,p_x4],"1":[p_x3,p_x4],"2":[p_x3,p_x4]}
        laplace = {"0":[mu_x5,mu_x6,b_x5,b_x6],"1":[mu_x5,mu_x6,b_x5,b_x6],"2":[mu_x5,mu_x6,b_x5,b_x6]}
        exponential = {"0":[lambda_x7,lambda_x8],"1":[lambda_x7,lambda_x8],"2":[lambda_x7,lambda_x8]}
        multinomial = {"0":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"1":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"2":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]]}
        """
        priors = {}
        guassian = {}
        bernoulli = {}
        laplace = {}
        exponential = {}
        multinomial = {}

        """Start your code"""
        
        priors = p.getPriors()
        guassian = g.getParams()
        bernoulli = b.getParams()
        laplace = l.getParams()
        exponential = e.getParams()
        multinomial, _ = m.getParams()
        

        """End your code"""
        return (priors, guassian, bernoulli, laplace, exponential, multinomial)        


def save_model(model,filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open("model.pkl","wb")
    pkl.dump(model,file)
    file.close()

def load_model(filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename,"rb")
    model = pkl.load(file)
    file.close()
    return model

def visualise(data_points,labels):
    """
    datapoints: np.array of shape (n,2)
    labels: np.array of shape (n,)
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('Generated 2D Data from 5 Gaussian Distributions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def net_f1score(predictions, true_labels):
    """Calculate the multclass f1 score of the predictions.
    For this, we calculate the f1-score for each class 

    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.

    Returns:
        float(list): The f1 score of the predictions for each class
    """

    def precision(predictions, true_labels, label):
        """Calculate the multclass precision of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The precision of the predictions.
        """
        """Start of your code."""
        
        
        TP = ((predictions == label) & (true_labels == label)).sum()
        FP = ((predictions == label) & (true_labels != label)).sum()
        precision = TP / (TP+FP)
        
        return precision


        
        """End of your code."""
        


    def recall(predictions, true_labels, label):
        """Calculate the multclass recall of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.
        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The recall of the predictions.
        """
        """Start of your code."""
        
        
        
        TP = ((predictions == label) & (true_labels == label)).sum()
        FN = ((predictions != label) & (true_labels == label)).sum()
        recall = TP / (TP+FN)
        
        return recall



        """End of your code."""
        

    def f1score(predictions, true_labels, label):
        """Calculate the f1 score using it's relation with precision and recall.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The f1 score of the predictions.
        """

        """Start of your code."""
        
        p = precision(predictions, true_labels, label)
        r = recall(predictions, true_labels, label)
        
        f1 = (2*p*r)/(p+r)



        """End of your code."""
        return f1
    

    f1s = []
    for label in np.unique(true_labels):
        f1s.append(f1score(predictions, true_labels, label))
    return f1s

def accuracy(predictions,true_labels):
    """

    You are not required to modify this part of the code.

    """
    return np.sum(predictions==true_labels)/predictions.size



if __name__ == "__main__":
    """

    You are not required to modify this part of the code.

    """

    # Load the data
    train_dataset = pd.read_csv('./data/train_dataset.csv',index_col=0).to_numpy()
    validation_dataset = pd.read_csv('./data/validation_dataset.csv',index_col=0).to_numpy()

    # Extract the data
    train_datapoints = train_dataset[:,:-1]
    train_labels = train_dataset[:, -1]
    validation_datapoints = validation_dataset[:, 0:-1]
    validation_labels = validation_dataset[:, -1]

    # Visualize the data
    # visualise(train_datapoints, train_labels, "train_data.png")

    # Train the model
    model = NaiveBayes()
    model.fit(train_datapoints, train_labels)

    # Make predictions
    train_predictions = model.predict(train_datapoints)
    validation_predictions = model.predict(validation_datapoints)

    # Calculate the accuracy
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(validation_predictions, validation_labels)

    # Calculate the f1 score
    train_f1score = net_f1score(train_predictions, train_labels)
    validation_f1score = net_f1score(validation_predictions, validation_labels)

    # Print the results
    print('Training Accuracy: ', train_accuracy)
    print('Validation Accuracy: ', validation_accuracy)
    print('Training F1 Score: ', train_f1score)
    print('Validation F1 Score: ', validation_f1score)

    # Save the model
    save_model(model)

    # Visualize the predictions
    # visualise(validation_datapoints, validation_predictions, "validation_predictions.png")

