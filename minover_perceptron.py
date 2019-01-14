import numpy as np
import plotly as py
import plotly.graph_objs as go


class Perceptron:
    
    def __init__(self, n, w_star):
        self.n = n
        # init weights to zeroes
        self.weights = [0] * n
        self.w_star = w_star
        
    def train(self, x, y, n_epochs=1000):
        epoch = 0
        success = False
        equal_found = False
        max_epochs = n_epochs * len(x)

        # continue until the max n epochs is reached or all error terms are above 0
        while epoch < max_epochs and not success:
            success = True
            stabilities = []

            print("\n Training epoch: ", epoch)
            print("=====================")

            # calculate the stability for each pattern
            for idx in range(len(x)):
                k = (np.dot(self.weights, x[idx])) * y[idx]
                stabilities.append(k)

            # find the index of the pattern with the lowest stability
            k_min = stabilities.index(min(stabilities))

            # calculate the term to add to the weights (1/N * pattern * label)
            # with the pattern with the lowest stability
            weight_addition_term = np.dot(np.dot((1 / self.n), x[k_min]), y[k_min])
            # update the weights by adding the term to the current weights
            old_weights = self.weights
            self.weights = np.add(self.weights, weight_addition_term)

            epoch += 1
            success = False
            #print("Weights: ", self.weights)

        if epoch == max_epochs:
            print("Max epochs reached")
            print("Weights: ", self.weights)
        elif success:
            print("Success in ", epoch, " epochs!")

        # TODO generalization_error =

        return success

    # function that returns the output activation given an input pattern
    def present_input(self, input):
        return np.dot(input, self.weights)
                
# generate a dataset
def gen_dataset(w_star, p=5, n=20):
    x = np.random.normal(0, 1, p*n)
    x.shape = (p,n)
    y = []

    for p in x:
        label = np.sign(np.dot(w_star, p))
        y.append(label)

    return x,y

def test_single_run():
    p = 5
    n = 20
    w_star = n * [1]

    x, y = gen_dataset(w_star, p, n)

    perceptron = Perceptron(n, w_star)
    perceptron.train(x, y)

if __name__ =="__main__":
    test_single_run()

    # n = 20 # number of dimensions
    # nd = 2  # number of datasets
    # alpha=np.linspace(0.75,3,3)
    # p=alpha*n # number of vectors
    # success_total = np.zeros(len(p)) # total success for each p value
    # for j in range(0,len(p)): # first loop over values of p
    #     for i in range(0,nd): # second loop over different datasets
    #         x, y = gen_dataset(int(p[j]), n)
    #         perceptron = Perceptron(n)
    #         success=perceptron.train(x, y)
    #         print("Success:",success)
    #         if success == True:
    #             success_total[j]+=1
    #     print("Value of p:",p[j],
    #           "\nFraction of successful runs:",success_total[j]/nd)
