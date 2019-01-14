import numpy as np
import plotly as py
import plotly.graph_objs as go


class Perceptron:
    
    def __init__(self, n):
        self.n = n
        # init weights to zeroes
        self.weights = [0] * n
        
    def train(self, x, y, n_epochs=100):
        epoch = 0
        success = False

        # continue until the max n epochs is reached or all error terms are above 0
        while epoch < n_epochs and not success:
            success = True
            stabilities = []

            print("\n Training epoch: ", epoch)
            print("=====================")

            for idx in range(len(x)):


            epoch += 1

        if epoch == n_epochs:
            print("Max epochs reached")
        elif success:
            print("Success in ", epoch, " epochs!")
        return success

    # function that returns the output activation given an input pattern
    def present_input(self, input):
        return np.dot(input, self.weights)
                
# generate a dataset
def gen_dataset(p=5, n=20):
    x = np.random.normal(0, 1, p*n)
    x.shape = (p,n)
    y = []

    w = n*[1]

    for p in x:
        label = np.sign(np.dot(w, p))
        y.append(label)

    return x,y

def test_single_run():
    p = 5
    n = 20

    x, y = gen_dataset(p, n)

    perceptron = Perceptron(n)

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
