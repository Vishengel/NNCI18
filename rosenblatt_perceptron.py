import numpy as np
random_number = 2018
#np.random.seed(random_number)

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

            print("\n Training epoch: ", epoch)
            print("=====================")

            for idx in range(len(x)):
                # calculate the error E (weights * pattern * label)
                error = np.dot(self.weights, x[idx]) * y[idx]

                # debug statements
                #print("Weights: ", self.weights, " - x: ", x[idx], " - y: ", y[idx])
                #print("Error: ", error)

                # if error is lower than or equal to 0, update weights
                if error <= 0:
                    # if we have at least one below-zero error, set success to False
                    success = False
                    # calculate the term to add to the weights (1/N * pattern * label)
                    weight_addition_term = np.dot(np.dot((1/self.n), x[idx]), y[idx])
                    # update the weights by adding the term to the current weights
                    self.weights = np.add(self.weights, weight_addition_term)

            epoch += 1

        if epoch == n_epochs:
            print("Max epochs reached")
        elif success:
            print("Success in ", epoch, " epochs!")

    # function that returns the output activation given an input pattern
    def present_input(self, input):
        return np.dot(input, self.weights)
                
# generate a dataset
def gen_dataset(p=5, n=20):
    x = np.random.normal(0, 1, p*n)
    x.shape = (p,n)
    y = np.random.choice([-1,1], p, True)

    return x,y

# test perceptron by training it on a logical and
# training does not work correctly at the moment
def gen_and_gate():
    x = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]
         ]

    y = [-1, -1, -1, 1]

    x = np.array(x)
    y = np.array(y)

    perceptron = Perceptron(2)

    perceptron.train(x, y, 100)

    print(perceptron.weights)

    print(perceptron.present_input([0, 0]))
    print(perceptron.present_input([0, 1]))
    print(perceptron.present_input([1, 0]))
    print(perceptron.present_input([1, 1]))

# test perceptron by training it on a logical or
# training does not work correctly at the moment
def gen_or_gate():
    x = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]
         ]

    y = [-1, 1, 1, 1]

    x = np.array(x)
    y = np.array(y)

    perceptron = Perceptron(2)
    perceptron.train(x, y, 100)

    print(perceptron.weights)

    print(perceptron.present_input([0, 0]))
    print(perceptron.present_input([0, 1]))
    print(perceptron.present_input([1, 0]))
    print(perceptron.present_input([1, 1]))

if __name__ =="__main__":
    p = 15
    n = 20
    x, y = gen_dataset(p, n)
    perceptron = Perceptron(n)
    perceptron.train(x, y)

    #gen_and_gate()
    #gen_or_gate()

