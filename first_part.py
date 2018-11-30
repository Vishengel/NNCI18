import numpy as np
random_number = 2018
np.random.seed(random_number)

class Perceptron:
    
    def __init__(self, n):
        self.n = n
        self.weights = [0] * n
        
    def train(self, x, y, n_epochs=100):
        for epoch in range(n_epochs):
            for pattern in x:
                
# generating dataset
def gen_dataset(p=5, n=20):
    x = np.random.normal(0, 1, p*n)
    x.shape = (p,n)
    y = np.random.choice([-1,1], p, True)

    return x,y

if __name__ =="__main__":
    p = 15 
    n = 20
    x, y = gen_dataset(p, n)
    perceptron = Perceptron(n)
    
    print(y)
