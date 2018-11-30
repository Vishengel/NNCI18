import numpy as np
random_number = 2018
np.random.seed(random_number)

# generating dataset
def gen_dataset(p,n):
    n=10
    p=5
    x = np.random.normal(0, 1, p*n)
    x.shape=(p,n)
    y=np.random.binomial(1,0.5,p)
    return x,y

