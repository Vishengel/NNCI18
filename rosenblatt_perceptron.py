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

            print("\n Training epoch: ", epoch)
            print("=====================")

            for idx in range(len(x)):
                # calculate the error E (weights * pattern * label)
                error = np.dot(self.weights, x[idx]) * y[idx]

                # debug statements
                # print("Weights: ", self.weights, " - x: ", x[idx], " - y: ", y[idx])
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
        return success

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


if __name__ =="__main__":
    p = 15
    n = 20
    alpha=np.linspace(0.75,3,3)
    p=alpha*n
    nd = 2 # number of datasets
    success_ratios = np.zeros(len(p))
    for j in range(0,len(p)): # first loop over values of p
        for i in range(0,nd): # second loop over different datasets
            x, y = gen_dataset(int(p[j]), n)
            perceptron = Perceptron(n)
            success=perceptron.train(x, y)
            print("Success:",success)
            if success == True:
                success_ratios[j]+=1
        print("Value of p:",p[j],
              "\nFraction of successful runs:",success_ratios[j]/nd)

    #gen_and_gate()
    #gen_or_gate()

    # Plotting chart
    trace = go.Scatter(x=alpha, y=success_ratios/nd)
    layout = go.Layout(
        title='<b>{}</b>'.format('Dependence of successful runs ratio on alpha'),
        titlefont=dict(family='Open Sans', size=20),
        font=dict(family='Open Sans'),
        xaxis=dict(title='<i>{}</i>'.format('alpha'), titlefont=dict(size=16), tickfont=dict(size=12)),
        yaxis=dict(title='<i>{}</i>'.format('Success ratio'), titlefont=dict(size=16), tickfont=dict(size=12)),
        legend=dict(font=dict(size=16), orientation='v'),
    )

    fig = go.Figure(data=[trace], layout=layout)
    py.offline.plot(fig,
                    filename='rosenblatt_perceptron_test.html',
                    auto_open=False)