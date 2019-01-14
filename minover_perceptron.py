import math
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
        stop_criterion = 0.001
        max_epochs = n_epochs * len(x)

        # continue until the max n epochs is reached or all error terms are above 0
        while epoch < max_epochs and not success:
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

            # store the old weights
            old_weights = self.weights
            # update the weights by adding the term to the current weights
            self.weights = np.add(self.weights, weight_addition_term)

            # calculate the angle in radians between the old and the new weights
            angle = math.acos(np.dot(old_weights, self.weights) / (np.linalg.norm(old_weights) * np.linalg.norm(self.weights)))

            # if the angle between the old and new weights is sufficiently small, stop training
            if angle < stop_criterion:
                success = True

            epoch += 1
            #print("Weights: ", self.weights)

        if epoch == max_epochs:
            print("Max epochs reached")
            print("Weights: ", self.weights)
        elif success:
            print("Success in ", epoch, " epochs!")

        generalization_error = np.dot(1/np.pi,
                                      np.arccos(np.dot(np.dot(self.weights,self.w_star),
                                                       (1/np.linalg.norm(self.weights)) *
                                                       (1/np.linalg.norm(self.w_star)))))
        print('Generalization error: {}'.format(generalization_error))

        return success, generalization_error

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


if __name__ == "__main__":

    # test_single_run()

    n = 20  # number of dimensions
    w_star = n * [1]
    nd = 2  # number of datasets
    alpha = np.linspace(0.75, 3, 3)
    p = alpha * n  # number of examples
    generalization_error = np.zeros(len(p))  # generalization errors for each p value

    for j in range(0, len(p)):  # first loop over values of p
        for i in range(0, nd):  # second loop over different datasets
            x, y = gen_dataset(w_star, int(p[j]), n)
            perceptron = Perceptron(n, w_star)
            generalization_error[j] += generalization_error[j] + (1/nd)*perceptron.train(x, y)[1]
        print("Value of p:", p[j],
              "\n Average generalization error: {}".format(generalization_error[j]))


    # Plotting chart
    trace = go.Scatter(x=alpha, y=generalization_error)
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