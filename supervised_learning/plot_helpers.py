import matplotlib.pyplot as plt

def plot_accuracy(pred, truth):
    #flatten the array inputs
    pred = pred.flatten()
    truth = truth.flatten()
    plt.plot(pred, label='predicted')
    plt.plot(truth, label='truth')

    # plt.set(xlabel='Index', ylabel='classification value',title='Accuracy')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
