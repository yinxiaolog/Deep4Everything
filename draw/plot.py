import matplotlib.pyplot as plt


def plot(X, Y, xlabel, ylabel, labels):
    fig, ax = plt.subplots(dpi=1000)
    for i in range(len(Y)):
        ax.plot(X, Y[i], label=labels[i])
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.show()
    # fig.savefig(ylabel + '.png')
