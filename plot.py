import matplotlib.pyplot as plt

def plot(values, ylabel):
    """Plots the results from the model at the very end of training"""
    plt.xlabel("Number of Runs")
    plt.ylabel(ylabel)
    plt.plot(values)
    plt.savefig(f"{ylabel}.png")
    plt.show()

def plot_all_sentiments(values):
    """Plots the loss of all sentiments at the very end of training"""
    plt.xlabel("Number of Epochs")
    plt.ylabel("Level of Sentiment")
    for i in range(values.shape[0]):
        # assumes that values will be a 2d matrix, where each row represents a sentiment
        plt.plot(values[i])
    plt.savefig("LOSS.png")
    plt.legend(["Anger", "Anticipation", "Joy", "Trust", "Fear", "Sadness", "Disgust"])
    plt.show()