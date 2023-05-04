import matplotlib.pyplot as plt

# Function that is meant for plotting the results from the model at the very end of training
def plot(values, ylabel, save_fig):
    plt.xlabel("Number of Runs")
    plt.ylabel(ylabel)
    plt.plot(values)
    plt.savefig(f"{save_fig}.png")
    plt.show()

def plot_all_sentiments(values, save_fig):
    plt.xlabel("Number of Epochs")
    plt.ylabel("Level of Sentiment")
    for i in range(values.shape[0]):
        # assumes that values will be a 2d matrix, where each row represents a sentiment
        plt.plot(values[i])
    plt.savefig(f"{save_fig}.png")
    plt.show()