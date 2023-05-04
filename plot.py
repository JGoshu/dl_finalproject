import matplotlib.pyplot as plt

# Function that is meant for plotting the results from the model at the very end of training
def plot(values,ylabel, save_fig):
    plt.xlabel("Number of Runs")
    plt.ylabel(ylabel)
    plt.plot(values)
    plt.savefig(f"{save_fig}.png")
    plt.show()