import matplotlib.pyplot as plt
import numpy as np

def draw_picture(mean, var, n):
    plt.figure(figsize=(15, 4))
    x = np.arange(-2, 2, 0.01)

    #draw the pictures for mean = 1, var = 9, n = 10
    plt.subplot(1, 3, 1)
    plt.ylim(0, 50)
    y = bias(x, mean)
    plt.plot(x, y)
    plt.xlabel("$lambda$")
    plt.ylabel("Bias")

    plt.subplot(1, 3, 2)
    plt.ylim(0, 50)
    y = variance(x, var, n)
    plt.plot(x, y)
    plt.xlabel("$lambda$")
    plt.ylabel("Variance")

    plt.subplot(1, 3, 3)
    plt.ylim(0, 50)
    y = expected_square_error(x, mean, var, n)
    plt.plot(x, y)
    plt.xlabel("$lambda$")
    plt.ylabel("Expected_square_error")

    plt.show()



def variance(x, var, n):
    variance = (var * var) / ((1 + x) * (1 + x) * n)
    return variance

def bias(x, mean):
    bias = (mean * mean * x * x) / ((1 + x) * (1 + x))
    return bias

def expected_square_error(x, mean, var, n):
    bias = (mean * mean * x * x) / ((1 + x) * (1 + x))
    variance = (var * var) / ((1 + x) * (1 + x) * n)
    return bias + variance

if __name__ == "__main__":
    draw_picture(1, 9, 5)