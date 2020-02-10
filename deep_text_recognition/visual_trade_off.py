import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def trade_off(csv, lower, upper):
    data = pd.read_csv(csv)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(data['epoch'][lower:upper], data['acc'][lower:upper], label='accuracy')
    ax2.plot(data['epoch'][lower:upper], data['train_loss'][lower:upper], label='train_loss')
    ax2.plot(data['epoch'][lower:upper], data['valid_loss'][lower:upper], label='valid_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    ax1.legend()
    ax2.legend()
    plt.show()
    plt.savefig("R0114_FT_b32e_10k_log.png")


trade_off("./train_log/R0114_log.csv", 0, -1)