# GRAFICOS DE OCUPAÇÃO DA CACHE

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

def read_data(file_name):
    data = []
    # Using readlines()
    file = open(file_name, 'r')
    lines = file.readlines()

    count = 0
    
    for line in lines:
        count += 1
        data.append(float(line.strip()))
        
    return data
    

    
def plot():
    
    data1 = read_data('../output/cache_op1.txt')
    data2 = read_data('../output/cache_op2.txt')
    data3 = read_data('../output/cache_op3.txt')

    x = list(range(1, len(data1)+1))

    fig,ax = plt.subplots()

    fig.set_size_inches(8, 6)
    ax.plot(x, data3, "-", label="DTMCash", linewidth=2.0, markersize=12, color='#006bb3')
    ax.plot(x, data1, "-", label="OpCASH v1", linewidth=2.0, markersize=12, color='red')
    ax.plot(x, data2, "-", label="OpCASH v2", linewidth=2.0, markersize=12, color='green')
    ax.set_xlabel("Estampa de tempo", fontsize=26, labelpad=8)
    ax.set_ylabel("Ocupação da cache (Mb)",fontsize=26)

    ticksx = [0, 38, 76, 114, 152, 190]
    ax.set_xticks(ticksx)
    ax.set_xticklabels(ticksx, fontsize=26)

    ticksy = [0, 5, 10, 15]
    ax.set_yticks(ticksy)
    ax.set_yticklabels(ticksy, fontsize=26)

    ax.legend(fontsize=20)
    ax.grid(axis='y', which='major', linestyle='--')
    plt.savefig("cache.pdf", bbox_inches='tight')
    plt.savefig("cache.png", bbox_inches='tight')


if __name__ == "__main__":
    plot()