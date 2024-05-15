# GRAFICO DE CACHE HIT POR QUANTIDADE

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

def read_data(file_name):
    data = []
    # Using readlines()
    file = open(file_name, 'r')
    lines = file.readlines()


    
    for line in lines:
        att = float(line.split(' ')[3])
        req = float(line.split(' ')[4])
        data.append((att/req)*100)
        
    return data
    

    
def plot():
    
    data1 = read_data('../output/hits_op1.txt')
    data2 = read_data('../output/hits_op2.txt')
    data3 = read_data('../output/hits_op3.txt')
    

    x = list(range(1, len(data1)+1))


    fig,ax = plt.subplots()

    fig.set_size_inches(6, 6)
    ax.plot(x, data1, "s", label="OpCASH v1", linewidth=1.0, markersize=14, color='red')
    ax.plot(x, data2, "o", label="OpCASH v2", linewidth=1.0, markersize=12, color='green')
    ax.plot(x, data3, "o", label="DTMCash", linewidth=1.0, markersize=10, color='#006bb3')
    ax.set_xlabel("Estampa de tempo", fontsize=26, labelpad=8)
    ax.set_ylabel("Cache hit: quantidade (%)",fontsize=26)

    ticksx = [0, 38, 76, 114, 152, 190]
    ax.set_xticks(ticksx)
    ax.set_xticklabels(ticksx, fontsize=26)

    ticksy = [0, 25, 50, 75, 100]
    ax.set_yticks(ticksy)
    ax.set_yticklabels(ticksy, fontsize=26)

    ax.legend(fontsize=20)
    ax.grid(axis='y', which='major', linestyle='--')
    plt.savefig("hits_l.pdf", bbox_inches='tight')
    plt.savefig("hits_l.png", bbox_inches='tight')

if __name__ == "__main__":
    plot()