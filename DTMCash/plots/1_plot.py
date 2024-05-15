# GRAFICOS DE OCUPAÇÃO DE LINKS

import pandas as pd
import matplotlib.pyplot as plt 

def plot(int_or_wir):
    df = pd.read_csv('../output/op3.csv', sep=';')
    df1 = pd.read_csv('../output/op1.csv', sep=';')
    df2 = pd.read_csv('../output/op2.csv', sep=';')

    x = df['batch'].unique()
    x = [i + 1 for i in x]
    y = df.groupby('batch')[int_or_wir].sum()

    x1 = df1['batch'].unique()
    x1 = [i + 1 for i in x1]
    y1 = df1.groupby('batch')[int_or_wir].sum()

    x2 = df2['batch'].unique()
    x2 = [i + 1 for i in x2]
    y2 = df2.groupby('batch')[int_or_wir].sum()

    fig,ax = plt.subplots()

    fig.set_size_inches(10, 6)
    ax.plot(x1, y1, "-", label="OpCASH v1", linewidth=1.0, markersize=26, color='red')
    ax.plot(x2, y2, "-", label="OpCASH v2", linewidth=1.0, markersize=26, color='green')
    ax.plot(x, y, "-", label="DTMCash", linewidth=1.0, markersize=26, color='#006bb3')
    ax.set_xlabel("Estampa de tempo", fontsize=26, labelpad=8)
    ax.set_ylabel("Ocupação: " + int_or_wir.lower() + " (Mbps)",fontsize=26)


    if int_or_wir == 'Internet':
        ticksy = [0, 0.2, 0.4, 0.6, 0.8, 1]
    elif int_or_wir == 'Wireless':
        ticksy = [0, 0.6, 1.2, 1.8, 2.4, 3]
        
    ticksx = [0, 38, 76, 114, 152, 190]

    ax.set_yticks(ticksy)
    ax.set_yticklabels(ticksy, fontsize=26)

    ax.set_xticks(ticksx)
    ax.set_xticklabels(ticksx, fontsize=26)

    ax.legend(fontsize=20)
    ax.grid(b=True, which='major', linestyle='--')
    plt.savefig("data_" + int_or_wir + ".pdf", bbox_inches='tight')
    plt.savefig("data_" + int_or_wir + ".png", bbox_inches='tight')


if __name__ == '__main__':

    plot('Internet')
    plot('Wireless')