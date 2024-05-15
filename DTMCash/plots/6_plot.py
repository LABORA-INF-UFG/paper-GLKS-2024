# GRAFICO DE OCUPAÇÃO ACUMULADA DOS LINKS

import pandas as pd
import matplotlib.pyplot as plt 

def plot(int_or_wir):
    df = pd.read_csv('../output/op3.csv', sep=';')
    df1 = pd.read_csv('../output/op1.csv', sep=';')
    df2 = pd.read_csv('../output/op2.csv', sep=';')

    index_s = [1, 39, 77, 115, 153, 190]
    index_sm1 = [0, 38, 76, 114, 152, 189]


    y = df.groupby('batch')[int_or_wir].sum()
    y = y.cumsum()

    y1 = df1.groupby('batch')[int_or_wir].sum()
    y1 = y1.cumsum()

    y2 = df2.groupby('batch')[int_or_wir].sum()
    y2 = y2.cumsum()


    fig,ax = plt.subplots()

    fig.set_size_inches(4, 6)

    ax.plot(index_s, y1.loc[index_sm1], 'v', label="OpCASH v1", color='red', markerfacecolor='none', markeredgecolor='red', markersize=12)

    ax.plot(index_s, y2.loc[index_sm1], '^', label="OpCASH v2", color='green', markerfacecolor='none', markeredgecolor='green', markersize=12)

    ax.plot(index_s, y.loc[index_sm1],  's', label="DTMCash", color='#006bb3', markerfacecolor='none', markeredgecolor='#006bb3', markersize=12)

    ax.set_xlabel("Estampa de tempo", fontsize=26, labelpad=8)
    ax.set_ylabel("Ocup. acum.: " + int_or_wir.lower() + " (Mbps)",fontsize=26)


    ticksy = [0, 50, 100, 150, 200, 250]
    #ticksy = [0, 20, 40, 60, 80, 100]
    ticksx = [0, 38, 76, 114, 152, 190]

    ax.set_yticks(ticksy)
    ax.set_yticklabels(ticksy, fontsize=26)
    ax.yaxis.set_label_coords(-.25, .4)

    ax.set_xticks([1, 190])
    ax.set_xticklabels([1, 190], fontsize=26)

    ax.legend(fontsize=16)
    ax.grid(b=True, which='major', linestyle='--')
    plt.savefig("cumulative_" + int_or_wir + ".pdf", bbox_inches='tight')
    plt.savefig("cumulative_" + int_or_wir + ".png", bbox_inches='tight')


if __name__ == '__main__':

    plot('Internet')
    plot('Wireless')
