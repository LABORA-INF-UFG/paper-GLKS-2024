# GRAFICO DE LATÊNCIA FIM-A-FIM

import pandas as pd
import matplotlib.pyplot as plt 

def plot():
    df = pd.read_csv('../output/op3.csv', sep=';')
    df1 = pd.read_csv('../output/op1.csv', sep=';')
    df2 = pd.read_csv('../output/op2.csv', sep=';')

    int = 'Internet'
    wir = 'Wireless'

    link_capacity = 100#Mbits


    x = df['batch'].unique()
    internet_latency = (df.groupby('batch')[int].sum()/link_capacity)*1000
    wireless_latency = (df.groupby('batch')[wir].sum()/link_capacity)*1000
    y = internet_latency + wireless_latency

    x1 = df1['batch'].unique()
    internet_latency = (df1.groupby('batch')[int].sum()/link_capacity)*1000
    wireless_latency = (df1.groupby('batch')[wir].sum()/link_capacity)*1000
    y1 = internet_latency + wireless_latency

    x2 = df2['batch'].unique()
    internet_latency = (df2.groupby('batch')[int].sum()/link_capacity)*1000
    wireless_latency = (df2.groupby('batch')[wir].sum()/link_capacity)*1000
    y2 = internet_latency + wireless_latency

    fig,ax = plt.subplots()

    fig.set_size_inches(8, 4)
    ax.fill_between(x2, y2, label="OpCASH v2", color='green')
    ax.plot(x1, y1, color='white', linewidth=0.5)
    ax.fill_between(x1, y1, label="OpCASH v1", color='red')
    ax.plot(x, y, color='white', linewidth=0.5)
    ax.fill_between(x, y, label="DTMCash", color='#006bb3')
    ax.set_xlabel("Estampa de tempo", fontsize=26, labelpad=8)
    ax.set_ylabel("Latência (ms)",fontsize=26)


    ticksy = [0, 10, 20, 30]
    ticksx = [0, 38, 76, 114, 152, 190]

    ax.set_yticks(ticksy)
    ax.set_yticklabels(ticksy, fontsize=26)

    ax.set_xticks(ticksx)
    ax.set_xticklabels(ticksx, fontsize=26)

    ax.legend(fontsize=20)
    ax.grid(b=True, which='major', linestyle='--')
    plt.savefig("latency.pdf", bbox_inches='tight')
    plt.savefig("latency.png", bbox_inches='tight')


if __name__ == "__main__":
    plot()