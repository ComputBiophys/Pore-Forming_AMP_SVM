import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as col
import matplotlib.pyplot as plt

def plot_dimer(id_list):
    for pdbid in id_list:
        a = np.loadtxt('./result/' + pdbid + '_CM.txt')
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(3.28, 3)
        startcolor = 'w'
        midcolor = 'violet'
        cmap = col.LinearSegmentedColormap.from_list('own',[startcolor,midcolor])
        l = np.shape(a)[0]
        b = a[:int(l/2),int(l/2):int(l)]
        probability = []
        for k in range(int(l/2)):
            for p in range(int(l/2)):
                probability.append(b[k][p])
        probability.sort(reverse = True)
        for p in range(int(l/2)):
            for g in range(int(l/2)):
                if b[p][g] < probability[int(l)]:
                    b[p][g] = 0
        plt.matshow(b, cmap = cmap)
        plt.colorbar()
        plt.tick_params(labelsize = 15)
        plt.ylabel('Residue index',fontsize = 20)
        plt.xlabel('Residue index',fontsize = 20)
        ax = plt.gca()
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.gca().invert_yaxis()
        plt.savefig('./result/' + pdbid + '_CM.png', format = 'png', transparent = True, dpi = 300, pad_inches = 0, bbox_inches = 'tight', facecolor = 'white')
        
list1 = sys.argv[1]
id_list = list1.split()
plot_dimer(id_list)
