import matplotlib as mpl
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler
#matplotlib.matplotlib_fname()

def init_params():
    mlp_default_params = plt.rcParams.copy()  

    colors = cycler('color',
                    ['#EE6666', '#3388BB', '#9988DD',
                     '#EECC55', '#88BB44', '#FFBBBB'])
    #####plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True, prop_cycle=colors)
    #####plt.rc('grid', color='w', linestyle='solid')
    #mpl.rcParams['xtick.major.size'] = 10
    #plt.rc('xtick',labelsize=8)
    #plt.rc('ytick',labelsize=8)
    plt.rc('xtick', labelsize=16)#, direction='out', color='gray')
    plt.rc('ytick', labelsize=16)#, direction='out', color='gray')
    #####plt.rc('patch', edgecolor='#E6E6E6')
    plt.rc('lines', linewidth=2)

    plt.rc('figure', figsize=(16,8))
    #plt.rcParams["figure.figsize"] = (20,23)
    #plt.figure(figsize=(12, 6))

    plt.rc('figure', titlesize='large')
    plt.rc('axes', titlesize='large')
    plt.rcParams.update({'figure.titlesize': 'large'})
    plt.rcParams.update({'axes.titlesize': 'large'})
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['figure.titlesize'] = 20
    #plt.xlabel('Degrees', fontsize=16);
    #plt.ylabel('Number of nodes', fontsize=16);


    # colori

    colors_to_cycle = get_colors_to_cycle_rainbow8()
    plt.rc('axes', prop_cycle=(cycler('color', colors_to_cycle)))



def get_colors_to_cycle_rainbow8():
    n = 8
    order = np.array([6, 0, 3, 1, 7, 5, 2, 4]) + 0.1
    return [plt.get_cmap('gist_rainbow')(1. * i/n) for i in order]

def get_colors_to_cycle_rainbowN(n):
    order = np.arange(n)
    return [plt.get_cmap('gist_rainbow')(1. * i/n) for i in order]

def get_colors_to_cycle_sequential(n):
    order = np.arange(n)
    return [plt.get_cmap('Spectral')(1. * i / n) for i in order] # 'winter'