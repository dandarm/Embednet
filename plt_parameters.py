import matplotlib as mpl
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

    #plt.xlabel('Degrees', fontsize=16);
    #plt.ylabel('Number of nodes', fontsize=16);