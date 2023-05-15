import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot_as(scores):
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    plt.title('Training...')
    plt.xlabel('Number of Games')
    
    plt.ylabel('Move Taken')
    plt.plot(scores, label = 'Move taken/ game')


    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    
    plt.show(block=False)
    plt.pause(.1)