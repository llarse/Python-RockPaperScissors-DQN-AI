import matplotlib.pyplot as plt

def plotWinRate(rounds, scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label = "1")
    ax2 = fig.add_subplot(111, label = "2", frame_on = False)
    
    ax.plot(rounds, epsilons, color = "C0")
    ax.set_xlabel("Game", color = "C0")
    ax.set_ylabel("Epsilon", color = "C0")
    ax.tick_params(axis = 'x', colors = "C0")
    ax.tick_params(axis = 'y', color = "C0")
    
    ax2.scatter(rounds, scores, color = "C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color = "C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', color = "C1")
    
    plt.savefig(filename)