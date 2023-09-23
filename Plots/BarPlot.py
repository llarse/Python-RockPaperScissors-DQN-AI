import matplotlib.pyplot as plt

class BarPlot():
    def __init__(self, categories, initial_heights, x_label, y_label, title):
        self.categories = categories
        self.initial_heights = initial_heights
        self.heights = initial_heights
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        
        self.fig, self.ax = plt.subplots()
        self.bars = self.ax.bar(self.categories, self.initial_heights)
    
    def draw(self):
        self.update_heights()
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.show(block=False)
        plt.draw()
        plt.pause(0.001)
    
    def update_heights(self):
        for bar, height in zip(self.bars, self.heights):
            bar.set_height(height)
        plt.ylim(0, max(self.heights) + 1)