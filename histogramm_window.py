import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tkinter as tk
import os
import pandas as pd

class HistogrammWindow(tk.Toplevel):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.title("Диаграмма рассеивания")

        self.bar(self.data)

        self.im = tk.PhotoImage(file='pic.png')
        self.label = tk.Label(self, image=self.im)
        self.label.pack()

        self.grab_set()
        self.focus_set()
        os.remove("pic.png")

    def bar(self, data):
        df = pd.DataFrame({'lab': data.ocean_proximity.unique(), 'val': data.ocean_proximity.value_counts()})
        ax = df.plot.bar(x='lab', y='val', rot=0)
        ax.figure.savefig('pic.png')
        ax.plot()
        return None
