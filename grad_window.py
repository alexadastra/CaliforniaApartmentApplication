import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tkinter as tk
import os


class GradationWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.title("Диаграмма рассеивания")

        self.grad()

        self.im = tk.PhotoImage(file='pic.png')
        self.label = tk.Label(self, image=self.im)
        self.label.pack()

        self.grab_set()
        self.focus_set()
        os.remove("pic.png")

    def grad(self):
        self.master.dataset.ocean_proximity, value = self.master.dataset.ocean_proximity.factorize()
        california_img = mpimg.imread('C:\california.png')
        image = self.master.dataset.plot.scatter(x='longitude', y='latitude', s=self.master.dataset['population'] / 100, label='Population',
                                  alpha=0.8, c='ocean_proximity', colormap='jet', figsize=(10, 5))
        plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5, cmap=plt.get_cmap('jet'))
        plt.ylabel("Latitude", fontsize=14)
        plt.xlabel("Longitude", fontsize=14)
        plt.title("Распределение классов")
        image.figure.savefig('pic.png')
