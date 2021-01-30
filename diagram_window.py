import tkinter as tk
import os


class WiskerWindow(tk.Toplevel):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.title("Диаграмма Бокса-Вискерса")

        self.Box_Visc()

        self.im = tk.PhotoImage(file='pic.png')
        self.label = tk.Label(self, image=self.im)
        self.label.pack()

        self.grab_set()
        self.focus_set()
        os.remove("pic.png")

    def Box_Visc(self):
        plot = self.data.assign(index=self.data.groupby('ocean_proximity').
                                cumcount()).pivot('index', 'ocean_proximity','median_house_value').plot(kind='box')
        plot.figure.savefig('pic.png')
