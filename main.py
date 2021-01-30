import tkinter as tk
from main_window import DataFrame, FunctionFrame

import pandas as pd

root = tk.Tk()
root.dataset = pd.read_csv("C:\housing.csv")
data_frame = DataFrame(root)
data_frame.grid(row=0, column=0)
functions_frame = FunctionFrame(root, data_frame)
functions_frame.grid(row=0, column=1, sticky='nw')
root.mainloop()
