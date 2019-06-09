import matplotlib.pyplot as plt
import tkinter as tk


def grad(data):
    f = plt.figure()
    data.plot.scatter(x='longitude',
                      y='latitude',
                      c='ocean_proximity',
                      colormap='viridis', figsize=(10, 5))
    plt.show()

    f.savefig("foo.pdf", bbox_inches='tight')

    root = tk.Tk()
    img = ImageTk.PhotoImage("foo.pdf")
    panel = tk.Label(root, image=img)
    panel.pack(side="bottom", fill="both", expand="yes")
    root.mainloop()
