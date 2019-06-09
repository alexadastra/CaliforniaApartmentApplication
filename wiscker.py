import matplotlib.pyplot as plt
import tkinter as tk


def wiscer(data):
    f = plt.figure()
    data.boxplot(['median_house_value', 'ocean_proximity'])
    plt.show()

    f.savefig("foo.pdf", bbox_inches='tight')

    root = tk.Tk()
    img = ImageTk.PhotoImage("foo.pdf")
    panel = tk.Label(root, image=img)
    panel.pack(side="bottom", fill="both", expand="yes")
    root.mainloop()
