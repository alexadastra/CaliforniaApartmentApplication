import tkinter as tk
import tkinter.ttk as ttk

class SaveFile(tk.Toplevel):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.label = tk.Label(self, width=30, text='Введите имя файла').pack()
        self.entry = tk.Entry(self, width=30)
        self.entry.pack()
        self.button = tk.Button(self, command=self.new_csv, text='Создать файл').pack()
        self.grab_set()
        self.focus_set()

    def writing_to_text_file(self, dataset, file_name):
        dataset.to_csv(file_name, encoding='utf-8', index=False)

    def new_csv(self):
        string = str(self.entry.get()) + '.csv'
        with open(string, 'w'):
            pass
        self.writing_to_text_file(self.dataset, string)
