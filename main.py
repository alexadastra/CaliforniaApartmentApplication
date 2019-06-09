import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adding_window import AddingWindow
from searching_window import SearchingWindow
from sorting_window import SortingWindow
from changing_window import ChangingWindow
from writing_file_window import SaveFile
from static_report_window import StaticWindow
from text_report_window import TextWindow
from wiscker import wiscer
from grad import grad


class DataFrame(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)

        self.master = master
        self.master.title('Квартиры Калифорнии')
        self.master.geometry('1340x650+0+0')

        self.titles = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                       "population", "households", "median_income", "ocean_proximity", "median_house_value"]

        self.table = ttk.Treeview(self, height=30, selectmode=tk.EXTENDED)
        self.scrollbar = tk.Scrollbar(self)
        self.matrix = self.master.dataset.values
        self.design()

    def design(self):
        self.table["columns"] = ("longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                                 "population", "households", "median_income", "ocean_proximity", "median_house_value")
        self.table.heading('#0', text='number')
        self.table.column('#0', width=100, stretch=tk.YES)
        for i in range(10):
            self.table.heading(str(self.titles[i]), text=self.titles[i])
            self.table.column(self.titles[i], width=100, stretch=tk.YES)

        for i in range(len(self.master.dataset.longitude)):
            cells = []
            for j in range(len(self.titles)):
                try:
                    cells.append(self.matrix[i][j])
                except IndexError:
                    break
            tuple(cells)
            self.table.insert('', 'end', text=str(i), values=cells)

        self.table.grid(row=0, column=0)
        self.table.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.table.yview)
        self.scrollbar.grid(row=0, column=1, ipady=300)


class FunctionFrame(tk.Frame):
    def __init__(self, master, prev):
        super().__init__(master)

        self.master = master
        self.prev = prev

        self.label_func = tk.Label(self, width=30, text='Функции')

        self.add_button = tk.Button(self, width=30, text=u"Добавить запись")
        self.delete_button = tk.Button(self, width=30, text=u"Удалить запись")
        self.search_button = tk.Button(self, width=30, text=u"Искать записи")
        self.sort_button = tk.Button(self, width=30, text=u"Сортировать записи")
        self.change_button = tk.Button(self, width=30, text=u"Изменить запись")
        self.save_button = tk.Button(self, width=30, text=u"Сохранить бвзу данных")

        self.label_reports = tk.Label(self, width=30, text='Отчёты')
        self.text_report_button = tk.Button(self, width=30, text=u"Простой текстовый отчет")
        self.static_report_button = tk.Button(self, width=30, text=u"Текстовый статистический отчет")
        self.wiscker_button = tk.Button(self, width=30, text=u"Диаграмма Бокса-Вискера")
        self.grad_button = tk.Button(self, width=30, text=u"Диаграмма рассеивания")

        self.design()

    def design(self):
        self.label_func.pack()

        self.add_button.config(command=self.adding)
        self.add_button.pack()

        self.change_button.config(command=self.changing)
        self.change_button.pack()

        self.delete_button.config(command=self.deleting)
        self.delete_button.pack()

        self.search_button.config(command=self.searching)
        self.search_button.pack()

        self.sort_button.config(command=self.sorting)
        self.sort_button.pack()

        self.save_button.config(command=self.saving)
        self.save_button.pack()

        self.label_reports.pack()

        self.text_report_button.config(command=self.text_report)
        self.text_report_button.pack()

        self.static_report_button.config(command=self.static_report)
        self.static_report_button.pack()

        self.wiscker_button.config(command=self.wiscker)
        self.wiscker_button.pack()

        self.grad_button.config(command=self.grad)
        self.grad_button.pack()

    def adding(self):
        AddingWindow(self.master, self.prev)

    def deleting(self):
        if len(self.prev.table.selection()) == 0:
            messagebox.showinfo("Ошибка!", "Не указаны строки под удаление")
        else:
            self.direct_deleting()

    def searching(self):
        SearchingWindow(self.master)

    def sorting(self):
        SortingWindow(self.master, self.prev)

    def changing(self):
        if len(self.prev.table.selection()) == 0:
            messagebox.showinfo("Ошибка!", "Не указана строка под изменение")
        else:
            ChangingWindow(self.master, self.prev)

    def direct_deleting(self):
        items = self.prev.table.selection()
        for i in items:
            self.prev.table.delete(i)
            self.master.dataset.drop((int(i[1:], 16) - 1), inplace=True)
        x = self.prev.table.get_children()
        i = 0
        for item in x:
            self.prev.table.item(item, text=str(i))
            i += 1

    def saving(self):
        SaveFile(self.master.dataset)

    def text_report(self):
        TextWindow(self.master)

    def static_report(self):
        StaticWindow(self.master)

    def wiscker(self):
        pass

    def grad(self):
        pass

root = tk.Tk()
root.dataset = pd.read_csv("C:\housing.csv")
data_frame = DataFrame(root)
data_frame.grid(row=0, column=0)
functions_frame = FunctionFrame(root, data_frame)
functions_frame.grid(row=0, column=1, sticky='nw')
root.mainloop()
