import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import messagebox

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, accuracy_score, precision_score


class AddingWindow(tk.Toplevel):
    def __init__(self, master, window):
        super().__init__()
        self.master = master
        self.window = window
        self.flag_prediction1 = True
        self.flag_prediction2 = True

        self.longitude_label = tk.Label(self, width=30, text='Введите параметр longitude', anchor='w')
        self.longitude_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 0]), to=np.max(self.master.dataset.iloc[:, 0]))

        self.latitude_label = tk.Label(self, width=30, text='Введите параметр latitude', anchor='w')
        self.latitude_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 1]), to=np.max(self.master.dataset.iloc[:, 1]))

        self.housing_median_age_label = tk.Label(self, width=30, text='Введите параметр housing_median_age', anchor='w')
        self.housing_median_age_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 2]), to=np.max(self.master.dataset.iloc[:, 2]))

        self.total_rooms_label = tk.Label(self, width=30, text='Введите параметр total_rooms', anchor='w')
        self.total_rooms_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 3]), to=np.max(self.master.dataset.iloc[:, 3]))

        self.total_bedrooms_label = tk.Label(self, width=30, text='Введите параметр total_bedrooms', anchor='w')
        self.total_bedrooms_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 4]), to=np.max(self.master.dataset.iloc[:, 4]))

        self.population_label = tk.Label(self, width=30, text='Введите параметр population', anchor='w')
        self.population_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 5]), to=np.max(self.master.dataset.iloc[:, 5]))

        self.households_label = tk.Label(self, width=30, text='Введите параметр households', anchor='w')
        self.households_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 6]), to=np.max(self.master.dataset.iloc[:, 6]))

        self.median_income_label = tk.Label(self, width=30, text='Введите параметр median_income', anchor='w')
        self.median_income_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 7]), to=np.max(self.master.dataset.iloc[:, 7]))

        self.median_house_value_label = tk.Label(self, width=30, text='Введите параметр median_house_value', anchor='w')
        self.median_house_value_entry = tk.Spinbox(self, width=30, from_=np.min(self.master.dataset.iloc[:, 9]), to=np.max(self.master.dataset.iloc[:, 9]))

        self.ocean_proximity_label = tk.Label(self, width=30, text='Введите параметр ocean_proximity', anchor='w')
        self.ocean_proximity_entry = tk.Listbox(self, width=30, height=6, selectmode=tk.SINGLE)

        self.price_predict = tk.Checkbutton(self, text='Хотите ли вы, чтобы цена квартиры прогнозировалась '
                                                       'автоматически?', variable=self.flag_prediction1)

        self.class_predict = tk.Checkbutton(self, text='Хотите ли вы, чтобы зона квартиры прогнозировалась '
                                                       'автоматически?', variable=self.flag_prediction2)

        self.Button = tk.Button(self, text='Добавить', command=self.apply)

        self.init_child()

    def init_child(self):
        classes = ['INLAND', 'NEAR BAY', '<1H OCEAN', 'NEAR OCEAN', 'ISLAND']
        self.title('Добавить запись')
        self.geometry('460x365+100+100')
        self.resizable(False, False)

        self.longitude_label.grid(row=0, column=0)
        self.longitude_entry.grid(row=0, column=1)

        self.latitude_label.grid(row=1, column=0)
        self.latitude_entry.grid(row=1, column=1)

        self.housing_median_age_label.grid(row=2, column=0)
        self.housing_median_age_entry.grid(row=2, column=1)

        self.total_rooms_label.grid(row=3, column=0)
        self.total_rooms_entry.grid(row=3, column=1)

        self.total_bedrooms_label.grid(row=4, column=0)
        self.total_bedrooms_entry.grid(row=4, column=1)

        self.population_label.grid(row=5, column=0)
        self.population_entry.grid(row=5, column=1)

        self.households_label.grid(row=6, column=0)
        self.households_entry.grid(row=6, column=1)

        self.median_income_label.grid(row=7, column=0)
        self.median_income_entry.grid(row=7, column=1)

        self.median_house_value_label.grid(row=8, column=0)
        self.median_house_value_entry.grid(row=8, column=1)

        self.ocean_proximity_label.grid(row=9, column=0)
        for i in range(len(classes)):
            self.ocean_proximity_entry.insert(tk.END, classes[i])
        self.ocean_proximity_entry.grid(row=9, column=1)

        self.price_predict.grid(row=10, column=0, columnspan=2, sticky='w')

        self.class_predict.grid(row=11, column=0, columnspan=2, sticky='w')

        self.Button.grid(row=12, column=0, columnspan=2, ipadx=200)

        self.grab_set()
        self.focus_set()

    def apply(self):
        flag = True
        dict = {
            0: self.longitude_entry.get(),
            1: self.latitude_entry.get(),
            2: self.housing_median_age_entry.get(),
            3: self.total_bedrooms_entry.get(),
            4: self.total_bedrooms_entry.get(),
            5: self.population_entry.get(),
            6: self.households_entry.get(),
            7: self.median_income_entry.get(),
            9: self.median_house_value_entry.get(),
        }
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 9]:
            try:
                a = float(dict[i])
            except ValueError:
                flag = False
                messagebox.showinfo("Ошибка!", "Неверный тип переменной " + str(self.master.dataset.columns[i]) + "!")

        if len(self.ocean_proximity_entry.curselection()) != 0 and not(self.flag_prediction2):
            flag = False
            messagebox.showinfo("Ошибка!", "Не указан параметр ocean_proximity_entry!")

        if flag:
            data_frame = self.appending(dict)
            self.master.dataset = self.master.dataset.append(data_frame, ignore_index=True)
            matrix = data_frame.values
            self.window.table.insert('', 'end', text=str((len(self.master.dataset.longitude) - 1)),
                                     values=(matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3], matrix[0][4], matrix[0][5],
                                     matrix[0][6], matrix[0][7], matrix[0][8], matrix[0][9]))

    def appending(self, dict):
        ocean_proximity_array = self.master.dataset.ocean_proximity.unique()
        auxiliary_array = np.array([])
        columns_of_dataset = self.master.dataset.columns

        for iteration in np.arange(self.master.dataset.shape[1]):
            # type_of_prev_string_of_data = type(dataset.iloc[-1, iteration])  # тип переменной i - ого столбца последней строки
            # if type(self.master.dataset.iloc[:, iteration][0]) != str and self.master.dataset.iloc[:,
            #                                                   iteration].mean() > 100000 and self.flag_prediction1:
            if columns_of_dataset[iteration] == 'median_house_value' and self.flag_prediction1:
                # проверка на то, что найден именно столбец median_house_value
                X = self.master.dataset.drop(['median_house_value', 'ocean_proximity'], axis=1)  # важно
                y = self.master.dataset.median_house_value  # важно
                auxiliary_sub_array = auxiliary_array[:-1].astype(float)  # важно
                weights = regression_weights(X, y)  # важно
                variable = int(predict((auxiliary_sub_array), regression_weights(X, y)))  # важно
            elif columns_of_dataset[iteration] == 'ocean_proximity' and self.flag_prediction2:  # важно
                auxiliary_sub_array = [auxiliary_array[0:2]]  # важно
                variable = classificator(self.master.dataset, auxiliary_sub_array)  # важно
            else:
                variable = dict[iteration]

            auxiliary_array = np.append(auxiliary_array, variable)

        df = pd.DataFrame(columns=self.master.dataset.columns)
        df.loc[len(self.master.dataset)] = [float(auxiliary_array[0]), float(auxiliary_array[1]), float(auxiliary_array[2]),
                                float(auxiliary_array[3]),
                                float(auxiliary_array[4]), float(auxiliary_array[5]), float(auxiliary_array[6]),
                                float(auxiliary_array[7]), (auxiliary_array[8]), float(auxiliary_array[9])]

        return df


def class_balancing(dataset):
    helped_dataset = dataset.copy()
    helped_dataset.ocean_proximity, d = helped_dataset.ocean_proximity.factorize()
    Series = pd.Series(helped_dataset.ocean_proximity)
    maximum = Series.value_counts().max()

    X = pd.DataFrame(columns=['longitude', 'latitude', 'ocean_proximity']).values
    for i in np.arange(len(Series.unique())):
        helped_2 = Series.unique()[i]
        array = helped_dataset[helped_dataset['ocean_proximity'] == helped_2].loc[:,
                ['longitude', 'latitude', 'ocean_proximity']].values.reshape(-1, 3)
        help_3 = array.copy()
        while len(array) < maximum:
            if maximum - len(array) >= len(help_3):
                array = np.append(array, help_3, axis=0)
            else:
                array = np.append(array, help_3[0: maximum - len(help_3)], axis=0)
        X = np.append(X, array, axis=0)

    X = pd.DataFrame(X, columns=['longitude', 'latitude', 'ocean_proximity'])
    return X


def classificator(dataset, array):
    balanced_dataset = class_balancing(dataset)
    variable = len(balanced_dataset)

    value = dataset.ocean_proximity.unique()
    y = pd.DataFrame(balanced_dataset).iloc[:, 2].values.reshape((variable,)).astype(int)
    X = balanced_dataset.drop('ocean_proximity', axis=1).values.astype(float)

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(n_estimators=500, max_depth=20)
    clf.fit(X_train, y_train)
    prediction = clf.predict(array)
    y_pred = clf.predict(X_test)
    z = pd.Series(y).unique()
    for i in range(len(value)):
        if prediction == z[i]:
            prediction = value[i]
    return prediction


def regression_weights(X, y):
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    weights = np.linalg.inv(X.T @ X) @ X.T @ y.values
    return weights


def predict(X, weights):  # Не работает - доделать
    B = weights[0]
    weights = weights[1: len(weights)]
    predictions = B + X @ weights.T
    return predictions


def score(y_test, y_pred):
    d = np.abs(y_test - y_pred)
    epsilons = list(map(float, input().split()))
    scores = np.array([])
    for i in np.arange(len(epsilons)):
        summary = 0
        for j in np.arange(len(d)):
            if d[j] <= epsilons[i]:
                summary += 1
        score = summary / len(d)
        scores = np.append(scores, score)
    scoresDF = pd.DataFrame({"Epsilon": epsilons, "Score": scores})

    return scoresDF


def make_DataFrame(dataset):  # Создает DataFrame, состоящий из 1 строки
    ocean_proximity_array = dataset.ocean_proximity.unique()
    auxiliary_array = np.array([])
    columns_of_dataset = dataset.columns
    operations = ('y', 'n')
    dictionary = {'longitude': 'числом',
                  'latitude': 'числом',
                  'housing_median_age': 'числом',
                  'total_rooms': 'числом',
                  'total_bedrooms': 'числом',
                  'population': 'числом',
                  'households': 'числом',
                  'median_income': 'числом',
                  'median_house_value': 'числом',
                  'ocean_proximity': 'строкой'}
    while True:   # хуйня
        print("Хотите ли вы, чтобы цена квартиры прогнозировалась автоматически? y / n")  # хуйня
        operation = str(input())  # хуйня
        if operation in operations:  # хуйня
            if (operation == 'y'):  # хуйня
                flag_prediction1 = True  # хуйня
            else:  # хуйня
                flag_prediction1 = False  # хуйня
            break  # хуйня
        else:  # хуйня
            continue  # хуйня
    while True:  # хуйня
        print("Хотите ли вы, чтобы зона квартиры прогнозировалась автоматически? y / n")  # хуйня
        operation = str(input())  # хуйня
        if operation in operations:  # хуйня
            if (operation == 'y'):  # хуйня
                flag_prediction2 = True  # хуйня
            else:  # хуйня
                flag_prediction2 = False  # хуйня
            break  # хуйня
        else:  # хуйня
            continue  # хуйня
    for iteration in np.arange(dataset.shape[1]):  # пока не заполнены все поля строки данных - выполняем:
        type_of_prev_string_of_data = type(
            dataset.iloc[-1, iteration])  # тип переменной i - ого столбца последней строки
        while True:  # бесконечный цикл
            if type(dataset.iloc[:, iteration][0]) != str and dataset.iloc[:,
                                                              iteration].mean() > 100000 and flag_prediction1:
                # проверка на то, что найден именно столбец median_house_value
                X = dataset.drop(['median_house_value', 'ocean_proximity'], axis=1)  # важно
                y = dataset.median_house_value  # важно
                auxiliary_sub_array = auxiliary_array[:-1].astype(float)  # важно
                weights = regression_weights(X, y)  # важно
                variable = int(predict((auxiliary_sub_array), regression_weights(X, y)))  # важно
                break
            else:
                if columns_of_dataset[iteration] == 'ocean_proximity' and flag_prediction2:  # важно
                    auxiliary_sub_array = [auxiliary_array[0:2]]  # важно
                    variable = classificator(dataset, auxiliary_sub_array)  # важно
                    break
                else:  # хуйня
                    print(  # хуйня
                        "Введите значение для столбца {}, оно должно быть {}".format(columns_of_dataset[iteration],  # хуйня
                                                                                     dictionary[columns_of_dataset[  # хуйня
                                                                                         iteration]]))  # хуйня
                    try:  # хуйня
                        variable = type_of_prev_string_of_data(input())  # хуйня
                    except ValueError:  # если переменная другого типа - вводим значение снова
                        print("Переменная не соответствует типу столбца, введите новое значение:")  # хуйня
                        continue  # хуйня
                    else:  # если значение переменной совпадает с нужным - завершаем цикл
                        if type_of_prev_string_of_data is str:
                            # если введенное пользователем значение существует - завершаем цикл
                            if variable in ocean_proximity_array:
                                break
                            else:
                                print("Введите значение из списка предложенных: {}".format(ocean_proximity_array))
                                continue
                        break
        # после обработки ошибки ввода добавляем значение в массив
        auxiliary_array = np.append(auxiliary_array, variable)
    df = pd.DataFrame(columns=dataset.columns)
    df.loc[len(dataset)] = [float(auxiliary_array[0]), float(auxiliary_array[1]), float(auxiliary_array[2]),
                            float(auxiliary_array[3]),
                            float(auxiliary_array[4]), float(auxiliary_array[5]), float(auxiliary_array[6]),
                            float(auxiliary_array[7]), auxiliary_array[8], float(auxiliary_array[9])]
    return df