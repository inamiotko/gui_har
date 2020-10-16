import tkinter as tk
from tkinter import filedialog
import tkinter.ttk as ttk
import pandas as pd
import os
from keras.models import load_model
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from collections import Counter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.label_representation = tk.Label(window, text="Human Activity data representation", width=40, height=2,
                                             fg="black",
                                             bg="lightgrey", font="-weight bold")
        self.label_recognition = tk.Label(window, text="Human Activity Recognition", width=40, height=3, fg="black",
                                          bg="lightgrey",
                                          wraplength=500,
                                          font="-weight bold")

        self.label_files = tk.Label(window, text="Choose signal to upload", width=45, height=2, fg="blue")
        self.button1 = tk.Button(text="Browse", width=10, bg="grey", fg="black", command=self.browse_files)
        self.label_files_img = tk.Label(window, text="Choose signal to upload", width=45, height=2, fg="blue")
        self.button3 = tk.Button(text="Browse", width=10, height=2, bg="grey", fg="black",
                                 command=self.browse_files_graph)
        self.filename = ''
        choices = ['First model', 'Second model']
        self.chooser = ttk.Combobox(window)
        self.chooser['state'] = 'disabled'
        self.graphs_chooser = ttk.Combobox(window)
        self.graphs_chooser['state'] = 'disabled'
        self.chooser['values'] = choices
        self.chooser.current(0)
        self.button2 = tk.Button(text="Execute", width=10, height=2, bg="grey", fg="black", command=self.chosen_model)
        self.label_results = tk.Label(window, text="Recognised human activity:  ", width=50, height=5, fg="black")
        self.dataset = []
        self.graphs_chooser.place(x=510, y=255)
        self.label_recognition.place(x=20, y=10)
        self.label_files.place(x=20, y=50)
        self.label_representation.place(x=20, y=210)
        self.label_files_img.place(x=20, y=250)
        self.button1.place(x=390, rely=0.062, relheight=0.0485)
        self.button3.place(x=390, rely=0.312, relheight=0.0485)
        self.chooser.place(x=510, y=55)
        self.label_results.place(x=20, y=110)
        self.button2.place(x=730, y=50)
        self.parent = parent


    def browse_files(self):
        try:
            filename = filedialog.askopenfilename(filetypes=(('Accelerometer data files', '.txt'),))
            self.dataset = np.loadtxt(filename, delimiter=',')
            self.chooser['state'] = 'active'
            self.button2['state'] = 'normal'
            file = os.path.basename(filename)
            self.label_files.config(fg='black', text=f'Chosen file:{file}')
        except(OSError, FileNotFoundError):
            print(f'Unable to find or open file')
            self.label_files.config(fg='red', text='You need to choose file')

    def chosen_model(self):
        chosen = self.chooser.get()
        if chosen == 'First model':
            loaded_model = load_model('./model_1.h5', compile=False)
        else:
            loaded_model = load_model('./model_2.h5', compile=False)

        predict = loaded_model.predict(self.dataset)
        predict = np.argmax(predict, axis=1)
        predictions = Counter(predict)
        predictions_label = []
        sum_val = sum(predictions.values())
        for k, v in predictions.items():
            if k == 0:
                activity = 'Walking Downstairs'
            elif k == 1:
                activity = 'Jogging'
            elif k == 2:
                activity = 'Sitting'
            elif k == 3:
                activity = 'Standing'
            elif k == 4:
                activity = 'Walking Upstairs'
            else:
                activity = 'Walking'
            predictions_label.append(activity)
            percentage = (v / sum_val) * 100
            percentage = "{:.2f}".format(percentage)
            percentage = str(percentage) + '%'
            predictions_label.append(percentage)
            results = ""
            for activity, percentage in zip(predictions_label[0::2], predictions_label[1::2]):
                results = results + activity + " " + percentage + "\n" + " "
            self.label_results.config(anchor='nw', justify='left', wraplength='550',
                                      text=f'Recognized Human Activities: \n {results}')

    def browse_files_graph(self):
        try:
            self.filename = filedialog.askopenfilename(filetypes=(('Physical Activity data files', '.json .txt .npy'),))
            self.graphs_chooser['state'] = 'active'
            file = os.path.basename(self.filename)
            self.label_files_img.config(fg='black', text=f'Chosen file: {file}')
            if self.filename.endswith('.json'):
                with open(self.filename) as f:
                    data = json.load(f)
                    df_j = pd.DataFrame.from_dict(data)
                    df_j[['date', 'time']] = df_j["dateTime"].str.split(" ", 1, expand=True)

                    if 'heart' in self.filename:
                        graphs = ['BPM measured throughout the day']
                        self.graphs_chooser['values'] = graphs
                        df_j[['bpm', 'confidence']] = df_j['value'].apply(pd.Series)
                        hrdf = df_j.drop(['dateTime', 'value'], axis=1)
                        hrdf[['hours', 'minutes', 'seconds']] = hrdf["time"].str.split(":", 2, expand=True)
                        hrdf['hours'] = hrdf['hours'].astype(int)
                        hrdf['minutes'] = hrdf['minutes'].astype(int)
                        hrdf['seconds'] = hrdf['seconds'].astype(int)
                        hrdf['overall_time_mins'] = hrdf['hours'] * 60 + hrdf['minutes']
                        hrdf['overall_time_sec'] = hrdf['hours'] * 60 + hrdf['minutes'] * 60 + hrdf['seconds']

                    else:
                        graphs = ['amount of steps throughout the time period',
                                  'intensity of movement throughout the days']
                        self.graphs_chooser['values'] = graphs

                        df_j['day_number'] = df_j.groupby('date').ngroup()
                        df_j[['hours', 'minutes']] = df_j["time"].str.split(":", 1, expand=True)
                        df_j['minutes'].replace(':00', '', regex=True, inplace=True)
                        df_j['hours'] = df_j['hours'].astype(int)
                        df_j['minutes'] = df_j['minutes'].astype(int)
                        df_j['value'] = df_j['value'].astype(int)
                        df_j['oneday_time_mins'] = df_j['hours'] * 60 + df_j['minutes']
                        df_j['overall_time_mins'] = df_j['day_number'] * 24 * 60 + df_j['hours'] * 60 + df_j['minutes']
                        data2 = df_j.groupby('day_number')
                        data2 = data2['value'].sum().to_frame('total').reset_index()
                        data3 = df_j[['value', 'day_number', 'oneday_time_mins']]
                        data3 = data3.pivot(index='day_number', columns='oneday_time_mins', values='value')
                        data3[np.isnan(data3)] = 0

            elif self.filename.endswith('.npy'):
                graphs = ['intensity of activities']
                df_c = np.load(self.filename)
                self.graphs_chooser['values'] = graphs

            else:
                if 'WISDM' in self.filename:
                    graphs = ['acceleration data combined', 'acceleration data X', 'acceleration data Y',
                              'acceleration data Z',
                              'number of samples per activity']
                    self.graphs_chooser['values'] = graphs
                    columns = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
                    numb = 20
                    df = pd.read_csv(self.filename, header=None,
                                     sep=',',
                                     names=columns)
                    df['z-axis'].replace(';', '', regex=True, inplace=True)
                    df['z-axis'] = df['z-axis'].astype('float64')
                else:
                    graphs = ['acceleration data combined', 'acceleration data X', 'acceleration data Y',
                              'acceleration data Z']
                    self.graphs_chooser['values'] = graphs
                    columns = ['date and time', 'x-axis', 'y-axis', 'z-axis']
                    numb = 500
                    df = pd.read_csv(self.filename, header=None,
                                     sep=',',
                                     names=columns)

        except(OSError, FileNotFoundError):
            print(f'Unable to find or open file')
            self.label_files_img.config(fg='red', text='You need to choose file')

        def callback(event):
            if self.filename.endswith(
                    '.json') and self.graphs_chooser.get() == 'amount of steps throughout the time period':
                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(111)
                ax.bar(data2['day_number'], data2['total'])
                ax.set_ylabel('Total number of steps')
                ax.set_xlabel('Number of the day')

            elif self.filename.endswith(
                    '.json') and self.graphs_chooser.get() == 'intensity of movement throughout the days':
                fig, ax = plt.subplots(figsize=(8, 5))
                ax = sns.heatmap(data3, cbar_kws={'label': 'Intensity'}, xticklabels=False)
                ax.invert_yaxis()
                ax.set_xlabel('Minutes in one day period')
                ax.set_ylabel('Number of the day')

            elif self.filename.endswith('.json') and self.graphs_chooser.get() == 'BPM measured throughout the day':
                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(111)
                ax.set_xlabel('Time of the day [seconds]')
                ax.set_ylabel('Measured BPM')
                ax.scatter(hrdf['overall_time_sec'], hrdf['bpm'], s=2)

            elif self.filename.endswith('.npy') and self.graphs_chooser.get() == 'intensity of activities':
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.imshow(df_c)
                h = [10, 12, 16, 20]
                ax.set_yticks([5, 20, 35, 50])
                ax.set_yticklabels(h)
                ax.set_xlabel('Days')
                ax.set_ylabel('Time of day')

            else:
                fig, ax = plt.subplots(figsize=(8, 5))
                x = df['x-axis'][:numb]
                y = df['y-axis'][:numb]
                z = df['z-axis'][:numb]
                if self.graphs_chooser.get() == 'acceleration data combined':
                    ax.plot(x)
                    ax.plot(y)
                    ax.plot(z)
                elif self.graphs_chooser.get() == 'acceleration data X':
                    ax.plot(x)
                elif self.graphs_chooser.get() == 'acceleration data Y':
                    ax.plot(y)
                elif self.graphs_chooser.get() == 'number of samples per activity' and 'WISDM' in self.filename:
                    ax.bar(df.activity.unique(), df['activity'].value_counts())
                    ax.set_xlabel('Activity')
                    ax.set_ylabel('Number of Samples')
                else:
                    ax.plot(z)

            canvas = FigureCanvasTkAgg(fig, master=window)
            canvas.draw()
            canvas.get_tk_widget().place(x=20, y=300)
            self.label_files_img['text'] = self.graphs_chooser.get()

        self.graphs_chooser.bind("<<ComboboxSelected>>", callback)


if __name__ == "__main__":
    window = tk.Tk()
    window.resizable(False, False)
    window.geometry("900x800")
    window.title("Human Activity Recognition Software")
    window.config(background="lightgrey")
    MainApplication(window)
    window.mainloop()
