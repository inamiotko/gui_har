import tkinter as tk
from tkinter import filedialog
import tkinter.ttk as ttk
import pandas as pd
import os
import tensorflow
from keras.models import load_model
# from tensorflow.keras.models import load_model
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from collections import Counter

# print(tensorflow.__version__)
# print(tensorflow.keras.__version__)

# 0-Downstairs
# 1-Jogging
# 2-Sitting
# 3-Standing
# 4-Upstairs
# 5-Walking

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
window = tk.Tk()
window.geometry("900x900")
window.title("Human Activity Recognition Software")
window.config(background="lightgrey")


def browse_files():
    dataset = filedialog.askopenfilename()
    dataset = np.loadtxt(dataset, delimiter=',')

    def chosen_model():
        chosen = chooser.get()
        if chosen == 'First model':
            loaded_model = load_model('/Users/iganamiotko/Desktop/trackers/model_1.h5', compile=False)
        else:
            loaded_model = load_model('/Users/iganamiotko/Desktop/trackers/model_2.h5', compile=False)

        predict = loaded_model.predict(dataset)
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
            label_results['anchor'] = 'nw'
            label_results['justify'] = 'left'
            label_results['wraplength'] = 500
            label_results["text"] = 'Recognised Human Activities with probability rates: \n' \
                                    f'{predictions_label}'

    button2 = tk.Button(text="Execute", width=15, height=3, bg="grey", fg="black", command=chosen_model)
    button2.place(x=730, y=50)


def browse_files_graph():
    filename = filedialog.askopenfilename()
    graphs_chooser = ttk.Combobox(window)
    sns.set()
    graphs_chooser.place(x=520, y=255)

    if filename.endswith('.json'):
        with open(filename) as f:
            data = json.load(f)
            df_j = pd.DataFrame.from_dict(data)
            df_j[['date', 'time']] = df_j["dateTime"].str.split(" ", 1, expand=True)

            if 'heart' in filename:
                graphs = ['BPM measured throughout the day']
                graphs_chooser['values'] = graphs
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
                graphs_chooser['values'] = graphs

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

    elif filename.endswith('.txt'):
        graphs = ['acceleration data combined', 'acceleration data X', 'acceleration data Y', 'acceleration data Z',
                  'number of samples per activity']
        graphs_chooser['values'] = graphs
        columns = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
        numb = 20
        df = pd.read_csv(filename, header=None,
                         sep=',',
                         names=columns)
        df['z-axis'].replace(';', '', regex=True, inplace=True)
        df['z-axis'] = df['z-axis'].astype('float64')
    elif filename.endswith('.npy'):
        graphs = ['intensity of activities']
        df_c = np.load(filename)
        graphs_chooser['values'] = graphs
    else:
        graphs = ['acceleration data combined', 'acceleration data X', 'acceleration data Y', 'acceleration data Z']
        graphs_chooser['values'] = graphs
        columns = ['date and time', 'x-axis', 'y-axis', 'z-axis']
        numb = 500
        df = pd.read_csv(filename, header=None,
                         sep=',',
                         names=columns)

    def callback(event):
        if filename.endswith('.json') and graphs_chooser.get() == 'amount of steps throughout the time period':
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            ax.bar(data2['day_number'], data2['total'])
            ax.set_ylabel('Total number of steps')
            ax.set_xlabel('Number of the day')

        elif filename.endswith('.json') and graphs_chooser.get() == 'intensity of movement throughout the days':
            fig, ax = plt.subplots(figsize=(8, 5))
            ax = sns.heatmap(data3, cbar_kws={'label': 'Intensity'}, xticklabels=False)
            ax.invert_yaxis()
            ax.set_xlabel('Minutes in one day period')
            ax.set_ylabel('Number of the day')

        elif filename.endswith('.json') and graphs_chooser.get() == 'BPM measured throughout the day':
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            ax.set_xlabel('Time of the day [seconds]')
            ax.set_ylabel('Measured BPM')
            ax.scatter(hrdf['overall_time_sec'], hrdf['bpm'], s=2)

        elif filename.endswith('.npy') and graphs_chooser.get() == 'intensity of activities':
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
            if graphs_chooser.get() == 'acceleration data combined':
                ax.plot(x)
                ax.plot(y)
                ax.plot(z)
            elif graphs_chooser.get() == 'acceleration data X':
                ax.plot(x)
            elif graphs_chooser.get() == 'acceleration data Y':
                ax.plot(y)
            elif (graphs_chooser.get() == 'number of samples per activity' and
                  filename == "/Users/iganamiotko/dataset/activityprediction/WISDM_ar_v1.1_raw.txt"):
                ax.bar(df.activity.unique(), df['activity'].value_counts())
                ax.set_xlabel('Activity')
                ax.set_ylabel('Number of Samples')
            else:
                ax.plot(z)

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().place(x=20, y=300)
        label_files_img['text'] = graphs_chooser.get()

    graphs_chooser.bind("<<ComboboxSelected>>", callback)


button1 = tk.Button(text="Browse", width=10, height=2, bg="grey", fg="black", command=browse_files)
label_files = tk.Label(window, text="Choose desired signal", width=40, height=2, fg="blue")
label_files_img = tk.Label(window, text="Choose desired signal", width=40, height=2, fg="blue")
button3 = tk.Button(text="Browse", width=10, height=2, bg="grey", fg="black", command=browse_files_graph)
choices = ['First model', 'Second model']
chooser = ttk.Combobox(window)
chooser['values'] = choices
chooser.current(0)

label_results = tk.Label(window, text="Recognised human activity:  ", width=50, height=7, fg="black")

label_files.place(x=20, y=50)
label_files_img.place(x=20, y=250)
button1.place(x=390, y=50)
button3.place(x=390, y=250)
chooser.place(x=510, y=55)
label_results.place(x=20, y=100)

window.mainloop()
