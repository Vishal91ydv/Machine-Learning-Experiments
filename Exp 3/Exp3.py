import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("C:/Users/ajeet/Downloads/add.csv")

x=df.iloc[:,0:2]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


import tkinter as tk
import tkinter.messagebox as messagebox

def predict_sum():
    try:
        x1 = float(entry_x1.get())
        x2 = float(entry_x2.get())
        y_pred = lr.predict(pd.DataFrame({"x": [x1], "y": [x2]}))
        result_label.config(text=f"Predicted Sum: {y_pred[0]}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers for x1 and x2.")

# Create main window
window = tk.Tk()
window.title("Sum Prediction")

# Add padding to elements
padx = 10
pady = 5

# Input fields
label_x1 = tk.Label(window, text="x1:")
label_x1.grid(row=0, column=0,padx=padx, pady=pady)
entry_x1 = tk.Entry(window)
entry_x1.grid(row=0, column=1,padx=padx, pady=pady)

label_x2 = tk.Label(window, text="x2:")
label_x2.grid(row=1, column=0,padx=padx, pady=pady)
entry_x2 = tk.Entry(window)
entry_x2.grid(row=1, column=1,padx=padx, pady=pady)

# Predict button
predict_button = tk.Button(window, text="Vishal's Predict", command=predict_sum)
predict_button.grid(row=2, column=0, columnspan=2)

# Result label
result_label = tk.Label(window, text="")
result_label.grid(row=3, column=0, columnspan=2,padx=padx, pady=pady)

# Start the GUI
window.mainloop()