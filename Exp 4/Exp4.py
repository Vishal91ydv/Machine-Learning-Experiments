import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Step 1: Load and preprocess the Titanic dataset
df = pd.read_csv("C:/Users/ajeet/Downloads/titanic.csv")

# Fill missing values
df['age'].fillna(value=df['age'].mean(), inplace=True) 
df['fare'].fillna(value=df['fare'].mean(), inplace=True)
df['embarked'].fillna(value=df['embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
df.drop(labels=['cabin', 'name', 'ticket'], axis=1, inplace=True)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# Step 2: Split the data into features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 4: Create the Tkinter GUI
root = tk.Tk()
root.title("Logistic Regression Results By Vishal Boss 211P042")

# Display the classification report in the GUI
report = classification_report(y_test, model.predict(X_test), output_dict=True)
report_text = pd.DataFrame(report).transpose().round(2).to_string()

# Text widget to display classification report
text = tk.Text(root, height=10, width=60)
text.insert(tk.END, report_text)
text.pack()

# Dropdown for selecting gender
gender_label = ttk.Label(root, text="Select Gender: ")
gender_label.pack(pady=5)
gender = ttk.Combobox(root, values=["male", "female"])
gender.pack(pady=5)

# Dropdown for selecting pclass
pclass_label = ttk.Label(root, text="Select Pclass:")
pclass_label.pack(pady=5)
pclass = ttk.Combobox(root, values=[1, 2, 3])
pclass.pack(pady=5)

# Function to filter survivors based on gender and pclass
def show_survivors():
    # Fix: Use the correct dataframe 'df'
    filtered = df[(df['sex_male'] == (1 if gender.get() == "male" else 0)) &
                  (df['pclass'] == int(pclass.get())) & 
                  (df['survived'] == 1)]
    result_text = f"Survivors: {len(filtered)}"
    result_label.config(text=result_text)

# Button to show survivors
button = ttk.Button(root, text="Show Survivors", command=show_survivors)
button.pack(pady=10)

# Label to display the result
result_label = ttk.Label(root, text="")
result_label.pack(pady=5)

# Start the Tkinter event loop
root.mainloop()
