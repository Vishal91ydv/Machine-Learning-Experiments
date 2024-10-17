import tkinter as tk 
from tkinter import messagebox, ttk 
import pandas as pd 
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix 
import seaborn as sn 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to train the model 
def train_model(): 
    try:
        # Load the digits dataset 
        digits = load_digits() 
        df = pd.DataFrame(digits.data) 
        df['target'] = digits.target
        # Split the dataset into training and test sets 
        X = df.drop('target', axis='columns') 
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Select the base estimator based on user selection 
        base_estimator_name = estimator_combo.get() 
        if base_estimator_name == "Decision Tree":
            estimator = DecisionTreeClassifier()
        elif base_estimator_name == "Logistic Regression":
            estimator = LogisticRegression(max_iter=1000)
        # Get the number of estimators from the input 
        n_estimators = int(estimator_entry.get())
        # Train Bagging model (use 'estimator' instead of 'base_estimator')
        model = BaggingClassifier(estimator=estimator, n_estimators=n_estimators) 
        model.fit(X_train, y_train)
        # Evaluate the model 
        accuracy = model.score(X_test, y_test) 
        accuracy_label.config(text=f"Model Accuracy: {accuracy:.2f}")
        # Make predictions and create a confusion matrix 
        y_pred = model.predict(X_test) 
        cm = confusion_matrix(y_test, y_pred)
        # Plot the confusion matrix using seaborn 
        fig, ax = plt.subplots(figsize=(6, 4)) 
        sn.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax) 
        ax.set_xlabel("Predicted") 
        ax.set_ylabel("Actual")
        # Clear any previous plots and display the new plot in the canvas 
        for widget in canvas_frame.winfo_children(): 
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame) 
        canvas.draw() 
        canvas.get_tk_widget().pack()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Set up the GUI window 
window = tk.Tk() 
window.title("Bagging Classifier GUI by Vishal Boss 211P042")

# Label for selecting the base estimator 
estimator_label = tk.Label(window, text="Select Base Estimator:") 
estimator_label.pack(pady=10)

# Combobox for base estimator selection 
estimator_combo = ttk.Combobox(window, values=["Decision Tree", "Logistic Regression"]) 
estimator_combo.current(0) # Default to Decision Tree 
estimator_combo.pack(pady=10)

# Label for the number of estimators 
estimator_entry_label = tk.Label(window, text="Number of Estimators:") 
estimator_entry_label.pack(pady=10)

# Entry box for number of estimators input 
estimator_entry = tk.Entry(window) 
estimator_entry.insert(0, "10") # Default value is 10 
estimator_entry.pack(pady=10)

# Button to start training the model 
train_button = tk.Button(window, text="Train Model", command=train_model) 
train_button.pack(pady=20)

# Label to display accuracy 
accuracy_label = tk.Label(window, text="Model Accuracy: N/A") 
accuracy_label.pack(pady=10)

# Frame to hold the confusion matrix 
canvas_frame = tk.Frame(window) 
canvas_frame.pack(pady=20)

# Run the Tkinter event loop 
window.mainloop()
