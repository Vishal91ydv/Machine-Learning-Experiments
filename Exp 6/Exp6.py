import tkinter as tk 
from tkinter import messagebox 
import pandas as pd 
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
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

        # Split the dataset
        X = df.drop('target', axis='columns') 
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=int(estimator_entry.get())) 
        model.fit(X_train, y_train)

        # Test accuracy 
        accuracy = model.score(X_test, y_test) 
        accuracy_label.config(text=f"Model Accuracy: {accuracy:.2f}")

        # Prediction and Confusion Matrix 
        y_pred = model.predict(X_test) 
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix 
        fig = plt.figure(figsize=(6, 4)) 
        sn.heatmap(cm, annot=True, fmt="d", cmap="Blues") 
        plt.xlabel("Predicted") 
        plt.ylabel("Actual")

        # Clear previous canvas and display the new one 
        for widget in canvas_frame.winfo_children(): 
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame) 
        canvas.draw() 
        canvas.get_tk_widget().pack()

    except Exception as e:
        messagebox.showerror("Error", str(e))
# Set up the GUI window 
window = tk.Tk() 
window.title("Random Forest Classifier GUI by Vishal Boss 211P042")
# Label for number of estimators
estimator_label = tk.Label(window, text="Number of Estimators:") 
estimator_label.pack(pady=10)
# Entry for number of estimators 
estimator_entry = tk.Entry(window) 
estimator_entry.insert(0, "20") # Default value 
estimator_entry.pack(pady=10)

# Train button 
train_button = tk.Button(window, text="Train Model", command=train_model) 
train_button.pack(pady=20)

# Accuracy label 
accuracy_label = tk.Label(window, text="Model Accuracy: N/A") 
accuracy_label.pack(pady=10)

# Frame to hold the confusion matrix 
canvas_frame = tk.Frame(window) 
canvas_frame.pack(pady=20)

# Run the GUI event loop 
window.mainloop()
