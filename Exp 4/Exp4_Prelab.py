import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Function to predict digit based on drawn image
def predict_digit(img):
    # Resize image to 8x8 pixels and convert to grayscale
    img = img.resize((8, 8), Image.Resampling.LANCZOS).convert('L')  # Use LANCZOS instead of ANTIALIAS
    img = np.array(img)

    # Invert the image color (as the model was trained on white digits on a black background)
    img = 16 - (img / 16)
    img = img.reshape(1, -1)

    # Predict the digit
    res = model.predict(img)
    return res[0]

# GUI to draw and predict digit
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        self.root.geometry("400x400")

        self.canvas = Canvas(self.root, width=200, height=200, bg="white")
        self.canvas.pack(pady=20)
        
        self.clear_btn = Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side=LEFT, padx=20)

        self.predict_btn = Button(self.root, text="Predict", command=self.classify_handwritten_digit)
        self.predict_btn.pack(side=RIGHT, padx=20)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.image = Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw_img = ImageDraw.Draw(self.image)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw_img = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw_img.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def classify_handwritten_digit(self):
        digit = predict_digit(self.image)
        messagebox.showinfo("Prediction", f"The digit is: {digit}")

# Create GUI window
root = Tk()
app = DigitApp(root)
root.mainloop()
