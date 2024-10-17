import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and prepare the data
df = pd.read_csv("C:/Users/ajeet/Downloads/placement.csv")
x = df.iloc[:, 0:1]  # CGPA column
y = df.iloc[:, -1]   # Package column

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Create the prediction function
def predict_package(cgpa):
  predicted_package = lr.predict([[cgpa]])[0]
  return f"Predicted Package: {predicted_package:.2f}"

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_package,
    inputs=gr.inputs.Number(label="Enter CGPA"),
    outputs="text",
    title="Package Predictor",
    description="Enter your CGPA to predict your package."
)

# Launch the interface
iface.launch()