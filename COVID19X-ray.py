import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
import numpy as np


# Load the model from the local directory
model = load_model('model.h5')


# Create a TkinterDnD window
window = TkinterDnD.Tk()
window.title("COVID-19 X-ray Classifier")


# Set the window icon

icon_image = Image.open('icon.jpg')
icon_photo = ImageTk.PhotoImage(icon_image)
window.iconphoto(True, icon_photo)

# Set the window size
window.geometry("800x600")

# Set the background color to a darker shade
window.configure(bg="#555555")

# Create a label for the text "Good Morning"
text_label_1 = tk.Label(window, text="COVID-19 X-ray Classifier: Assessing Disease\nStage with a Simple Drag and Drop", bg="#222222", fg="white", font=("Arial", 18))
text_label_1.pack()

# Function to handle dropping files
def drop_files(event):
    image_path = event.data
    image_path = image_path.strip('{}')
    load_image(image_path)

# Function to open a file dialog and select an image file
def explore_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    load_image(image_path)

# Function to load and display the image
def load_image(image_path):
    # Load and resize the image
    image = Image.open(image_path)

    classname = ['Early', 'Normal', 'Severe']
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    predicted_label = classname[predicted_class]
    confidence = prediction[0][predicted_class] * 100

    image = image.resize((300, 300))  # Adjust the size as needed

    # Create a Tkinter-compatible photo image
    photo = ImageTk.PhotoImage(image)

    # Update the image label with the selected image
    image_label.config(image=photo)
    image_label.image = photo

    # Show the "Hello" text
    text_label_2.config(text=f"Predicted class: {predicted_label}\nConfidence: {confidence:.2f}%")

if __name__ == "__main__":
    
    # Enable drop events on the window
    window.drop_target_register(DND_FILES)
    window.dnd_bind('<<Drop>>', drop_files)

    # Create a button to explore and select an image
    explore_button = tk.Button(window, text="Explore Image", command=explore_image)
    explore_button.pack(pady=20)

    # Create a frame for the image
    frame = tk.Frame(window, bg="#333333")
    frame.pack(pady=50)

    # Create a label to display the image within the frame
    image_label = tk.Label(frame, bg="#333333")
    image_label.pack()

    # Create a label for the text "Hello"
    text_label_2 = tk.Label(window, text="", bg="#333333", fg="white")
    text_label_2.pack()

    window.mainloop()
