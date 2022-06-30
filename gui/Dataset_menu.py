from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
from MyLabel import myLabel

class Dataset_menu:
    # DATASETS
    DATASETS_OPTIONS = [
        "AFAD",
        "UTKFace",
        "AFAD (Adversarial-attack)",
        "UTKFace (Adversarial-attack)",
    ]
    
    CURRENT_DATASET = None

    def __init__(self, window):
        self.datasetValue = StringVar()  # datatype of menu text
        self.datasetValue.set(Dataset_menu.DATASETS_OPTIONS[0])  # initial menu text
        self.window = window

    def create(self, x, y):
        drop = OptionMenu(
            self.window, self.datasetValue, *(Dataset_menu.DATASETS_OPTIONS),
            command= lambda _: self.menu_check_btn(self.datasetValue)
        )
        drop.pack()
        drop.place(x=x, y=y)

        # Label for datasetLabel:
        datasetLabel = myLabel(self.window, "Dataset", "Helvetica", "#ffffff", "#000000")
        datasetLabel.create(x=65, y=80)

        # Add image:
        frame = tk.Frame(self.window)
        frame.place(x=50, y=120)
        img = ImageTk.PhotoImage(Image.open("image-dataset/53537-0.jpg").resize((100, 100)))
        label = tk.Label(frame, image=img)
        label.pack()

    # Dropdown function:
    def menu_check_btn(self, datasetValue):
        # Add image:
        frame = tk.Frame(self.window)
        frame.place(x=50, y=120)
        img = ImageTk.PhotoImage(Image.open("image-dataset/53537-0.jpg").resize((100, 100)))
        label = tk.Label(frame, image=img)
        label.pack()            

        Dataset_menu.CURRENT_DATASET = datasetValue.get()
        if Dataset_menu.CURRENT_DATASET == "AFAD":
            img = ImageTk.PhotoImage(
                Image.open("image-dataset/53537-0.jpg").resize((100, 100))
            )
            label.configure(image=img)
            label.img = img

        elif Dataset_menu.CURRENT_DATASET == "UTKFace":
            img = ImageTk.PhotoImage(
                Image.open("image-dataset/56951-0.jpg").resize((100, 100))
            )
            label.configure(image=img)
            label.img = img
        pass