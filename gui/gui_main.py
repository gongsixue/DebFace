from tkinter import *
import tkinter as tk

from MyLabel import myLabel
from Dataset_menu import Dataset_menu
from Test_acc_gender import Test_acc_gender
from Instructions import Instructions


if __name__ == "__main__":
     # Create and configure window:
     window = tk.Tk()
     window.configure(background="#ffffff")  # background
     window.geometry("800x600")  # size
     window.title("Gender bias in Face Recognition GUI")  # title

     # create dropdown menu for datasets
     dataset_menu = Dataset_menu(window)
     dataset_menu.create(x=190, y=80)

     # create accuracy and gender testing for (Current, previous, comparison)
     test_model = Test_acc_gender(window, dataset_menu)
     test_model.main()

     # create instructions for buttons
     instructions = Instructions(window)
     instructions.main()

     # Start the window, Create text entry box for : status field
     statusField = tk.Entry(window)
     window.mainloop()