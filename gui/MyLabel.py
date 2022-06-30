import tkinter as tk

class myLabel:
    def __init__(self, window, text, font, bg, fg):
        self.window = window
        self.text = text
        self.font = font        # ("Helvetica", 12, "bold")
        self.bg = bg
        self.fg = fg


    def create(self, x, y):
        self.mylabel =  tk.Label(self.window, text = self.text, font=self.font, bg=self.bg, fg=self.fg)
        self.mylabel.place(x=x, y=y)
        return self.mylabel
