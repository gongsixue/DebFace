import tkinter as tk

from MyLabel import myLabel
from Buttons import Buttons

class Instructions:

    def __init__(self, window) -> None:
        self.window = window

    def main(self):
        ############ Label for instructions ############
        legend_labels = myLabel(
            self.window, "Buttons Legend", ("Helvetica", 12), "#ffffff", "#000000"
        )
        legend_labels.create(x=60, y=400)

        # Test (sample)
        btn_sample = myLabel(
            self.window, "Test (Sample)", ("Helvetica", 8), "#ffffff", "#000000"
        )
        btn_sample.create(x=60, y=425)
        ins_sample = myLabel(
            self.window,
            ": Sample one image from the dataset",
            ("Helvetica", 8),
            "#ffffff",
            "#000000",
        )
        ins_sample.create(x=140, y=425)

        # Test (full)
        btn_full = myLabel(self.window, "Test (FULL)", ("Helvetica", 8), "#ffffff", "#000000")
        btn_full.create(x=60, y=450)
        ins_full = myLabel(
            self.window, ": Test the entire dataset", ("Helvetica", 8), "#ffffff", "#000000"
        )
        ins_full.create(x=140, y=450)