import tkinter as tk
import random as rd

class Buttons:
    def __init__(self, window, value=None) -> None:
        self.window = window
        self.value = value

    def test_btn(self, accuracy, updated_accuracy, gender, gender_value):
            updated_accuracy = round(rd.uniform(0, 100), 2)
            accuracy["text"] = str(updated_accuracy) + "%"

            GENDER_VAL = ["MALE", "FEMALE"]
            gender = rd.choices(GENDER_VAL)
            gender_value["text"] = str(gender.pop())

    def save_btn(self, accuracy_val, dataset, accuracy_current, dataset_current):
        accuracy_val["text"] = accuracy_current["text"]
        dataset["text"] = dataset_current

    def compare_btn(self, accuracy_current, accuracy_prev, accuracy_diff):
        def p2f(x):
            return float(x.strip('%'))

        accuracy_diff["text"] = (
            str(round(abs(p2f(accuracy_current["text"]) - p2f(accuracy_prev["text"])), 2)) + "%"
        )

