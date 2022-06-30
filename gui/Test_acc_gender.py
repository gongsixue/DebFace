import tkinter as tk

from MyLabel import myLabel
from Buttons import Buttons

class Test_acc_gender:

    # DATASET_VAL = None

    def __init__(self, window, dataset_val) -> None:
        self.window = window
        self.dataset_val = dataset_val

    def main(self):

        TEST_VALUE = "0.0%"
        ###### Current Accuracy ######
        accuracy_label = myLabel(self.window, "Accuracy: ", "Helvetica", "#ffffff", "#000000")
        accuracy_value = myLabel(self.window, TEST_VALUE, "Helvetica", "#12ed96", "#000000")
        accuracy_label.create(x=430, y=100)
        accuracy_value = accuracy_value.create(x=525, y=100)

        GENDER = None
        ###### Current Gender ######
        genderLabel = myLabel(self.window, "Gender: ", "Helvetica", "#ffffff", "#000000")
        gender_value = myLabel(self.window, GENDER, "Helvetica", "#ffffff", "#000000")
        genderLabel.create(x=610, y=100)
        gender_value = gender_value.create(x=690, y=100)

        # Label for test result
        test_result = myLabel(self.window, "Test Result", "Helvetica", "#ffffff", "#000000")
        test_result.create(x=500, y=50)

        ###### Test Sample button ######
        btn_sample = Buttons(self.window, value=TEST_VALUE)
        sampleButton = tk.Button(
            btn_sample.window,
            text="Test (SAMPLE)",
            bg="yellow",
            fg="black",
            command=lambda: btn_sample.test_btn(accuracy_value, TEST_VALUE, genderLabel, gender_value),
        )
        sampleButton.place(x=55, y=235)

        ###### Test FULL button ######
        btn_test = Buttons(self.window, value=TEST_VALUE)
        testButton = tk.Button(
            btn_test.window,
            text="Test (FULL)",
            bg="red",
            fg="black",
            command=lambda: btn_test.test_btn(accuracy_value, TEST_VALUE, genderLabel, gender_value),
        )
        testButton.place(x=55, y=265)

        # COMPARISON
        ###### Previous Dataset #######
        prev_datasetLabel = myLabel(self.window, "Prev-Dataset: ", "Helvetica", "#ffffff", "#000000")
        prev_datasetLabel.create(x=430, y=380)
        PREV_DS = ""
        prev_dataset = myLabel(self.window, PREV_DS, "Helvetica", "#ffffff", "#000000")
        prev_dataset = prev_dataset.create(x=560, y=380)

        PREV_ACCURACY_VAL = ""
        ####### Previous Accuracy #######
        prev_accuracy_label = myLabel(self.window, "Prev-Accuracy: ", "Helvetica", "#ffffff", "#000000")
        prev_accuracy_value = myLabel(self.window, PREV_ACCURACY_VAL, "Helvetica", "#ffffff", "#000000")
        prev_accuracy_label.create(x=430, y=420)
        prev_accuracy_value = prev_accuracy_value.create(x=570, y=420)

        ACCURACY_DIFF = "0.0%"
        ####### Accuracy Diff #######
        accuracy_diff_label = myLabel(self.window, "Accuracy Difference: ", "Helvetica", "#ffffff", "#000000")
        accuracy_diff_value = myLabel(self.window, ACCURACY_DIFF, "Helvetica", "#12ed96", "#000000")
        accuracy_diff_label.create(x=430, y=460)
        accuracy_diff_value = accuracy_diff_value.create(x=620, y=460)

        
        ###### Save button ######
        btn_save = Buttons(self.window, value=TEST_VALUE)
        saveButton = tk.Button(
            btn_save.window,
            text="Save",
            bg="green",
            fg="black",
            command=lambda: btn_save.save_btn(
                prev_accuracy_value, prev_dataset, accuracy_value, self.dataset_val.CURRENT_DATASET
            ),
        )
        saveButton.place(x=630, y=53)

        ###### Compare button ######
        btn_compare = Buttons(self.window, value=TEST_VALUE)
        compareButton = tk.Button(
            btn_compare.window,
            text="Compare Previous",
            bg="orange",
            fg="black",
            command=lambda: btn_compare.compare_btn(accuracy_value, prev_accuracy_value, accuracy_diff_value),
        )
        compareButton.place(x=680, y=53)

        ####### Button config #######
        sampleButton.config(height=1, width=12)
        testButton.config(height=1, width=12)
