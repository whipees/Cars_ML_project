"""
This is the main Graphical User Interface (GUI) application.
It integrates both the computer vision model (MobileNetV2) for brand recognition
and the Random Forest regression model for price prediction into a single executable app.
"""

import tkinter as tk
from tkinter import filedialog, ttk
import joblib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageTk


class SafeDense(tf.keras.layers.Dense):
    """
    A custom wrapper for the Keras Dense layer.
    This safely handles backward compatibility issues between different TensorFlow versions.
    It pops the 'quantization_config' argument which causes errors in older TF versions
    when loading models trained in Google Colab.
    """
    def __init__(self, *args, **kwargs):
        try:
            kwargs.pop('quantization_config', None)
        except Exception as e:
            pass
        super().__init__(*args, **kwargs)


class App:
    """
    The main application class that handles the window lifecycle, UI layout,
    model loading, and user interactions.
    """
    def __init__(self, master):
        """
        Initializes the application window, sets the window title, default size,
        background color, and applies a modern UI theme if available.
        Then it triggers the setup methods.
        """
        try:
            self.master = master
            self.master.title("Car ML Predictor")
            self.master.geometry("650x850")
            self.master.configure(bg="#F0F2F5")

            try:
                self.style = ttk.Style()
                self.style.theme_use('clam')
            except Exception as e:
                pass

            self.setup_scroll()
            self.load_models()
            self.create_widgets()
        except Exception as e:
            print(str(e))

    def setup_scroll(self):
        """
        Configures a scrollable window. Since Tkinter does not have a native ScrollFrame,
        this method creates a Canvas, attaches a vertical Scrollbar to it, and places
        a standard Frame inside the Canvas to hold all the UI widgets.
        It also binds mouse wheel events to enable smooth scrolling.
        """
        try:
            self.main_container = tk.Frame(self.master, bg="#F0F2F5")
            self.main_container.pack(fill="both", expand=True)

            self.canvas = tk.Canvas(self.main_container, bg="#F0F2F5", highlightthickness=0)
            self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
            self.scrollable_frame = tk.Frame(self.canvas, bg="#F0F2F5")

            try:
                """ Updates the scroll region dynamically when the inner frame changes size """
                self.scrollable_frame.bind(
                    "<Configure>",
                    lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                )
            except Exception as e:
                pass

            self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

            try:
                """ Ensures the inner frame stretches to fill the canvas width """
                self.canvas.bind(
                    "<Configure>",
                    lambda e: self.canvas.itemconfig(self.canvas_frame, width=e.width)
                )
            except Exception as e:
                pass

            self.canvas.configure(yscrollcommand=self.scrollbar.set)
            self.canvas.pack(side="left", fill="both", expand=True)
            self.scrollbar.pack(side="right", fill="y")

            try:
                """ Binds the mouse wheel scrolling across the entire application window """
                self.master.bind_all("<MouseWheel>",
                                     lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
            except Exception as e:
                pass

        except Exception as e:
            print(str(e))

    def load_models(self):
        """
        Loads all required machine learning assets from the disk:
        1. The trained Random Forest regressor (price_model.pkl).
        2. The label encoders for categorical data (encoders.pkl).
        3. The custom MobileNetV2 image classifier (brand_classifier.h5).
        4. The JSON list of brand names recognized by the vision model.
        5. The cleaned dataset used for cascading dynamic dropdown menus.
        """
        try:
            self.price_model = joblib.load("../../models/price_model.pkl")
            self.encoders = joblib.load("../../models/encoders.pkl")

            """ Instantiates the vision model with the custom SafeDense layer to avoid crashes """
            self.vision_model = tf.keras.models.load_model(
                "../../models/brand_classifier.h5",
                custom_objects={'Dense': SafeDense},
                compile=False
            )

            try:
                with open("../../models/class_names.json", "r") as f:
                    self.class_names = json.load(f)
            except Exception as e:
                print(str(e))

            try:
                """ Loads the CSV specifically to populate the dynamic dependent dropdowns """
                self.df = pd.read_csv("../../data/processed/cleaned_cars_data.csv")
            except Exception as e:
                self.df = None

        except Exception as e:
            print(str(e))

    def create_widgets(self):
        """
        Constructs the entire graphical user interface, utilizing 'Cards' (Frames)
        to separate the Computer Vision section from the Price Prediction section.
        Configures colors, fonts, buttons, and input fields dynamically.
        """
        try:
            try:
                title_lbl = tk.Label(self.scrollable_frame, text="AI Car Assistant", font=("Helvetica", 20, "bold"),
                                     bg="#F0F2F5", fg="#1C1E21")
                title_lbl.pack(pady=(20, 10))
            except Exception as e:
                pass

            try:
                """ Builds the top UI card dedicated to image recognition """
                card1 = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=0, highlightbackground="#E4E6E9",
                                 highlightcolor="#E4E6E9", highlightthickness=1)
                card1.pack(fill="x", padx=30, pady=10, ipadx=10, ipady=15)

                tk.Label(card1, text="1. Visual Brand Recognition", font=("Helvetica", 14, "bold"), bg="#FFFFFF",
                         fg="#1C1E21").pack(pady=(0, 10))

                btn1 = tk.Button(card1, text="Upload Photo", font=("Helvetica", 11, "bold"), bg="#0064E0", fg="#FFFFFF",
                                 activebackground="#0056C2", activeforeground="#FFFFFF", relief="flat", cursor="hand2",
                                 command=self.process_image)
                btn1.pack(pady=5, ipadx=10, ipady=3)

                self.img_lbl = tk.Label(card1, bg="#FFFFFF")
                self.img_lbl.pack(pady=5)

                self.vision_res = tk.Label(card1, text="No image selected", font=("Helvetica", 12), bg="#FFFFFF",
                                           fg="#65676B")
                self.vision_res.pack()
            except Exception as e:
                print(str(e))

            try:
                """ Builds the bottom UI card dedicated to manual inputs and price prediction """
                card2 = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=0, highlightbackground="#E4E6E9",
                                 highlightcolor="#E4E6E9", highlightthickness=1)
                card2.pack(fill="both", expand=True, padx=30, pady=10, ipadx=10, ipady=15)

                tk.Label(card2, text="2. Smart Price Predictor", font=("Helvetica", 14, "bold"), bg="#FFFFFF",
                         fg="#1C1E21").pack(pady=(0, 15))

                self.inputs = {}
                fields = ["brand", "model", "fuel", "engine_type", "power_kw", "drivetrain", "transmission", "year",
                          "mileage_km"]
                self.cat_fields = ["brand", "model", "fuel", "engine_type", "drivetrain", "transmission"]

                """ Iterates through all required fields and dynamically generates Entry or Combobox widgets """
                for f in fields:
                    try:
                        frame = tk.Frame(card2, bg="#FFFFFF")
                        frame.pack(fill="x", padx=40, pady=4)

                        lbl_text = str(f).replace("_", " ").title()
                        tk.Label(frame, text=lbl_text, width=15, anchor="w", font=("Helvetica", 11), bg="#FFFFFF",
                                 fg="#333333").pack(side="left")

                        try:
                            if f in self.cat_fields:
                                ent = ttk.Combobox(frame, font=("Helvetica", 11), state="normal")
                                """ Binds events to trigger the cascading dependent dropdown logic """
                                ent.bind("<<ComboboxSelected>>", self.update_dropdowns)
                                ent.bind("<FocusOut>", self.update_dropdowns)
                            else:
                                ent = tk.Entry(frame, font=("Helvetica", 11), relief="solid", bd=1)
                        except Exception as e:
                            ent = tk.Entry(frame)

                        ent.pack(side="right", fill="x", expand=True)
                        self.inputs[f] = ent
                    except Exception as e:
                        print(str(e))

                try:
                    """ Populates the primary 'Brand' dropdown with unique values from the dataset """
                    if self.df is not None:
                        self.inputs["brand"]["values"] = sorted(list(self.df["brand"].dropna().astype(str).unique()))
                    else:
                        self.inputs["brand"]["values"] = sorted(list(self.encoders["brand"].classes_))
                except Exception as e:
                    pass

                btn2 = tk.Button(card2, text="Calculate Market Value", font=("Helvetica", 12, "bold"), bg="#31A24C",
                                 fg="#FFFFFF", activebackground="#2B8C42", activeforeground="#FFFFFF", relief="flat",
                                 cursor="hand2", command=self.process_price)
                btn2.pack(pady=(20, 10), ipadx=15, ipady=5)

                self.price_res = tk.Label(card2, text="— CZK", font=("Helvetica", 18, "bold"), bg="#FFFFFF",
                                          fg="#1C1E21")
                self.price_res.pack()
            except Exception as e:
                print(str(e))

        except Exception as e:
            print(str(e))

    def update_dropdowns(self, event=None):
        """
        Handles cascading dropdown logic. When the user selects a Brand,
        it filters the dataset and updates the Model dropdown to only show
        models belonging to that brand. Selecting a Model further filters
        available engines, fuels, and transmissions.
        """
        try:
            if self.df is None:
                return

            brand = self.inputs["brand"].get().strip().lower()
            model = self.inputs["model"].get().strip().lower()

            temp_df = self.df.copy()

            try:
                """ Clears child dropdowns if the parent 'Brand' is changed """
                if event and event.widget == self.inputs["brand"]:
                    self.inputs["model"].set("")
                    self.inputs["fuel"].set("")
                    self.inputs["engine_type"].set("")
                    self.inputs["drivetrain"].set("")
                    self.inputs["transmission"].set("")
            except Exception as e:
                pass

            try:
                """ Filters valid models based on the selected brand """
                if brand:
                    temp_df = temp_df[temp_df["brand"].astype(str).str.lower() == brand]
                    self.inputs["model"]["values"] = sorted(list(temp_df["model"].dropna().astype(str).unique()))
            except Exception as e:
                pass

            try:
                """ Clears grandchild dropdowns if the parent 'Model' is changed """
                if event and event.widget == self.inputs["model"]:
                    self.inputs["fuel"].set("")
                    self.inputs["engine_type"].set("")
                    self.inputs["drivetrain"].set("")
                    self.inputs["transmission"].set("")
            except Exception as e:
                pass

            try:
                """ Filters engine variants, fuels, and transmissions based on the chosen model """
                if brand and model:
                    temp_df = temp_df[temp_df["model"].astype(str).str.lower() == model]
                    self.inputs["fuel"]["values"] = sorted(list(temp_df["fuel"].dropna().astype(str).unique()))
                    self.inputs["engine_type"]["values"] = sorted(
                        list(temp_df["engine_type"].dropna().astype(str).unique()))
                    self.inputs["drivetrain"]["values"] = sorted(
                        list(temp_df["drivetrain"].dropna().astype(str).unique()))
                    self.inputs["transmission"]["values"] = sorted(
                        list(temp_df["transmission"].dropna().astype(str).unique()))
            except Exception as e:
                pass

        except Exception as e:
            print(str(e))

    def process_image(self):
        """
        Triggers a file dialog to open an image. The image is resized to 224x224,
        converted to a numpy array, preprocessed using MobileNetV2 standards,
        and fed into the neural network.
        The prediction result is displayed, and the detected brand is automatically
        pushed to the price prediction form.
        """
        try:
            path = filedialog.askopenfilename()
            try:
                if path:
                    """ Displays the selected image on the GUI """
                    img = Image.open(path).resize((224, 224))
                    img_tk = ImageTk.PhotoImage(img)
                    self.img_lbl.config(image=img_tk)
                    self.img_lbl.image = img_tk

                    """ Prepares the image array for TensorFlow prediction """
                    arr = tf.keras.preprocessing.image.img_to_array(img)
                    arr = np.expand_dims(arr, axis=0)
                    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

                    """ Predicts the brand and extracts the highest confidence score """
                    preds = self.vision_model.predict(arr, verbose=0)
                    idx = np.argmax(preds[0])
                    brand = self.class_names[idx]
                    conf = preds[0][idx] * 100

                    self.vision_res.config(
                        text="Detected: " + str(brand).capitalize() + " (" + str(round(conf, 1)) + " %)", fg="#31A24C",
                        font=("Helvetica", 12, "bold"))

                    try:
                        """ Automagically inserts the detected brand into the text input """
                        self.inputs["brand"].set(str(brand).lower())
                        self.update_dropdowns(event=type('Event', (object,), {'widget': self.inputs["brand"]})())
                    except Exception as e:
                        pass

            except Exception as e:
                print(str(e))
        except Exception as e:
            print(str(e))

    def process_price(self):
        """
        Gathers all text inputs from the GUI form, validates and formats them,
        encodes categorical text inputs into numbers using the pre-fitted LabelEncoders,
        enforces strict column ordering, feeds the data array to the Random Forest model,
        and displays the calculated estimated market value in CZK.
        """
        try:
            data = {}
            """ Collects inputs into a dictionary format compatible with pandas DataFrames """
            for k, v in self.inputs.items():
                try:
                    val = str(v.get()).strip().lower()
                    data[k] = [val]
                except Exception as e:
                    print(str(e))

            try:
                df = pd.DataFrame(data)
                """ Replaces raw text entries with corresponding encoded integer labels """
                for c in self.cat_fields:
                    try:
                        df[c] = self.encoders[c].transform(df[c].astype(str))
                    except Exception as e:
                        df[c] = 0

                """ Ensures numerical inputs are treated strictly as floats/integers """
                num_cols = ["power_kw", "year", "mileage_km"]
                for c in num_cols:
                    try:
                        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
                    except Exception as e:
                        df[c] = 0

                """ Enforces the exact feature sequence required by Scikit-learn Random Forest """
                expected_order = ['brand', 'model', 'year', 'mileage_km', 'fuel', 'engine_type', 'power_kw',
                                  'drivetrain', 'transmission']
                try:
                    df = df[expected_order]
                except Exception as e:
                    pass

                """ Predicts the target value (price) and updates the UI label """
                pred = self.price_model.predict(df)
                price = int(pred[0])
                self.price_res.config(text=f"{price:,}".replace(",", " ") + " CZK", fg="#31A24C")
            except Exception as e:
                self.price_res.config(text="Error in inputs", fg="#E0245E")
        except Exception as e:
            print(str(e))


if __name__ == "__main__":
    """ 
    Standard python execution entry point. 
    Initializes the root Tkinter application process and starts the main event loop. 
    """
    try:
        root = tk.Tk()
        app = App(root)
        root.mainloop()
    except Exception as e:
        print(str(e))