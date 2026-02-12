import csv
from tkinter import *
from tkinter import ttk, font, filedialog
from tkinter.messagebox import WARNING, askokcancel, showinfo
from PIL import ImageTk, Image
import cv2
import numpy as np
import tkcap

from model.model_loader import ModelLoader
from model.predictor import Predictor
from image.reader import ImageReader
from image.preprocessor import Preprocessor
from services.history_service import HistoryService
from services.report_service import ReportService
import pydicom as dicom


class App:
    def __init__(self):
        self.root = Tk()
        self.model = ModelLoader.get_model()
        self.predictor = Predictor(self.model)
        self.root.title("Herramienta para la detección rápida de neumonía")

        #   BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        #   LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        #   TWO STRING VARIABLES TO CONTAIN ID AND RESULT
        self.ID = StringVar()
        self.result = StringVar()

        #   TWO INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        #   GET ID
        self.ID_content = self.text1.get()

        #   TWO IMAGE INPUT BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        #   BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        #   WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        #   FOCUS ON PATIENT ID
        self.text1.focus_set()

        #  se reconoce como un elemento de la clase
        self.array = None

        #   NUMERO DE IDENTIFICACIÓN PARA GENERAR PDF
        self.reportID = 0

        #   RUN LOOP
        self.root.mainloop()

    #   METHODS
    
    def read_dicom_file(self, path):
        self.img = dicom.dcmread(path) #Se cambia pydicom.read_file por dicom.dcmread (Mas actualizado)
        self.img_array = self.img.pixel_array
        self.img2show = Image.fromarray(self.img_array)
        self.img2 = self.img_array.astype(float)
        self.img2 = (np.maximum(self.img2, 0) / self.img2.max()) * 255.0
        self.img2 = np.uint8(self.img2)
        self.img_RGB = cv2.cvtColor(self.img2, cv2.COLOR_GRAY2RGB)
        return self.img_RGB, self.img2show


    def read_jpg_file(self,path):
        self.img = cv2.imread(path)
        self.img_array = np.asarray(self.img)
        self.img2show = Image.fromarray(self.img_array)
        self.img2 = self.img_array.astype(float)
        self.img2 = (np.maximum(self.img2, 0) / self.img2.max()) * 255.0
        self.img2 = np.uint8(self.img2)
        return self.img2, self.img2show
    
        
    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("JPG", "*.jpg"),
                ("PNG", "*.png"),
            ),
        )

        if filepath:
        # Detectar tipo de archivo
            if filepath.lower().endswith(".dcm"):
                self.array, self.img2show = self.read_dicom_file(filepath)
            else:
                self.array, self.img2show = self.read_jpg_file(filepath)

            # Redimensionar imagen
            self.img1 = self.img2show.resize((250, 250), Image.Resampling.LANCZOS) # Se canbia por que en versiones nuevas ya esta deprecado
            self.img1 = ImageTk.PhotoImage(self.img1)

            self.text_img1.delete("1.0", END)
            self.text_img1.image_create(END, image=self.img1)

            self.button1["state"] = "enabled"


    def run_model(self):
        self.label, self.proba, self.heatmap = self.predictor.predict(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.Resampling.LANCZOS) # Se canbia por que en versiones nuevas ya esta deprecado
        self.img2 = ImageTk.PhotoImage(self.img2)
        print("OK")
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    def save_results_csv(self):
        with open("historial.csv", "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        cap = tkcap.CAP(self.root)
        ID = "Reporte" + str(self.reportID) + ".jpg"
        self.img = cap.capture(ID)
        self.img = Image.open(ID)
        self.img = self.img.convert("RGB")
        pdf_path = r"Reporte" + str(self.reportID) + ".pdf"
        self.img.save(pdf_path)
        self.reportID += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        answer = askokcancel(
                title="Confirmación",
                message="Se borrarán todos los datos.",
                icon=WARNING
            )

        if answer:
            self.text1.delete(0, END) # Se cambia por error TypeError: bad text index
            self.text2.delete("1.0", END)
            self.text3.delete("1.0", END)
            self.text_img1.delete("1.0", END)
            self.text_img2.delete("1.0", END)

            showinfo(title="Borrar", message="Los datos se borraron con éxito")