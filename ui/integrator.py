import os
from tkinter import *
from tkinter import ttk, font, filedialog, messagebox
from PIL import ImageTk, Image
import pydicom as dicom
import cv2
import numpy as np
import csv
from model.load_model import ModelLoader
from model.predictor_img import Predictor
from services.history_service import HistoryService
from services.report_service import ReportService

# =========================
# PARCHE INTELIGENTE PDF
# =========================
try:
    import tkcap
    import tkinter.tix
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Sistema de Diagnóstico Asistido por IA")
        self.root.geometry("850x630")
        self.root.resizable(False, False)

        # =========================
        # CARGA DEL MODELO
        # =========================
        self.model = ModelLoader.get_model()
        if self.model is None:
            messagebox.showwarning(
                "Modelo no encontrado",
                "No se encontró el modelo .h5.\nLa app abrirá pero no podrá predecir."
            )
            self.predictor = None
        else:
            self.predictor = Predictor(self.model)

        # =========================
        # VARIABLES
        # =========================
        self.array = None
        self.label = ""
        self.proba = 0.0
        self.reportID = 0

        # =========================
        # DISEÑO VISUAL
        # =========================
        font_bold = font.Font(weight="bold")

        Label(self.root, text="DIAGNÓSTICO ASISTIDO POR IA",
              font=("Helvetica", 16, "bold")).place(x=280, y=10)

        Label(self.root, text="Radiografía Original",
              font=font_bold).place(x=100, y=60)

        Label(self.root, text="Análisis IA (Heatmap)",
              font=font_bold).place(x=550, y=60)

        # CANVAS VISUALES
        self.canvas1 = Label(self.root, bg="#E0E0E0", relief="sunken")
        self.canvas1.place(x=65, y=90, width=250, height=250)

        self.canvas2 = Label(self.root, bg="#E0E0E0", relief="sunken")
        self.canvas2.place(x=500, y=90, width=250, height=250)

        # DATOS PACIENTE
        Label(self.root, text="ID Paciente:",
              font=font_bold).place(x=65, y=380)

        self.entry_id = Entry(self.root, width=15)
        self.entry_id.place(x=180, y=380)

        Label(self.root, text="Diagnóstico:",
              font=font_bold).place(x=450, y=380)

        self.lbl_result = Label(self.root, text="---",
                                font=("Arial", 14), fg="blue")
        self.lbl_result.place(x=560, y=378)

        Label(self.root, text="Confianza:",
              font=font_bold).place(x=450, y=420)

        self.lbl_proba = Label(self.root, text="0.0%",
                               font=("Arial", 14))
        self.lbl_proba.place(x=560, y=418)

        # =========================
        # BOTONES
        # =========================
        self.btn_load = ttk.Button(
            self.root, text="📂 Cargar Imagen", command=self.load_image)
        self.btn_load.place(x=70, y=460, width=150, height=40)

        self.btn_predict = ttk.Button(
            self.root, text="⚡ Analizar",
            command=self.run_prediction, state="disabled")
        self.btn_predict.place(x=240, y=460, width=120, height=40)

        self.btn_save = ttk.Button(
            self.root, text="💾 Guardar",
            command=self.save_csv)
        self.btn_save.place(x=380, y=460, width=100, height=40)

        # PDF INTELIGENTE
        state_pdf = "normal" if PDF_AVAILABLE else "disabled"
        text_pdf = "📄 PDF" if PDF_AVAILABLE else "🚫 PDF (N/A)"

        self.btn_pdf = ttk.Button(
            self.root, text=text_pdf,
            command=self.create_pdf,
            state=state_pdf)
        self.btn_pdf.place(x=500, y=460, width=110, height=40)

        self.btn_clear = ttk.Button(
            self.root, text="🗑️ Limpiar",
            command=self.clear_data)
        self.btn_clear.place(x=630, y=460, width=110, height=40)

        self.root.mainloop()

    # ==========================================================
    # CARGAR IMAGEN
    # ==========================================================

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar Radiografía",
            filetypes=[("Imágenes Médicas", "*.dcm *.jpg *.jpeg *.png")]
        )

        if not file_path:
            return

        try:
            if file_path.lower().endswith(".dcm"):
                img = dicom.dcmread(file_path)
                img_array = img.pixel_array
                img_pil = Image.fromarray(img_array)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                self.array = img_array
            else:
                img = cv2.imread(file_path)
                img_array = np.asarray(img)
                img_pil = Image.fromarray(img_array)
                self.array = img_array

            img_resized = img_pil.resize(
                (250, 250), Image.Resampling.LANCZOS)
            self.tk_img1 = ImageTk.PhotoImage(img_resized)
            self.canvas1.config(image=self.tk_img1)

            if self.predictor is not None:
                self.btn_predict["state"] = "normal"

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")

    # ==========================================================
    # PREDICCIÓN
    # ==========================================================

    def run_prediction(self):
        if self.array is None or self.predictor is None:
            return

        try:
            label, proba, heatmap = self.predictor.predict(self.array)

            self.label = label
            self.proba = proba

            color = "green" if label.lower() == "normal" else "red"
            self.lbl_result.config(text=label, fg=color)
            self.lbl_proba.config(text=f"{proba:.2f}%")

            img_pil = Image.fromarray(heatmap)
            img_pil = img_pil.resize((250, 250),
                                     Image.Resampling.LANCZOS)
            self.tk_img2 = ImageTk.PhotoImage(img_pil)
            self.canvas2.config(image=self.tk_img2)

        except Exception as e:
            messagebox.showerror("Error en análisis", str(e))

    # ==========================================================
    # GUARDAR CSV
    # ==========================================================

    def save_csv(self):
        if not self.label:
            messagebox.showinfo("Aviso", "No hay resultado para guardar.")
            return

        try:
            with open("historial_pacientes.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.entry_id.get(),
                    self.label,
                    f"{self.proba:.2f}%"
                ])

            messagebox.showinfo("Guardado", "Datos guardados correctamente.")

        except Exception as e:
            messagebox.showerror("Error", str(e))


    # ==========================================================
    # GENERAR PDF
    # ==========================================================

    def create_pdf(self):
        if not PDF_AVAILABLE:
            messagebox.showinfo(
                "Función no disponible",
                "La generación de PDF requiere soporte Tix."
            )
            return

        try:
            cap = tkcap.CAP(self.root)
            img_name = f"Reporte_{self.reportID}.jpg"
            cap.capture(img_name)

            img = Image.open(img_name).convert("RGB")
            pdf_path = f"Reporte_{self.reportID}.pdf"
            img.save(pdf_path)

            os.remove(img_name)

            self.reportID += 1
            messagebox.showinfo("PDF", f"Reporte generado: {pdf_path}")

        except Exception as e:
            messagebox.showerror("Error PDF", str(e))


    # ==========================================================
    # LIMPIAR
    # ==========================================================

    def clear_data(self):
        self.canvas1.config(image="")
        self.canvas2.config(image="")
        self.lbl_result.config(text="---", fg="blue")
        self.lbl_proba.config(text="0.0%")
        self.entry_id.delete(0, END)
        self.btn_predict["state"] = "disabled"
        self.array = None
        self.label = ""
        self.proba = 0.0
