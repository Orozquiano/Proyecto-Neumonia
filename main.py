import tkinter as tk
from tkinter import ttk, font, filedialog, messagebox
from PIL import ImageTk, Image
import csv
import tkcap
import os

# --- AQUÍ ESTÁ LA MAGIA: IMPORTAMOS TUS NUEVOS MÓDULOS ---
# Fíjate cómo llamamos a las carpetas que creaste (src.models, src.data, etc.)
from src.models.load_model import load_trained_model
from src.data.read_img import read_dicom_file, read_jpg_file
from src.integrator import predict_and_visualize

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección de Neumonía - Modular")
        self.root.geometry("850x630")
        self.root.resizable(False, False)

        # 1. CARGA DEL MODELO (Busca en la carpeta models)
        model_path = os.path.join("models", "conv_MLP_84.h5") 
        print("--- Iniciando Sistema ---")
        
        # Usamos tu script load_model.py
        self.model = load_trained_model(model_path)
        
        if self.model is None:
            # Si no encuentra el .h5, avisa pero abre la ventana igual
            messagebox.showwarning("Falta el Modelo", f"No se encontró el archivo .h5 en la carpeta 'models'.\nLa App abrirá, pero no podrás predecir.")

        # Variables para guardar datos temporalmente
        self.img_path = None
        self.array = None
        self.last_label = ""
        self.last_proba = 0.0
        self.reportID = 0

        # --- DISEÑO DE LA VENTANA (GUI) ---
        font_bold = font.Font(weight="bold")

        # Títulos
        tk.Label(root, text="DIAGNÓSTICO ASISTIDO POR IA", font=("Helvetica", 16, "bold")).place(x=280, y=10)
        tk.Label(root, text="Radiografía Original", font=font_bold).place(x=100, y=60)
        tk.Label(root, text="Análisis de IA (Heatmap)", font=font_bold).place(x=550, y=60)

        # Áreas de Imagen (Canvas)
        self.canvas1 = tk.Label(root, bg="#E0E0E0", relief="sunken")
        self.canvas1.place(x=65, y=90, width=250, height=250)
        
        self.canvas2 = tk.Label(root, bg="#E0E0E0", relief="sunken")
        self.canvas2.place(x=500, y=90, width=250, height=250)

        # Resultados y Datos
        tk.Label(root, text="ID Paciente:", font=font_bold).place(x=65, y=380)
        self.entry_id = tk.Entry(root, width=15)
        self.entry_id.place(x=180, y=380)

        tk.Label(root, text="Diagnóstico:", font=font_bold).place(x=450, y=380)
        self.lbl_result = tk.Label(root, text="---", font=("Arial", 14), fg="blue")
        self.lbl_result.place(x=560, y=378)

        tk.Label(root, text="Confianza:", font=font_bold).place(x=450, y=420)
        self.lbl_proba = tk.Label(root, text="0.0%", font=("Arial", 14))
        self.lbl_proba.place(x=560, y=418)

        # --- BOTONES DE CONTROL ---
        self.btn_load = ttk.Button(root, text="📂 Cargar Imagen", command=self.load_image)
        self.btn_load.place(x=70, y=460, width=150, height=40)

        self.btn_predict = ttk.Button(root, text="⚡ Analizar", command=self.run_prediction, state="disabled")
        self.btn_predict.place(x=240, y=460, width=120, height=40)

        self.btn_save = ttk.Button(root, text="💾 Guardar", command=self.save_csv)
        self.btn_save.place(x=380, y=460, width=100, height=40)
        
        self.btn_pdf = ttk.Button(root, text="📄 PDF", command=self.create_pdf)
        self.btn_pdf.place(x=500, y=460, width=100, height=40)

        self.btn_clear = ttk.Button(root, text="🗑️ Limpiar", command=self.clear_data)
        self.btn_clear.place(x=620, y=460, width=100, height=40)

    # --- FUNCIONES LÓGICAS ---

    def load_image(self):
        # Abrir explorador de archivos
        file_path = filedialog.askopenfilename(title="Seleccionar Radiografía", 
                                             filetypes=[("Imágenes DICOM/IMG", "*.dcm *.jpg *.png *.jpeg")])
        if not file_path:
            return
            
        self.img_path = file_path
        
        # 1. Llamamos a tu módulo de lectura (src/data/read_img.py)
        if file_path.lower().endswith('.dcm'):
            self.array, img_pil = read_dicom_file(file_path)
        else:
            self.array, img_pil = read_jpg_file(file_path)

        # Mostrar imagen en pantalla (Resize solo visual)
        img_resized = img_pil.resize((250, 250), Image.Resampling.LANCZOS)
        self.tk_image1 = ImageTk.PhotoImage(img_resized)
        self.canvas1.config(image=self.tk_image1)
        
        # Habilitar botón de predicción solo si el modelo existe
        if self.model is not None:
            self.btn_predict["state"] = "normal"

    def run_prediction(self):
        if self.array is None or self.model is None:
            return

        try:
            # 2. Llamamos a tu Integrador Optimizado (src/integrator.py)
            label, proba, heatmap_arr = predict_and_visualize(self.array, self.model)
            
            # Guardar datos para el reporte
            self.last_label = label
            self.last_proba = proba

            # Actualizar textos de resultado
            color = "red" if label != "Normal" else "green"
            self.lbl_result.config(text=label, fg=color)
            self.lbl_proba.config(text=f"{proba:.2f}%")

            # Mostrar Heatmap (Grad-CAM)
            img_pil = Image.fromarray(heatmap_arr)
            img_pil = img_pil.resize((250, 250), Image.Resampling.LANCZOS)
            self.tk_image2 = ImageTk.PhotoImage(img_pil)
            self.canvas2.config(image=self.tk_image2)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el análisis:\n{e}")
            print(e)

    def save_csv(self):
        # Guardar en Excel/CSV
        try:
            with open("historial_pacientes.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.entry_id.get(), self.last_label, f"{self.last_proba:.2f}%"])
            messagebox.showinfo("Guardado", "Datos guardados en historial_pacientes.csv")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_pdf(self):
        # Generar reporte PDF con tkcap
        try:
            cap = tkcap.CAP(self.root)
            img_name = f"Reporte_{self.reportID}.jpg"
            
            # Captura la pantalla de la app
            cap.capture(img_name)
            
            # Convierte a PDF
            img = Image.open(img_name).convert("RGB")
            pdf_path = f"Reporte_{self.reportID}.pdf"
            img.save(pdf_path)
            
            self.reportID += 1
            os.remove(img_name) # Borra la imagen temporal
            messagebox.showinfo("PDF", f"Reporte generado: {pdf_path}")
        except Exception as e:
            messagebox.showerror("Error PDF", f"No se pudo crear el PDF.\n{e}")

    def clear_data(self):
        # Limpiar todo para el siguiente paciente
        self.canvas1.config(image='')
        self.canvas2.config(image='')
        self.lbl_result.config(text="---", fg="blue")
        self.lbl_proba.config(text="0.0%")
        self.entry_id.delete(0, tk.END)
        self.btn_predict["state"] = "disabled"
        self.img_path = None
        self.array = None

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()