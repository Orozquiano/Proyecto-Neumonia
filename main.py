import tkinter as tk
from tkinter import ttk, font, filedialog, messagebox
from PIL import ImageTk, Image
import csv
import os

# --- PARCHE DE SEGURIDAD PARA EL ERROR DE TIX ---
try:
    # Intentamos importar tkcap. Si tu Python es muy nuevo y no tiene Tix, fallará aquí.
    import tkcap
    # Truco extra: a veces importa pero falla al usarse, verificamos tk.tix
    import tkinter.tix
    PDF_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError, Exception) as e:
    # Si falla, entramos aquí y evitamos que el programa se cierre
    print(f"⚠️ AVISO DE COMPATIBILIDAD: No se puede generar PDF en esta versión de Python.")
    print(f"   (Error técnico: {e})")
    print("   -> El sistema funcionará correctamente, pero el botón PDF estará desactivado.")
    PDF_AVAILABLE = False

# --- IMPORTACIONES DE TU ARQUITECTURA ---
from src.models.load_model import load_trained_model
from src.data.read_img import read_dicom_file, read_jpg_file
from src.integrator import predict_and_visualize

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección de Neumonía - Modular")
        self.root.geometry("850x630")
        self.root.resizable(False, False)

        # 1. CARGA DEL MODELO
        model_path = os.path.join("models", "conv_MLP_84.h5") 
        print("--- Iniciando Sistema ---")
        
        self.model = load_trained_model(model_path)
        
        if self.model is None:
            messagebox.showwarning("Falta el Modelo", f"No se encontró el archivo .h5 en la carpeta 'models'.\nLa App abrirá, pero no podrás predecir.")

        # Variables
        self.img_path = None
        self.array = None
        self.last_label = ""
        self.last_proba = 0.0
        self.reportID = 0

        # --- GUI ---
        font_bold = font.Font(weight="bold")

        tk.Label(root, text="DIAGNÓSTICO ASISTIDO POR IA", font=("Helvetica", 16, "bold")).place(x=280, y=10)
        tk.Label(root, text="Radiografía Original", font=font_bold).place(x=100, y=60)
        tk.Label(root, text="Análisis de IA (Heatmap)", font=font_bold).place(x=550, y=60)

        # Canvas
        self.canvas1 = tk.Label(root, bg="#E0E0E0", relief="sunken")
        self.canvas1.place(x=65, y=90, width=250, height=250)
        
        self.canvas2 = tk.Label(root, bg="#E0E0E0", relief="sunken")
        self.canvas2.place(x=500, y=90, width=250, height=250)

        # Datos
        tk.Label(root, text="ID Paciente:", font=font_bold).place(x=65, y=380)
        self.entry_id = tk.Entry(root, width=15)
        self.entry_id.place(x=180, y=380)

        tk.Label(root, text="Diagnóstico:", font=font_bold).place(x=450, y=380)
        self.lbl_result = tk.Label(root, text="---", font=("Arial", 14), fg="blue")
        self.lbl_result.place(x=560, y=378)

        tk.Label(root, text="Confianza:", font=font_bold).place(x=450, y=420)
        self.lbl_proba = tk.Label(root, text="0.0%", font=("Arial", 14))
        self.lbl_proba.place(x=560, y=418)

        # Botones
        self.btn_load = ttk.Button(root, text="📂 Cargar Imagen", command=self.load_image)
        self.btn_load.place(x=70, y=460, width=150, height=40)

        self.btn_predict = ttk.Button(root, text="⚡ Analizar", command=self.run_prediction, state="disabled")
        self.btn_predict.place(x=240, y=460, width=120, height=40)

        self.btn_save = ttk.Button(root, text="💾 Guardar", command=self.save_csv)
        self.btn_save.place(x=380, y=460, width=100, height=40)
        
        # Botón PDF INTELIGENTE: Si no hay PDF_AVAILABLE, se desactiva y cambia el texto
        state_pdf = "normal" if PDF_AVAILABLE else "disabled"
        text_pdf = "📄 PDF" if PDF_AVAILABLE else "🚫 PDF (N/A)"
        
        self.btn_pdf = ttk.Button(root, text=text_pdf, command=self.create_pdf, state=state_pdf)
        self.btn_pdf.place(x=500, y=460, width=100, height=40)

        self.btn_clear = ttk.Button(root, text="🗑️ Limpiar", command=self.clear_data)
        self.btn_clear.place(x=620, y=460, width=100, height=40)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Seleccionar Radiografía", 
                                             filetypes=[("Imágenes Médicas", "*.dcm *.jpg *.png *.jpeg")])
        if not file_path:
            return
            
        self.img_path = file_path
        
        if file_path.lower().endswith('.dcm'):
            self.array, img_pil = read_dicom_file(file_path)
        else:
            self.array, img_pil = read_jpg_file(file_path)

        img_resized = img_pil.resize((250, 250), Image.Resampling.LANCZOS)
        self.tk_image1 = ImageTk.PhotoImage(img_resized)
        self.canvas1.config(image=self.tk_image1)
        
        if self.model is not None:
            self.btn_predict["state"] = "normal"

    def run_prediction(self):
        if self.array is None or self.model is None:
            return

        try:
            label, proba, heatmap_arr = predict_and_visualize(self.array, self.model)
            
            self.last_label = label
            self.last_proba = proba

            color = "red" if label != "Normal" else "green"
            self.lbl_result.config(text=label, fg=color)
            self.lbl_proba.config(text=f"{proba:.2f}%")

            img_pil = Image.fromarray(heatmap_arr)
            img_pil = img_pil.resize((250, 250), Image.Resampling.LANCZOS)
            self.tk_image2 = ImageTk.PhotoImage(img_pil)
            self.canvas2.config(image=self.tk_image2)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el análisis:\n{e}")
            print(e)

    def save_csv(self):
        try:
            with open("historial_pacientes.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.entry_id.get(), self.last_label, f"{self.last_proba:.2f}%"])
            messagebox.showinfo("Guardado", "Datos guardados.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_pdf(self):
        if not PDF_AVAILABLE:
            messagebox.showinfo("Función no disponible", "La generación de PDF requiere Python 3.11 o inferior (Librería Tix faltante).")
            return

        try:
            cap = tkcap.CAP(self.root)
            img_name = f"Reporte_{self.reportID}.jpg"
            cap.capture(img_name)
            
            img = Image.open(img_name).convert("RGB")
            pdf_path = f"Reporte_{self.reportID}.pdf"
            img.save(pdf_path)
            
            self.reportID += 1
            os.remove(img_name)
            messagebox.showinfo("PDF", f"Reporte generado: {pdf_path}")
        except Exception as e:
            messagebox.showerror("Error PDF", f"No se pudo crear el PDF.\n{e}")

    def clear_data(self):
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