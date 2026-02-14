import tkcap
from PIL import Image
from tkinter.messagebox import showinfo

class ReportService:

    def __init__(self):
        self.reportID = 0
    # Método para generar un PDF a partir de una captura de pantalla del root dado
    def generate(self, root):
        cap = tkcap.CAP(root)

        img_name = f"Reporte{self.reportID}.jpg"
        cap.capture(img_name)

        img = Image.open(img_name)
        img = img.convert("RGB")

        pdf_path = f"Reporte{self.reportID}.pdf"
        img.save(pdf_path)

        self.reportID += 1

        showinfo(title="PDF", message="El PDF fue generado con éxito.")
