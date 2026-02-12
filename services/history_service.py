import csv
from tkinter.messagebox import showinfo

class HistoryService:

    @staticmethod
    def save(patient_id, label, proba):
        with open("historial.csv", "a") as csvfile:
            writer = csv.writer(csvfile, delimiter="-")
            writer.writerow([patient_id, label, f"{proba:.2f}%"])

        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")
