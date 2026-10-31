import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

#Membaca model pkl
model = joblib.load("model_logistic_kelulusan.pkl")

#Format data masukkan sesuai fitur data CSV
class DataMahasiswa(BaseModel):
    jam_belajar_per_minggu : float
    kehadiran_persen : float
    rata_rata_tugas : float
    nilai_uts : float

#Membuat endpoint/alamt akses api
@app.get('/')
def read_root():
    return {"message" : "Prediksi Kelulusan Mahasiswa"}

@app.post("/predict")
def predict(data : DataMahasiswa):
    fitur = [[data.jam_belajar_per_minggu, data.kehadiran_persen, data.rata_rata_tugas, data.nilai_uts]]
    hasil = model.predict(fitur)

    if hasil[0] == 0:
        return {"Hasil" : "Tidak Lulus"}
    else:
        return {"Hasil" : "Lulus"}