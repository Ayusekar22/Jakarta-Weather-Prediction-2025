## Weather Prediction Using Two Stage Model (Random Forest Classifier + XGBRegressor) In DKI Jakarta 2025

## Description 
Indonesia adalah negara beriklim tropis yang memiliki dua musim utama, yaitu musim kemarau dan musim penghujan. Fluktuasi curah hujan, terutama di musim penghujan, seringkali sulit diprediksi. Oleh karena itu, penelitian ini bertujuan untuk membangun model prediksi yang dapat memprakirakan terjadinya hujan pada hari berikutnya (besok) berdasarkan data historis meteorologi yang diperoleh dari bank data BMKG.

## Dataset 
Dataset yang digunakan dalam penelitian ini adalah data meteorologi historis harian yang bersumber dari portal data online resmi Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) Indonesia dengan pemilihan lokasi di Jakarta Pusat. 
- Sumber: dataonline.bmkg.go.id
- Rentang Waktu: Data diambil untuk periode 1 Januari 2015 hingga 07 Desember 2025.
- Volume Data: Dataset ini terdiri dari 4019 baris data (observasi), di mana setiap baris merepresentasikan data cuaca harian.

Dataset ini memiliki 11 atribut orisinal yang mencatat berbagai parameter cuaca dengan keterangan sebagai berikut  :

| Nama Kolom | Keterangan | Satuan |
| :--- | :--- | :--- |
| `tanggal` | Waktu pencatatan data | - |
| `TN` | Temperatur Minimum | °C |
| `TX` | Temperatur Maksimum | °C |
| `TAVG` | Temperatur Rata-rata | °C |
| `RH_AVG` | Kelembapan Rata-rata | % |
| `RR` | Curah Hujan | mm |
| `SS` | Lamanya Penyinaran Matahari | jam |
| `FF_X` | Kecepatan Angin Maksimum | m/s |
| `DDD_X` | Arah Angin saat Kecepatan Maks. | ° (derajat) |
| `FF_AVG` | Kecepatan Angin Rata-rata | m/s |
| `DDD_CAR` | Arah Angin Terbanyak | ° (derajat) |

## Flowchart
Dibawah ini merupakan alur dari project ini 
![Alur Project](Flowchart.png)

## Data Understanding



## Objectives 

## Data Preprocessing 
# ## Menyatukan Data Historis Menjadi 1 Excel
# ## Cek Missing Values
# ## Melakukan Data Mapping 
# ## Mengisi missing values menggunakan Random Forest 
# ## Melakukan Feature Engineering



## Model Development

## Results 

## Conclusion 

=======

