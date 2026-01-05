## Weather Prediction Using Two Stage Model (Random Forest Classifier + XGBRegressor) In DKI Jakarta 2025

## Description 
Indonesia adalah negara beriklim tropis yang memiliki dua musim utama, yaitu musim kemarau dan musim penghujan. Fluktuasi curah hujan, terutama di musim penghujan, seringkali sulit diprediksi. Oleh karena itu, penelitian ini bertujuan untuk membangun model prediksi yang dapat memprakirakan terjadinya hujan pada hari berikutnya (besok) berdasarkan data historis meteorologi yang diperoleh dari bank data BMKG.

## Problem

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

![Alur Project2](Gambar\Flowchart.png)

## Data Understanding

### Rainfall Distribution Analysis
![Distribusi Hujan](Gambar\Distribusi_Curah.png)
Dari graphic ini menunjukkan dominasi nilai 0 mm, yang mengindikasikan bahwa sebagian besar hari tidak mengalami hujan. Sementara itu, nilai curah hujan non-nol memiliki distribusi yang sangat skewed ke arah kanan, dengan ekor panjang (heavy-tailed) yang merepresentasikan kejadian hujan kategori sedang hingga ekstrem.

![Boxplot Hujan](Gambar\Boxplot_RR.png)
Disini terlihat keberadaan outlier ekstrem yang jumlahnya relatif sedikit tetapi bernilai jauh lebih besar dibandingkan mayoritas data. Hal ini menegaskan bahwa data curah hujan bersifat zero-inflated dan heavy-tailed. 

### Correlation Analysis
![Heatmap](Gambar\Heatmap.png)
Disini digunakan Heatmap korelasi Spearman pada data mentah menunjukkan bahwa hubungan langsung antara curah hujan dan variabel meteorologi relatif lemah hingga moderat. Korelasi yang muncul lebih mencerminkan hubungan monotonic jangka pendek. Temuan ini mengindikasikan bahwa ketergantungan curah hujan tidak hanya bersifat instan, melainkan bergantung pada pola historis, sehingga diperlukan fitur berbasis waktu seperti lag, rolling statistics, dan encoding musiman.

### Baseline Models
PERSISTENCE BASELINE RESULTS:
MAE  : 9.285
RMSE : 21.275

BASELINE REGRESSION RESULTS:
MAE:  10.681 mm
RMSE: 20.818 mm

### Key Findings from Data Understanding
Berdasarkan analisis distribusi, korelasi, dan baseline model, curah hujan harian memiliki karakteristik zero-inflated, heavy-tailed, serta ketergantungan temporal yang kuat. Oleh karena itu, pendekatan two-stage modeling dipilih untuk memisahkan prediksi kejadian hujan dan intensitas hujan, sehingga diharapkan mampu menghasilkan prediksi yang lebih stabil dan representatif dibandingkan pendekatan satu tahap.


## Data Preprocessing 
- Menyatukan Data Historis Menjadi 1 Excel
- Cek Missing Values
- Melakukan Data Mapping 
- Mengisi missing values menggunakan Random Forest 
- Melakukan Feature Engineering

## Feature Engineering
1. Mengurutkan Data Berdasarkan Tanggal 

    Data diurutkan secara kronologis berdasarkan kolom Tanggal untuk memastikan struktur time series terjaga dan mencegah data leakage pada proses pemodelan.

2. Membuat Lag Features
    Lag features dibuat untuk variabel Curah Hujan (mm) serta variabel cuaca yang memiliki korelasi tinggi terhadapnya, yaitu:

    * Kelembapan Rata-rata
    * Temperatur Rata-rata
    * Temperatur Maksimum
    * Temperatur Minimum
    * Lamanya Penyinaran Matahari

    Dengan lag yang digunakan yakni dari 1 hingga 4 hari sebelumnya 

3. Membuat Rolling Mean (Moving Average)

    Rolling mean dilakukan untuk variabel-variabel yang berkorelasi tinggi dengan curah hujan menggunakan gap waktu 3 dan 7 hari. Setelah itu rolling mean di-shift satu hari ke belakang agar hanya memanfaatkan informasi historis dan tidak terjadi leakage.

4. Membuat Exponentially Weighted Moving Average (EWMA)

    EWMA dibuat khusus untuk variabel Curah Hujan (mm) dengan periode 3 dan 7 hari. Metode ini memberikan bobot lebih besar pada observasi terbaru sehingga lebih sensitif terhadap perubahan pola curah hujan dibanding rolling mean biasa.

5. Rolling Standard Deviation

    Rolling standard deviation digunakan untuk mengukur tingkat variasi curah hujan dalam jendela waktu tertentu (7 hari).

5. Cylical Encoding 

    Cyclical encoding diterapkan pada informasi waktu untuk menangkap pola musiman curah hujan. Kolom Tanggal diturunkan menjadi fitur dayofyear, kemudian direpresentasikan menggunakan fungsi sinus dan cosinus. Pendekatan ini digunakan agar model memahami bahwa waktu bersifat siklikal.

6. Fitur Kejadian Hujan Ekstrem
    
    Fitur kejadian hujan ekstrem dibuat untuk menangkap kondisi curah hujan yang tidak biasa. Ambang hujan ekstrem ditentukan menggunakan quantile ke-90 dari distribusi curah hujan.Fitur ini bertujuan untuk membantu model mengenali pengaruh kejadian hujan ekstrem sebelumnya terhadap kejadian hujan di hari berikutnya.

## Model Development

### Data Modelling
Karena curah hujan memiliki karakteristik zero-inflated, sehingga digunakan pendekatan two-stage modeling : 
1. Classification untuk memprediksi kejadian hujan.
2. Regression untuk memprediksi intensitas hujan hanya pada hari hujan.

Dengan menggunakan cara ini dapat membantu mengurangi bias data nol. 

### Data Splitting 
Data dibagi jadi 80% train dan 20% test secara time-ordered (tanpa shuffle) untuk menjaga struktur time series dan mencegah data leakage.

### Classification
Tahap pertama bertujuan untuk memprediksi kejadian hujan sebagai masalah klasifikasi biner, dengan label 0 untuk tidak hujan dan 1 untuk hujan. Model yang digunakan adalah Random Forest Classifier dengan class_weight="balanced" untuk menangani ketidakseimbangan kelas. Model menghasilkan probabilitas hujan, yang kemudian dikonversi menjadi prediksi kelas menggunakan threshold 0.5.

### Regression
Tahap kedua memprediksi intensitas curah hujan dan hanya dilatih menggunakan data pada hari-hari dengan hujan aktual. Target regresi adalah curah hujan hari berikutnya, yang ditransformasi menggunakan log untuk mengurangi skewness distribusi. Model yang digunakan adalah XGBoost Regressor, dengan pembagian data 80:20 secara time-ordered.

## Results and Findings

### The Impact of Feature Engineering
Penerapan feature engineering berbasis waktu memberikan peningkatan performa yang konsisten. Akurasi klasifikasi meningkat dari 66% pada baseline menjadi 71% pada model final, atau sekitar +5% absolut, sementara ROC-AUC mencapai 0.77, menunjukkan pemisahan kelas hujan dan tidak hujan yang lebih baik.

Pada tahap regresi, pemodelan intensitas hujan setelah pemisahan kejadian hujan menurunkan MAE dari 10.681 mm menjadi 6.764 mm (penurunan sekitar 36%) dan RMSE dari 20.818 mm menjadi 16.452 mm. Hal ini menegaskan bahwa fitur temporal seperti lag, rolling statistics, dan EWMA secara signifikan meningkatkan kemampuan model dalam mempelajari pola hujan.
### Major factors contributing to rainfall

![Top10Feature](Top10FeatureImportances.png)

Analisis korelasi dan kontribusi fitur menunjukkan bahwa curah hujan lebih dipengaruhi oleh kondisi atmosfer lembap dan pola historis dibandingkan kondisi instan. Variabel seperti kelembapan rata-rata, curah hujan pada hari-hari sebelumnya, serta penurunan durasi penyinaran matahari memiliki pengaruh lebih besar dibandingkan variabel temperatur maksimum.

Hal ini mengindikasikan bahwa hujan di wilayah tropis seperti Jakarta bersifat akumulatif dan dipengaruhi oleh kondisi atmosfer yang berkembang dalam beberapa hari, bukan hanya oleh satu snapshot cuaca harian.

### Pendekatan Two Stage Model 
Pendekatan one-stage regression cenderung bias terhadap hari tanpa hujan dan gagal menangkapx intensitas hujan ekstrem. Dengan memisahkan proses prediksi hujan atau tidak dengan curah hujannya, two-stage modeling memungkinkan setiap model fokus pada pola yang lebih homogen, sehingga menghasilkan prediksi yang lebih stabil dan akurat.

## Conclusion 


=======

