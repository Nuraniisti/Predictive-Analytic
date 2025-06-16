# Heart Attack Prediction using Logistic Regression and XGBoost


## Domain Proyek
Penyakit kardiovaskular (CVD) adalah penyebab kematian nomor satu di dunia, menyumbang sekitar 17,9 juta kematian setiap tahun menurut World Health Organization (WHO). Serangan jantung, salah satu manifestasi utama CVD, sering terjadi secara mendadak dan memerlukan deteksi dini untuk mencegah konsekuensi fatal. Faktor risiko seperti usia, tekanan darah, kadar gula darah, serta biomarker seperti CK-MB dan troponin memainkan peran penting dalam diagnosis. Dengan kemajuan machine learning, data klinis dapat digunakan untuk memprediksi risiko serangan jantung secara akurat, mendukung tenaga medis dalam pengambilan keputusan cepat.

Mengapa Masalah Ini Harus Diselesaikan?
Prediksi dini serangan jantung dapat meningkatkan peluang pasien untuk mendapatkan intervensi medis tepat waktu, mengurangi angka kematian, dan menekan biaya perawatan. Model prediktif berbasis machine learning memungkinkan identifikasi pasien berisiko tinggi secara efisien, terutama di fasilitas kesehatan dengan sumber daya terbatas. Selain itu, model ini dapat memberikan wawasan tentang faktor risiko utama, membantu pencegahan dan edukasi kesehatan.

Referensi :   
[1] World Health Organization, "Cardiovascular diseases (CVDs)," WHO, 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds).      
[2] R. Shouval et al., "Machine learning for prediction of 30-day mortality in patients with acute myocardial infarction," European Heart Journal, vol. 40, no. 14, pp. 1153–1161, 2019, doi: 10.1093/eurheartj/ehz057.


## Business Understanding
Problem Statements
- Bagaimana cara memprediksi keberadaan serangan jantung (positif/negatif) berdasarkan fitur klinis seperti usia, jenis kelamin, detak jantung, tekanan darah, kadar gula darah, CK-MB, dan troponin?   
Data klinis kompleks dan memerlukan analisis canggih untuk menghasilkan prediksi yang akurat, yang dapat membantu tenaga medis dalam diagnosis dini.  
- Bagaimana cara memastikan model prediktif memiliki performa yang optimal, untuk mendukung aplikasi medis?   
Dalam konteks medis, false negatives (gagal mendeteksi serangan jantung) dapat berakibat fatal, sehingga model harus dioptimalkan untuk recall tinggi tanpa mengorbankan precision.

Goals    
tujuan untuk menyelesaikan permasalahan diatas yaitu :  
- Membangun model machine learning yang dapat memprediksi keberadaan serangan jantung dengan akurasi, precision, dan recall yang tinggi berdasarkan fitur klinis.  
- Memilih model terbaik yang andal untuk konteks medis, dengan fokus pada minimisasi false negatives dan interpretasi faktor risiko utama.

Solution Statements
- Menggunakan algoritma Logistic Regression sebagai baseline model karena sederhana dan kemampuan interpretasi yang baik dalam klasifikasi biner. Selain itu, menerapkan Gradient Boosting (XGBoost) untuk meningkatkan performa dengan memanfaatkan pendekatan ensemble yang kuat. 
- Model akan dievaluasi menggunakan metrik akurasi, precision, recall, F1-score, dan AUC-ROC untuk mendapatkan model terbaik.


## Data Understanding
Dataset yang digunakan adalah Heart Disease Classification Dataset dengan 1319 sampel dan 9 kolom (8 fitur input, 1 kolom target). Dataset ini mencakup faktor risiko dan biomarker yang berkontribusi pada serangan jantung.   

| Jumlah baris     |  Jumlah kolom    |    
|------------------|------------------|        
|      1.319       |       9          |       

|  tipe data  |  jumlah kolom  |
|-------------|----------------|
|    integer  |        5       |
|    float    |        3       |
|    object   |        1       |

Title : Heart Disease Classification Dataset   
Source : Kaggle (https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset/data)  
Owner : Bharath_011   
License : Open Database  
Visibility : Publik   
Usability : 9.41

- **Numerik** : `age`, `impulse`, `pressurehight`, `pressurelow`, `glucose`, `kcm`, `troponin`
- **Target** : `class` yang dikonversi ke format numerik: 0 (negative), 1 (positive)

### Variabel

| Kolom            | Deskripsi                                                       |
|------------------|-----------------------------------------------------------------|
| age              | Usia pasien (numerik, tahun)                                    |
| gender           | Jenis kelamin ((0 = perempuan, 1 = laki-laki))                  |
| impulse          | Denyut jantung (numerik, denyut per menit)                      |
| pressurehight    | Tekanan darah atas (numerik, mmHg)                              |
| pressurelow      | Tekanan darah bawah (numerik, mmHg)                             |
| glucose          | Kadar gula darah (numerik, mg/dL)                               |
| kcm              | Kalium, kadar CK-MB, biomarker kerusakan jantung (numerik).     |
| troponin         | Kadar troponin, biomarker spesifik serangan jantung (numerik)   |
| class            | Keberadaan serangan jantung. Target : 0 (negatif), 1 (positif)  |


## Exploratory Data Analysis (EDA)
Deskripsi tabel

| No | Column        | Non-Null Count | Dtype   |
| -- | ------------- | -------------- | ------- |
| 0  | age           | 1319           | int64   |
| 1  | gender        | 1319           | int64   |
| 2  | impluse       | 1319           | int64   |
| 3  | pressurehight | 1319           | int64   |
| 4  | pressurelow   | 1319           | int64   |
| 5  | glucose       | 1319           | float64 |
| 6  | kcm           | 1319           | float64 |
| 7  | troponin      | 1319           | float64 |
| 8  | class         | 1319           | object  |

Mengubah nama kolom `impluse` menjadi `impulse`

|  Data duplikat  |
|-----------------|
|        0        |

Pada tabel diatas, menunjukkan bahwa tidak terdapat missing value dan data duplikat

Infomasi statistik masing-masing kolom
| Statistik | age    | gender | impulse | pressurehight | pressurelow | glucose | kcm    | troponin |
| --------- | ------ | ------ | ------- | ------------- | ----------- | ------- | ------ | -------- |
| count     | 1319   | 1319   | 1319    | 1319          | 1319        | 1319    | 1319   | 1319     |
| mean      | 56.19  | 0.66   | 78.34   | 127.17        | 72.27       | 146.63  | 15.27  | 0.36     |
| std       | 13.65  | 0.47   | 51.63   | 26.12         | 14.03       | 74.92   | 46.33  | 1.15     |
| min       | 14.00  | 0.00   | 20.00   | 42.00         | 38.00       | 35.00   | 0.32   | 0.001    |
| 25%       | 47.00  | 0.00   | 64.00   | 110.00        | 62.00       | 98.00   | 1.66   | 0.006    |
| 50%       | 58.00  | 1.00   | 74.00   | 124.00        | 72.00       | 116.00  | 2.85   | 0.014    |
| 75%       | 65.00  | 1.00   | 85.00   | 143.00        | 81.00       | 169.50  | 5.81   | 0.086    |
| max       | 103.00 | 1.00   | 1111.00 | 223.00        | 154.00      | 541.00  | 300.00 | 10.30    |

Dari tabel diatas, diketahui bahwa :    
- Age (usia)
  - mean = 56.19 (~56 tahun)
  - rentang usia sangat luas, yaitu min = 14 tahun dan max = 103 tahun
  - Distribusi sedikit miring ke kiri (median 58 > mean 56.19), dengan lebih banyak pasien di usia tua.
  - Sebagian besar pasien (50%) berusia antara 47–65 tahun (Q1–Q3), menunjukkan fokus pada kelompok usia menengah hingga lanjut.

- Gender (jenis kelamin)
  - mean = 0.66 menunjukkan ~66% pasien adalah laki-laki dan ~34% perempuan.
  - konsisten dengan literatur medis, di mana laki-laki memiliki risiko CVD lebih tinggi dibandingkan perempuan pada usia tertentu.

- Impulse (detak jantung)
  - Detak jantung rata-rata (78.34 bpm) berada dalam rentang normal (60–100 bpm).
  - Sebagian besar pasien (50%) memiliki detak jantung antara 64–85 bpm, yang normal.

- pressurehight (Tekanan Darah Sistolik) dan Pressurelow (Tekanan Darah Diastolik)
  - Sebagian besar pasien memiliki tekanan sistolik 110–143 mmHg dan diastolik 62–81 mmHg.

- Glucose (kadar gula darah)
  - Sebagian besar pasien (50%) memiliki kadar gula 98–169.5 mg/dL.

- Kcm (CK-MB, Biomarker Jantung)
  - Sebagian besar pasien (50%) memiliki CK-MB antara 1.655–5.805, yang berada di kisaran normal atau sedikit meningkat.

- Troponin (Biomarker Serangan Jantung)
  - Sebagian besar pasien (50%) memiliki troponin sangat rendah (0.006–0.0855), yang normal (< 0.04 ng/mL). Nilai tinggi kemungkinan terkait dengan kasus positif serangan jantung.
  - Variabilitas besar (std = 1.15) menunjukkan perbedaan signifikan antara pasien dengan dan tanpa serangan jantung.

karena model hanya dapat memproses data numerik. Kolom 'class' dengan nilai kategorikal ('negative' dan 'positive') tidak dapat digunakan langsung oleh model, sehingga harus diubah menjadi numerik (0 untuk 'negative', 1 untuk 'positive') agar model dapat memahami dan mempelajari pola dari data

Korelasi antar fitur
![image](https://github.com/user-attachments/assets/dfafc7ef-a8eb-4bf8-b618-1e01641bc429)
    
- Pressurehight dan Pressurelow -> Korelasi = 0.59 (sedang, positif) menunjukkan tekanan darah sistolik dan diastolik memiliki korelasi positif sedang, wajar karena keduanya mengukur tekanan darah secara keseluruhan.
- troponin (0.23), age (0.24), dan kcm (0.22) adalah prediktor utama serangan jantung berdasarkan korelasi dengan class. Fitur lain seperti impulse dan glucose memiliki pengaruh minimal.
- Korelasi rendah secara keseluruhan menunjukkan pentingnya model seperti XGBoost untuk menangkap pola kompleks.


Distribusi kelas

![image](https://github.com/user-attachments/assets/4ea0cf3f-c233-4244-bc28-609e4978b1f7)
    
- Kelas 0 (negative) = 509 sampel, Kelas 1 (positive) = 810 sampel dari total 1319 sampel.
- Dataset menunjukkan ketidakseimbangan kelas, dengan kelas 1 (positive, menunjukkan serangan jantung) lebih dominan dibandingkan kelas 0 (negative).
- ketidakseimbangan dapat menyebabkan model bias terhadap kelas mayoritas (positive). Maka akan digunakan teknik seperti SMOTE (Synthetic Minority Oversampling Technique) untuk menyeimbangkan data latih.

## Data Preparation
Melakukan standarisasi numerik untuk memastikan semua fitur memiliki skala yang seragam dengan mean 0 dan standar deviasi 1, sehingga algoritma machine learning seperti Logistic Regression atau XGBoost dapat bekerja secara optimal tanpa dipengaruhi oleh perbedaan skala asli fitur seperti age, impulse, pressurehight, pressurelow, glucose, kcm, dan troponin.

Pemisahan fitur dan target juga pembagian data train dan data test menjadi 80% train dan 20% test    
`X = df.drop('class', axis=1)`
`y = df['class']`

`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`

- data.drop('class', axis=1) menghapus kolom class dari DataFrame data, yang merupakan target atau label yang akan diprediksi (0 = negative, 1 = positive).
- Hasilnya, X berisi semua fitur numerik (age, impulse, pressurehight, pressurelow, glucose, kcm, troponin) yang akan digunakan untuk melatih model.
- train_test_split dari pustaka sklearn.model_selection membagi dataset menjadi dua bagian, yaitu data latih (training set) dan data uji (test set).
 
**SMOTE** diterapkan pada data latih untuk menyeimbangkan jumlah kelas, dimana SMOTE adalah teknik oversampling yang diimpor dari pustaka imblearn (imbalanced-learn), digunakan untuk menyeimbangkan dataset dengan ketidakseimbangan kelas dengan membuat sampel sintetis untuk kelas minoritas.

`smote = SMOTE(random_state=42)`
`X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)`

- SMOTE(random_state=42) -> Membuat objek SMOTE.
- random_state=42 digunakan agar proses oversampling bisa direproduksi dan hasilnya sama setiap kali dijalankan.
- smote.fit_resample(X_train, y_train) -> fit_resample() adalah metode utama SMOTE yang akan menganalisis distribusi kelas di y_train.
- Menambahkan sampel sintetis ke kelas minoritas hingga distribusinya seimbang dengan kelas mayoritas.
- Proses tersebut dilakukan hanya pada data pelatihan (x_train), bukan data uji, agar model tidak belajar dari data sintetis yang tidak realistis pada saat evaluasi.

Cara Kerja SMOTE :
- Untuk setiap sampel di kelas minoritas, SMOTE memilih beberapa tetangga terdekat kemudian mengambil titik acak di antara titik data asli dan tetangga terdekat untuk membuat data baru (sintetik).
- Proses tersebut dilakukan berulang kali sampai jumlah data minoritas setara dengan mayoritas.

Teknik SMOTE berhasil melakukan oversampling pada kelas minoritas sehingga jumlah data di setiap kelas seimbang.
Jumlah total data setelah SMOTE : 1294
|  Kelas 0 (negative) | kelas 1 (positive) |
|---------------------|--------------------|
|         647         |         647        |

### 6. Modeling
**Logistic Regression** dengan parameter :    
`lr_model = LogisticRegression(C=1.0, random_state=42)`
`lr_model.fit(X_train_balanced, y_train_balanced)`

- LogisticRegression(...)  -> Membuat objek model regresi logistik untuk klasifikasi biner (atau multiclass).
- C=1.0  -> Parameter regularisasi invers. Ini mengontrol seberapa banyak regularisasi yang diterapkan ke model :   
Semakin besar C, semakin lemah regularisasi → model bisa overfit.   
Semakin kecil C, semakin kuat regularisasi → model lebih sederhana dan bisa underfit.   
C=1.0 adalah nilai default, artinya regularisasi dalam tingkat moderat.
- random_state=42  -> Nilai acak untuk memastikan hasil yang reproducible atau hasil tidak berubah-ubah tiap dijalankan.
- fit(X_train_balanced, y_train_balanced)  -> Melatih model dengan data training yang sudah di-balance dengan SMOTE
- Kelebihan : Sederhana, interpretable, cocok untuk baseline model.
- Kekurangan : Tidak menangani hubungan non-linear dengan baik.

Cara kerja :    
- Model menghitung linear combination yang diproses dengan fungsi sigmoid/logistic untuk menghasilkan probabilitas.
Probabilitas ini dibandingkan dengan threshold (biasanya 0.5) untuk menentukan kelas.

  
**XGBoost Classifier** dengan parameter :     
`xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)`
`xgb_model.fit(X_train_balanced, y_train_balanced)`

- XGBClassifier(...) -> 	Memanggil constructor dari XGBoost Classifier (untuk klasifikasi).
- n_estimators=100  ->	Jumlah decision tree (pohon keputusan) yang akan dibangun secara bertahap dalam proses boosting.    
Nilai umum : 100–1000.
- learning_rate=0.1	 -> Ukuran langkah pembelajaran tiap pohon baru. Semakin kecil nilainya, maka model belajar lebih perlahan, lebih stabil dan bisa mencegah overfitting.   
Default-nya adalah 0.3.
- max_depth=6  ->	Kedalaman maksimal dari tiap pohon. Semakin besar, semakin kompleks pohonnya, artinya bisa overfit.    
- random_state=42	-> Nilai acak untuk memastikan hasil yang reproducible atau hasil tidak berubah-ubah tiap dijalankan.
- fit(X_train_balanced, y_train_balanced)  -> Melatih model dengan data training yang sudah di-balance dengan SMOTE
- Kelebihan : Performa tinggi, menangani ketidakseimbangan kelas, dan memberikan feature importance.
- Kekurangan : Rentan overfitting dan kurang cocok untuk data yang sangat imbalanced

Cara kerja :
- Model bekerja dengan membangun serangkaian pohon keputusan (decision trees) secara berurutan.
- Setiap pohon baru dibuat untuk memperbaiki kesalahan (error) dari pohon sebelumnya.
- Pada setiap iterasi, model meminimalkan loss function (seperti log-loss untuk klasifikasi) menggunakan teknik gradien.
- Hasil akhir adalah gabungan dari semua pohon

## Evaluation

#### Logistic Regression
Evaluasi Logistic Regression :
              precision    recall  f1-score   support

           0       0.67      0.86      0.76       101
           1       0.90      0.74      0.81       163

    accuracy                           0.79       264
   macro avg       0.79      0.80      0.78       264
weighted avg       0.81      0.79      0.79       264

AUC-ROC (Logistic Regression) : 0.8872927170017615

Hasil evaluasi :   
- Tidak terlalu bias ke salah satu kelas.
- AUC-ROC tinggi menunjukkan kemampuan klasifikasinya sudah baik.
- F1-score dan precision/recall seimbang

![image](https://github.com/user-attachments/assets/11a531df-71aa-41e4-8438-444c36ce4f65)
   
- True Negative (TN) = 87. Model benar memprediksi 87 sampel kelas 0 sebagai 0.
- False Positive (FP) = 14. Model salah memprediksi 14 sampel kelas 0 sebagai 1.
- False Negative (FN) = 42. Model salah memprediksi 42 sampel kelas 1 sebagai 0.
- True Positive (TP) = 121. Model benar memprediksi 121 sampel kelas 0 sebagai 1.
- Model cenderung lebih sensitif terhadap kelas 0 (negatif) dibanding kelas 1 (positif).
- 42 data kelas 1 salah diklasifikasi sebagai 0 menunjukkan kelemahan model dalam mengenali kelas positif.


#### XGBoost
Evaluasi XGBoost :
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       101
           1       0.99      0.99      0.99       163

    accuracy                           0.98       264
   macro avg       0.98      0.98      0.98       264
weighted avg       0.98      0.98      0.98       264

AUC-ROC (XGBoost) : 0.9950191338152221
   
- dibandingkan dengan hasil evaluasi model logistic regression, XGBoost unggul jauh di semua metrik dibanding Logistic Regression.
- AUC-ROC sangat tinggi menunjukkan kemampuan klasifikasinya sangat baik.

Confusion matrix 
![image](https://github.com/user-attachments/assets/c20fa139-92d2-4c29-8137-1e29c1680600)
  
- True Negative (TN) = 99. Model benar memprediksi 99 sampel kelas 0 sebagai 0.
- False Positive (FP) = 2. Model salah memprediksi 2 sampel kelas 0 sebagai 1.
- False Negative (FN) = 2. Model salah memprediksi 2 sampel kelas 1 sebagai 0.
- True Positive (TP) = 161. Model benar memprediksi 161 sampel kelas 1 sebagai 1.   

### Kesimpulan :
Model XGBoost memberikan performa yang lebih unggul dibandingkan Logistic Regression di semua metrik evaluasi, seperti accuracy, precision, recall, f1-score, dan AUC-ROC.
AUC-ROC XGBoost sebesar 0.995 menunjukkan bahwa model memiliki kemampuan sangat tinggi dalam membedakan antara kelas positif dan negatif.
Confusion matrix menunjukkan tingkat kesalahan yang sangat kecil (hanya 4 kesalahan dari 264 data), dengan hanya 2 data kelas 0 dan 2 data kelas 1 yang salah klasifikasi.

Sementara itu, model Logistic Regression masih menunjukkan performa yang cukup baik, dengan akurasi 79% dan AUC-ROC 0.887, serta hasil metrik yang cukup seimbang antara precision dan recall.
Namun, Logistic Regression masih mengalami kesulitan dalam mengklasifikasi kelas positif, terlihat dari 42 data kelas 1 yang salah diklasifikasikan sebagai kelas 0.




