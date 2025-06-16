# Laporan Project Predictive Analytics : Heart Disease Classification

## Domain Proyek
Penyakit kardiovaskular (CVD) adalah penyebab kematian nomor satu di dunia, menyumbang sekitar 17,9 juta kematian setiap tahun menurut World Health Organization (WHO). Serangan jantung, salah satu manifestasi utama CVD, sering terjadi secara mendadak dan memerlukan deteksi dini untuk mencegah konsekuensi fatal. Faktor risiko seperti usia, tekanan darah, kadar gula darah, serta biomarker seperti CK-MB dan troponin memainkan peran penting dalam diagnosis. Dengan kemajuan machine learning, data klinis dapat digunakan untuk memprediksi risiko serangan jantung secara akurat, mendukung tenaga medis dalam pengambilan keputusan cepat.

Mengapa Masalah Ini Harus Diselesaikan?
Prediksi dini serangan jantung dapat meningkatkan peluang pasien untuk mendapatkan intervensi medis tepat waktu, mengurangi angka kematian, dan menekan biaya perawatan. Model prediktif berbasis machine learning memungkinkan identifikasi pasien berisiko tinggi secara efisien, terutama di fasilitas kesehatan dengan sumber daya terbatas. Selain itu, model ini dapat memberikan wawasan tentang faktor risiko utama, membantu pencegahan dan edukasi kesehatan.

Referensi :   
[1] World Health Organization, "Cardiovascular diseases (CVDs)," WHO, 2021. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds).      
[2] N. L. K. A. Arsani, N. P. D. S. Wahyuni, N. N. M. Agustin, and M. Budiawan, “Deteksi Dini dan Pencegahan Penyakit Kardiovaskular,” Proceeding Senadimas Undiksha, vol. 1, no. 1, pp. 663–668, 2022.


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
Dataset yang digunakan adalah Heart Disease Classification Dataset dengan 1.319 sampel dan 9 kolom. Dataset ini mencakup faktor risiko dan biomarker yang berkontribusi pada serangan jantung.   

| Jumlah baris     |  Jumlah kolom    |    
|------------------|------------------|        
|      1.319       |       9          |       

Title : Heart Disease Classification Dataset   
Source : Kaggle (https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset/data)  
Owner : Bharath_011   
License : Open Database  
Visibility : Publik   
Usability : 9.41

### Variabel

| Kolom            | Deskripsi                                           |
|------------------|-----------------------------------------------------|
| age              | Usia pasien                                         |
| gender           | Jenis kelamin                                       |
| impulse          | Denyut jantung                                      |
| pressurehight    | Tekanan darah atas                                  |
| pressurelow      | Tekanan darah bawah                                 |
| glucose          | Kadar gula darah                                    |
| kcm              | Kalium, kadar CK-MB, biomarker kerusakan jantung    |
| troponin         | Kadar troponin, biomarker spesifik serangan jantung |
| class            | Keberadaan serangan jantung                         |


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

|  tipe data  |  jumlah kolom  |
|-------------|----------------|
|    integer  |        5       |
|    float    |        3       |
|    object   |        1       |

|  Data duplikat  |
|-----------------|
|        0        |

Pada tabel diatas, menunjukkan bahwa tidak terdapat missing value dan data duplikat

Infomasi statistik masing-masing kolom
| Statistik | age    | gender | impulse | pressurehight | pressurelow | glucose | kcm    | troponin |
| --------- | ------ | ------ | ------- | ------------- | ----------- | ------- | ------ | -------- |
| count     | 1319.000000   | 1319.000000   | 1319.000000    | 1319.000000          | 1319.000000        | 1319.000000    | 1319.000000   | 1319.000000     |
| mean      | 56.191812  | 0.659592   | 78.34336619   | 127.170584        | 72.269143       | 146.634344  | 15.274306  | 0.360942     |
| std       | 13.647315  | 0.474027   | 51.630270   | 26.122720         | 14.033924       | 74.923045   | 46.327083  | 1.154568     |
| min       | 14.000000  | 0.000000   | 20.000000   | 42.000000         | 38.000000       | 35.000000   | 0.321000   | 0.001000    |
| 25%       | 47.000000  | 0.000000   | 64.000000   | 110.000000        | 62.000000       | 98.000000   | 1.655000   | 0.006000    |
| 50%       | 58.000000  | 1.000000   | 74.000000   | 124.000000        | 72.000000       | 116.000000  | 2.850000   | 0.014000    |
| 75%       | 65.000000  | 1.000000   | 85.000000   | 143.000000        | 81.000000       | 169.500000  | 5.805000   | 0.085500    |
| max       | 103.000000 | 1.000000   | 1111.000000 | 223.000000        | 154.000000      | 541.000000  | 300.000000 | 10.300000    |

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
Mengubah nama kolom 'impluse' diubah menjadi 'impulse' agar memudahkan dan tidak membingungkan dan mengubah kolom 'class' menjadi numerik (negative=0, positive=1), melakukan encoding kembali sebagai langkah permanen untuk mempersiapkan data untuk pelatihan model.    

Melakukan standarisasi numerik untuk memastikan semua fitur memiliki skala yang seragam dengan mean 0 dan standar deviasi 1, sehingga algoritma machine learning seperti Logistic Regression atau XGBoost dapat bekerja secara optimal tanpa dipengaruhi oleh perbedaan skala asli fitur seperti age, impulse, pressurehight, pressurelow, glucose, kcm, dan troponin.   
`scaler = StandardScaler()`      
`numeric_features = ['age', 'impulse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin']`      
`df[numeric_features] = scaler.fit_transform(df[numeric_features])`    

- `scaler = StandardScaler()`-> Membuat objek StandardScaler dari library scikit-learn. StandardScaler adalah teknik standarisasi yang mengubah data sehingga memiliki mean = 0 dan standar deviasi = 1 (distribusi standar normal).
- `numeric_features = ['age', 'impulse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin']` ->  mendefinisikan daftar kolom numerik dalam df yang akan diskalakan. Kolom ini dipilih karena berisi data numerik.
- Kolom target `class` tidak diskalakan karena digunakan sebagai label untuk klasifikasi.
- `scaler.fit_transform:
fit`  -> Menghitung mean dan standar deviasi untuk setiap kolom dalam numeric_features.

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

- `SMOTE(random_state=42)` -> Membuat objek SMOTE.
- `random_state=42` digunakan agar proses oversampling bisa direproduksi dan hasilnya sama setiap kali dijalankan.
- `smote.fit_resample(X_train, y_train)` -> `fit_resample()` adalah metode utama SMOTE yang akan menganalisis distribusi kelas di `y_train`.
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

## Modeling
**Logistic Regression** dengan parameter :    
`lr_model = LogisticRegression(C=1.0, random_state=42)`   
`lr_model.fit(X_train_balanced, y_train_balanced)`

dengan parameter :
- `LogisticRegression(...)`  -> Membuat objek model regresi logistik untuk klasifikasi biner (atau multiclass).
- `C=1.0`  -> Parameter regularisasi invers. Ini mengontrol seberapa banyak regularisasi yang diterapkan ke model :   
Semakin besar C, semakin lemah regularisasi → model bisa overfit.   
Semakin kecil C, semakin kuat regularisasi → model lebih sederhana dan bisa underfit.   
`C=1.0` adalah nilai default, artinya regularisasi dalam tingkat moderat.
- `random_state=42`  -> Nilai acak untuk memastikan hasil yang reproducible atau hasil tidak berubah-ubah tiap dijalankan.
- `fit(X_train_balanced, y_train_balanced)`  -> Melatih model dengan data training yang sudah di-balance dengan SMOTE
- Kelebihan : Sederhana, interpretable, cocok untuk baseline model.
- Kekurangan : Tidak menangani hubungan non-linear dengan baik.

Cara kerja :    
- Model menghitung linear combination yang diproses dengan fungsi sigmoid/logistic untuk menghasilkan probabilitas.
Probabilitas ini dibandingkan dengan threshold (biasanya 0.5) untuk menentukan kelas.

  
**XGBoost Classifier** dengan parameter :     
`xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)`   
`xgb_model.fit(X_train_balanced, y_train_balanced)`

dengan parameter :   
- `XGBClassifier(...)` -> 	Memanggil constructor dari XGBoost Classifier (untuk klasifikasi).
- `n_estimators=100`  ->	Jumlah decision tree (pohon keputusan) yang akan dibangun secara bertahap dalam proses boosting.    
Nilai umum : 100–1000.
- `learning_rate=0.1`	 -> Ukuran langkah pembelajaran tiap pohon baru. Semakin kecil nilainya, maka model belajar lebih perlahan, lebih stabil dan bisa mencegah overfitting.   
Default-nya adalah 0.3.
- `max_depth=6`  ->	Kedalaman maksimal dari tiap pohon. Semakin besar, semakin kompleks pohonnya, artinya bisa overfit.    
- `random_state=42`	-> Nilai acak untuk memastikan hasil yang reproducible atau hasil tidak berubah-ubah tiap dijalankan.
- `fit(X_train_balanced, y_train_balanced)`  -> Melatih model dengan data training yang sudah di-balance dengan SMOTE
- Kelebihan : Performa tinggi, menangani ketidakseimbangan kelas, dan memberikan feature importance.
- Kekurangan : Rentan overfitting dan kurang cocok untuk data yang sangat imbalanced


Cara kerja :
- Model bekerja dengan membangun serangkaian pohon keputusan (decision trees) secara berurutan.
- Setiap pohon baru dibuat untuk memperbaiki kesalahan (error) dari pohon sebelumnya.
- Pada setiap iterasi, model meminimalkan loss function (seperti log-loss untuk klasifikasi) menggunakan teknik gradien.
- Hasil akhir adalah gabungan dari semua pohon.


## Evaluation

#### Logistic Regression
Evaluasi Logistic Regression :
              precision    recall  f1-score   support

           0       0.67      0.86      0.76       101
           1       0.90      0.74      0.81       163

    accuracy                           0.79       264
  `macro avg       0.79      0.80      0.78       264`    
  `weighted avg    0.81      0.79      0.79       264`

`AUC-ROC (Logistic Regression) : 0.8872927170017615`

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
  `macro avg       0.98      0.98      0.98       264`    
  `weighted avg    0.98      0.98      0.98       264`

`AUC-ROC (XGBoost) : 0.9950191338152221`
   
- dibandingkan dengan hasil evaluasi model logistic regression, XGBoost unggul jauh di semua metrik dibanding Logistic Regression.
- AUC-ROC sangat tinggi menunjukkan kemampuan klasifikasinya sangat baik.
   
![image](https://github.com/user-attachments/assets/c20fa139-92d2-4c29-8137-1e29c1680600)
  
- True Negative (TN) = 99. Model benar memprediksi 99 sampel kelas 0 sebagai 0.
- False Positive (FP) = 2. Model salah memprediksi 2 sampel kelas 0 sebagai 1.
- False Negative (FN) = 2. Model salah memprediksi 2 sampel kelas 1 sebagai 0.
- True Positive (TP) = 161. Model benar memprediksi 161 sampel kelas 1 sebagai 1.   

### Kesimpulan :
Model XGBoost memberikan performa yang lebih unggul untuk dapat memprediksi keberadaan serangan jantung dibandingkan Logistic Regression di semua metrik evaluasi, seperti accuracy, precision, recall, f1-score, dan AUC-ROC.
AUC-ROC XGBoost sebesar 0.995 menunjukkan bahwa model memiliki kemampuan sangat tinggi dalam membedakan antara kelas positif dan negatif.
Confusion matrix menunjukkan tingkat kesalahan yang sangat kecil (hanya 4 kesalahan dari 264 data), dengan hanya 2 data kelas 0 dan 2 data kelas 1 yang salah klasifikasi.

Sementara itu, model Logistic Regression masih menunjukkan performa yang cukup baik, dengan akurasi 79% dan AUC-ROC 0.887, serta hasil metrik yang cukup seimbang antara precision dan recall.
Namun, Logistic Regression masih mengalami kesulitan dalam mengklasifikasi kelas positif, terlihat dari 42 data kelas 1 yang salah diklasifikasikan sebagai kelas 0.






-------------


