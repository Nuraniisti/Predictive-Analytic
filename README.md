# Evaluasi Project Predictive Analytic Sistem Rekomendasi Musik

## Metrik Evaluasi yang Digunakan

### 1. Precision@K  
Precision@K adalah metrik yang mengukur proporsi item relevan di antara K item teratas yang direkomendasikan oleh sistem.

**Fungsi:**  
Mengukur ketepatan rekomendasi, yaitu seberapa banyak rekomendasi yang diberikan benar-benar sesuai dengan preferensi pengguna.

---

### 2. Recall@K  
Recall@K mengukur proporsi item relevan yang berhasil direkomendasikan dalam K item teratas dari total item relevan yang ada.

**Fungsi:**  
Mengukur kelengkapan rekomendasi, yaitu seberapa banyak item yang disukai pengguna berhasil ditemukan oleh sistem.

---

### Alasan Pemilihan Metrik  
- Sistem rekomendasi menghasilkan daftar peringkat (Top-K), sehingga metrik yang mengukur performa pada posisi teratas lebih relevan dibanding metrik klasifikasi biasa seperti accuracy atau F1-score.  

---

## Penjelasan Hasil Proyek Berdasarkan Metrik Evaluasi

Misalnya hasil evaluasi yang diperoleh adalah:  
- Precision@5 (Content-Based) = 0.2  
- Recall@5 (Content-Based) = 1.0  
- Precision@5 (Collaborative Filtering) = 0.2  
- Recall@5 (Collaborative Filtering) = 1.0  

### Interpretasi  
- **Recall@5 = 1.0** menunjukkan bahwa semua item yang relevan berhasil masuk ke dalam rekomendasi Top-5, artinya sistem tidak melewatkan item penting bagi pengguna.  
- **Precision@5 = 0.2** menunjukkan bahwa hanya 20% dari rekomendasi Top-5 yang benar-benar relevan, sisanya kurang tepat atau tidak sesuai preferensi pengguna.  
- Dengan kata lain, sistem sudah sangat baik dalam menangkap preferensi pengguna (Recall tinggi), tetapi masih perlu perbaikan agar rekomendasi lebih fokus dan relevan (Precision rendah).

### Implikasi  
- Sistem rekomendasi sudah efektif dalam menemukan semua item yang disukai pengguna, tetapi rekomendasi perlu dipersempit agar tidak banyak item tidak relevan yang muncul.  

---

## Kesimpulan

- Precision@K dan Recall@K adalah metrik yang tepat dan relevan untuk mengevaluasi sistem rekomendasi musik.  
- Hasil evaluasi menunjukkan model sudah mampu menangkap preferensi pengguna dengan baik (Recall tinggi), namun perlu peningkatan agar rekomendasi lebih tepat sasaran (Precision perlu ditingkatkan).  
