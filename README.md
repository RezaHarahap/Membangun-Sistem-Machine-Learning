# Membangun Sistem Machine Learning

Repository ini berisi proyek submission **Membangun Sistem Machine Learning** yang mencakup proses eksperimen, pembangunan model machine learning, otomatisasi workflow CI, serta monitoring model menggunakan Prometheus dan Grafana.

## Gambaran Proyek

Proyek menggunakan dataset **Breast Cancer** untuk membangun model klasifikasi diagnosis. Alur pengerjaan mencakup preprocessing data, training dan tuning model, pencatatan eksperimen/model, inference, serta monitoring layanan model.

## Struktur Repository

```text
Membangun-Sistem-Machine-Learning/
├── README.md
├── Eksperimen_SML_Reza-Harahap.txt
├── Workflow-CI.txt
│
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── requirements.txt
│   ├── screenshoot_artifak.jpg
│   ├── screenshoot_dashboard.jpg
│   └── breast_cancer_preprocessing/
│       ├── metadata.json
│       ├── train.csv
│       └── test.csv
│
└── Monitoring dan Logging/
    ├── 1.bukti_serving.jpg
    ├── 2.prometheus.yml
    ├── 3.prometheus_exporter.py
    ├── 4.bukti monitoring Prometheus/
    │   ├── 1.monitoring_requests.jpg
    │   ├── 2.monitoring_model_up.jpg
    │   └── 3.monitoring_latency.jpg
    ├── 5.bukti monitoring Grafana/
    │   ├── 1.dashboard_reza_harahap.jpg
    │   ├── 2.monitoring_requests.jpg
    │   └── 3.monitoring_latency.jpg
    ├── 7.Inference.py
    └── grafana_queries.txt
```

## Komponen Utama

### 1. Eksperimen Machine Learning

Eksperimen dilakukan pada data Breast Cancer untuk mempersiapkan dataset sebelum masuk ke tahap pembangunan model. File referensi eksperimen tersedia pada:

- `Eksperimen_SML_Reza-Harahap.txt`

### 2. Preprocessing Dataset

Folder `Membangun_model/breast_cancer_preprocessing/` berisi hasil preprocessing yang digunakan pada tahap modelling:

- `train.csv` — data training.
- `test.csv` — data testing.
- `metadata.json` — metadata preprocessing, daftar fitur, target, dan parameter scaler.

Target klasifikasi yang digunakan adalah **diagnosis**.

### 3. Pembangunan Model

Tahap pembangunan model terdapat pada folder `Membangun_model/`:

- `modelling.py` — proses training model utama.
- `modelling_tuning.py` — proses tuning/hyperparameter tuning model.
- `requirements.txt` — dependency Python yang diperlukan.

Bukti hasil eksperimen/model juga disertakan melalui:

- `screenshoot_artifak.jpg`
- `screenshoot_dashboard.jpg`

### 4. Workflow CI

Referensi repository workflow CI dicantumkan pada:

- `Workflow-CI.txt`

Workflow CI digunakan sebagai bagian dari proses otomatisasi pengembangan dan retraining model.

### 5. Model Serving dan Inference

Folder `Monitoring dan Logging/` berisi bagian deployment lokal, inference, dan monitoring sistem.

File utama:

- `7.Inference.py` — menjalankan inference terhadap model.
- `1.bukti_serving.jpg` — bukti model serving.

### 6. Monitoring dengan Prometheus

Monitoring dilakukan menggunakan **Prometheus**.

File yang digunakan:

- `2.prometheus.yml` — konfigurasi Prometheus.
- `3.prometheus_exporter.py` — exporter metric untuk aplikasi/model.

Bukti monitoring Prometheus terdapat pada folder:

```text
Monitoring dan Logging/4.bukti monitoring Prometheus/
```

Metric yang didokumentasikan antara lain:

- jumlah request,
- status/model up,
- latency.

### 7. Visualisasi Monitoring dengan Grafana

Metric dari sistem divisualisasikan menggunakan **Grafana**.

Bukti dashboard tersedia pada:

```text
Monitoring dan Logging/5.bukti monitoring Grafana/
```

Folder tersebut berisi bukti dashboard serta visualisasi metric request dan latency.

Query Grafana yang digunakan juga tersedia pada:

- `Monitoring dan Logging/grafana_queries.txt`

## Teknologi yang Digunakan

- Python
- Machine Learning
- MLflow
- GitHub Actions
- Prometheus
- Grafana

## Tujuan Proyek

Proyek ini bertujuan menunjukkan implementasi alur machine learning yang lebih mendekati praktik MLOps, mulai dari pengolahan data dan pembangunan model hingga CI, serving, inference, serta monitoring performa layanan model.

## Author

**Muhammad Reza Pahlevi Harahap**  
GitHub: [RezaHarahap](https://github.com/RezaHarahap)
