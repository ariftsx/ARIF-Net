# Kamus Data (Data Dictionary)

Dokumen ini menjelaskan variabel dan fitur yang digunakan dalam dataset ARIF-Net.

---

## Target Variable

| Variabel | Tipe | Deskripsi |
|----------|------|-----------|
| `r` | float | Return harga: `(P(t+h) - P(t)) / P(t)` |
| `comcat_id` | string | ID komoditas (`com_14`, `com_11`, `com_3`) |
| `h` | int | Horizon prediksi dalam hari (1-365) |
| `origin_date` | date | Tanggal observasi asal |
| `price_origin` | float | Harga pada tanggal origin (Rp) |
| `price_label` | float | Harga pada tanggal label (Rp) |

## Komoditas

| ID | Nama | Keterangan |
|-----|------|------------|
| `com_14` | Cabai Merah Keriting | Volatilitas tinggi, sensitif cuaca |
| `com_11` | Bawang Merah Ukuran Sedang | Volatilitas tinggi, musiman |
| `com_3` | Beras Kualitas Medium I | Relatif stabil, diatur pemerintah |

## Fitur Harga (Group: PRICE)

| Fitur | Deskripsi |
|-------|-----------|
| `price_lag_{n}` | Harga n hari lalu (n = 1, 2, 3, 5, 7, 14, 21, 30) |
| `return_{n}d` | Return harga n-hari (n = 1, 3, 7, 14, 30) |
| `price_ma_{n}` | Moving average n-hari (n = 5, 10, 20, 30, 60, 90) |
| `price_std_{n}` | Standard deviation harga n-hari |
| `price_ratio_ma_{n}` | Rasio harga / MA_n |
| `volatility_{n}d` | Volatilitas return n-hari (n = 5, 20) |
| `momentum_7d` | Perubahan harga 7 hari |
| `momentum_30d` | Perubahan harga 30 hari |

## Fitur Iklim (Group: CLIMATE)

Rata-rata Tier-1 (9 kabupaten pemasok utama):

| Fitur | Unit | Deskripsi |
|-------|------|-----------|
| `clim_temperature_2m_mean` | C | Suhu rata-rata harian |
| `clim_temperature_2m_max` | C | Suhu maksimum harian |
| `clim_temperature_2m_min` | C | Suhu minimum harian |
| `clim_precipitation_sum` | mm | Total curah hujan harian |
| `clim_relative_humidity_2m_mean` | % | Kelembapan relatif rata-rata |
| `clim_wind_speed_10m_max` | km/h | Kecepatan angin maksimum |
| `clim_et0_fao_evapotranspiration` | mm | Evapotranspirasi referensi FAO |
| `clim_sunshine_duration` | s | Durasi sinar matahari |

## Fitur Berita (Group: NEWS)

| Fitur | Deskripsi |
|-------|-----------|
| `news_count` | Jumlah berita relevan per hari |
| `news_count_with_body` | Jumlah berita dengan isi ter-scrape |
| `sentiment_mean` | Skor sentimen rata-rata (-1 s/d 1) |
| `sentiment_min` | Skor sentimen minimum |
| `sentiment_max` | Skor sentimen maksimum |
| `sentiment_std` | Standard deviation sentimen |
| `body_avg_length` | Rata-rata panjang isi berita (karakter) |
| `news_topic_{topic}` | Jumlah berita per topik per hari |

### Topik Berita

| Topik | Kata Kunci Contoh |
|-------|-------------------|
| `komoditas_cabai` | cabai merah keriting, harga cabai |
| `komoditas_bawang_merah` | bawang merah, harga bawang |
| `komoditas_beras` | harga beras, stok beras, impor beras |
| `pasar_pasokan` | pasar kramat jati, pasokan, panen |
| `harga_inflasi_kebijakan` | inflasi pangan, operasi pasar, het beras |
| `cuaca_iklim` | banjir, kemarau, el nino, kekeringan |
| `bbm_logistik` | harga bbm, ongkos angkut, distribusi |

## Fitur BBM (Group: BBM)

| Fitur | Unit | Deskripsi |
|-------|------|-----------|
| `bbm_pertalite_rp` | Rp/liter | Harga Pertalite |
| `bbm_solar_rp` | Rp/liter | Harga Solar |
| `bbm_pertamax_rp` | Rp/liter | Harga Pertamax |

## Fitur Kalender (Group: CALENDAR)

| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| `cal_day_of_week` | int (0-6) | Hari dalam minggu (0=Senin) |
| `cal_month` | int (1-12) | Bulan |
| `cal_quarter` | int (1-4) | Kuartal |
| `cal_is_weekend` | binary | Akhir pekan (Sabtu/Minggu) |
| `cal_is_holiday` | binary | Hari libur nasional |
| `cal_is_ramadan` | binary | Bulan Ramadan |
| `cal_is_idul_fitri` | binary | Periode Idul Fitri (+/- 3 hari) |
| `cal_is_idul_adha` | binary | Periode Idul Adha (+/- 1 hari) |
| `cal_is_natal_tahun_baru` | binary | 24 Des - 2 Jan |
| `cal_is_kemerdekaan` | binary | 17 Agustus |

## Fitur Horizon (Group: HORIZON)

| Fitur | Deskripsi |
|-------|-----------|
| `h_log` | log(h) -- horizon dalam skala logaritmik |
| `h_is_{n}` | One-hot encoding per horizon (n = 1,3,7,...,365) |
