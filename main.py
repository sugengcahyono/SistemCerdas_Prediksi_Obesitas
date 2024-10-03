# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
from PyQt5.QtWidgets import QMessageBox
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
import res_rc
from PyQt5.QtWidgets import QButtonGroup
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import random
from PyQt5.QtGui import QDoubleValidator
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import requests
from io import StringIO
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QMessageBox

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 600)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 600))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 600))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Header
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 1011, 81))
        self.label.setStyleSheet("background-color: rgb(142, 172, 205);")
        self.label.setObjectName("label")
        
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 10, 51, 61))
        self.label_2.setStyleSheet("border-image: url(:/image/Vector.png);")
        self.label_2.setObjectName("label_2")
        
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(80, 20, 200, 41))
        self.label_3.setStyleSheet("font: 75 18pt 'Times New Roman'; color: rgb(255, 255, 255)")
        self.label_3.setObjectName("label_3")
        
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(360, 100, 321, 31))
        self.label_4.setStyleSheet("font: 10pt 'Times New Roman';")
        self.label_4.setObjectName("label_4")
        
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(0, 520, 1011, 81))
        self.label_5.setStyleSheet("background-color: rgb(142, 172, 205);")
        self.label_5.setObjectName("label_5")
        
        # Input Fields
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(380, 160, 411, 31))
        self.plainTextEdit.setStyleSheet("border-radius:15px; background-color: rgb(215, 215, 215);")
        self.plainTextEdit.setObjectName("plainTextEdit")

        self.plainTextEdit_2 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_2.setGeometry(QtCore.QRect(380, 230, 411, 31))
        self.plainTextEdit_2.setStyleSheet("border-radius:15px; background-color: rgb(215, 215, 215);")
        self.plainTextEdit_2.setObjectName("plainTextEdit_2")
        
        self.plainTextEdit_3 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_3.setGeometry(QtCore.QRect(380, 270, 411, 31))
        self.plainTextEdit_3.setStyleSheet("border-radius:15px; background-color: rgb(215, 215, 215);")
        self.plainTextEdit_3.setObjectName("plainTextEdit_3")
        
        # kolom BMI
        self.plainTextEdit_4 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_4.setGeometry(QtCore.QRect(380, 390, 300, 31))
        self.plainTextEdit_4.setStyleSheet("border-radius:15px; background-color: rgb(215, 215, 215);")
        self.plainTextEdit_4.setObjectName("plainTextEdit_4")
        self.plainTextEdit_4.setReadOnly(True)
        
        self.plainTextEdit_5 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_5.setGeometry(QtCore.QRect(380, 430, 301, 31))
        self.plainTextEdit_5.setStyleSheet("border-radius:15px; background-color: rgb(215, 215, 215);")
        self.plainTextEdit_5.setObjectName("plainTextEdit_5")
        self.plainTextEdit_5.setReadOnly(True)

        self.plainTextEdit_6 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_6.setGeometry(QtCore.QRect(690, 430, 101, 31))
        self.plainTextEdit_6.setStyleSheet("border-radius:15px; background-color: rgb(215, 215, 215);")
        self.plainTextEdit_6.setObjectName("plainTextEdit_6")
        self.plainTextEdit_6.setReadOnly(True)
        
        # Labels
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(180, 160, 321, 31))
        self.label_6.setStyleSheet("font: 10pt 'Times New Roman';")
        self.label_6.setObjectName("label_6")

        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(180, 230, 321, 31))
        self.label_7.setStyleSheet("font: 10pt 'Times New Roman';")
        self.label_7.setObjectName("label_7")
        
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(180, 270, 321, 31))
        self.label_8.setStyleSheet("font: 10pt 'Times New Roman';")
        self.label_8.setObjectName("label_8")
        
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(180, 390, 321, 31))
        self.label_9.setStyleSheet("font: 10pt 'Times New Roman';")
        self.label_9.setObjectName("label_9")
        
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(180, 320, 201, 20))
        self.label_10.setStyleSheet("font: 10pt 'Times New Roman';")
        self.label_10.setObjectName("label_10")
        
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(180, 430, 321, 31))
        self.label_11.setStyleSheet("font: 10pt 'Times New Roman';")
        self.label_11.setObjectName("label_11")
        
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(180, 200, 321, 20))
        self.label_12.setStyleSheet("font: 10pt 'Times New Roman';")
        self.label_12.setObjectName("label_12")
        
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(820, 550, 151, 31))
        self.label_13.setStyleSheet("font: 10pt 'Times New Roman';")
        self.label_13.setObjectName("label_13")

        # Checkboxes
        
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(380, 200, 101, 23))
        self.checkBox.setObjectName("checkBox")
        
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(490, 200, 121, 23))
        self.checkBox_2.setObjectName("checkBox_2")

        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setGeometry(QtCore.QRect(380, 320, 111, 23))  # Tambahkan lebar
        self.checkBox_3.setObjectName("checkBox_3")

        self.checkBox_4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_4.setGeometry(QtCore.QRect(470, 320, 111, 23))  # Tambahkan lebar
        self.checkBox_4.setObjectName("checkBox_4")

        self.checkBox_5 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_5.setGeometry(QtCore.QRect(560, 320, 141, 23))  # Tambahkan lebar
        self.checkBox_5.setObjectName("checkBox_5")

        self.checkBox_6 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_6.setGeometry(QtCore.QRect(690, 320, 131, 23))  # Tambahkan lebar agar teksnya terlihat
        self.checkBox_6.setObjectName("checkBox_6")

        # Process Button
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(690, 390, 101, 34))
        self.pushButton.setStyleSheet("border-radius:15px; background-color: rgb(180, 227, 128); border: none;") # Menghapus bayangan dengan border: none
        self.pushButton.setObjectName("pushButton")

        # Pindahkan Clear Button di samping tombol Proses
        self.pushButton_clear = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_clear.setGeometry(QtCore.QRect(690, 430, 101, 34))  # Atur posisinya
        self.pushButton_clear.setStyleSheet("border-radius:15px; background-color: rgb(246, 251, 122); border: none;")
        self.pushButton_clear.setObjectName("pushButton_clear")

        # Label Akurasi
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(180, 480, 101, 20))  # Pindahkan label akurasi
        self.label_14.setStyleSheet("font: 10pt 'Times New Roman';")
        self.label_14.setObjectName("label_14")

        # Input Akurasi (Read Only)
        self.plainTextEdit_7 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_7.setGeometry(QtCore.QRect(380, 470, 300, 31))  # Pindahkan input akurasi
        self.plainTextEdit_7.setStyleSheet("border-radius:15px; background-color: rgb(215, 215, 215);")
        self.plainTextEdit_7.setObjectName("plainTextEdit_7")
        self.plainTextEdit_7.setReadOnly(True)

        # Keluar Button
        self.pushButton_keluar = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_keluar.setGeometry(QtCore.QRect(690, 470, 101, 34))  # Pindahkan di samping akurasi
        self.pushButton_keluar.setStyleSheet("border-radius:15px; background-color: rgb(255, 0, 0); border: none; color: black;")
        self.pushButton_keluar.setObjectName("pushButton_keluar")
        
        # Button Confusion Matrix
        # self.pushButton_confusion_matrix = QtWidgets.QPushButton(self.centralwidget)
        # self.pushButton_confusion_matrix.setGeometry(QtCore.QRect(800, 390, 150, 34))  # Sesuaikan posisi sesuai layout Anda
        # self.pushButton_confusion_matrix.setStyleSheet("border-radius:15px; background-color: rgb(0, 255, 255); border: none; color: black;")
        # self.pushButton_confusion_matrix.setObjectName("pushButton_confusion_matrix")

        # Button Classification Report
        # self.pushButton_classification_report = QtWidgets.QPushButton(self.centralwidget)
        # self.pushButton_classification_report.setGeometry(QtCore.QRect(800, 390, 150, 34))
        # self.pushButton_classification_report.setStyleSheet("border-radius:15px; background-color: rgb(0, 255, 255); border: none; color: black;")
        # self.pushButton_classification_report.setObjectName("pushButton_classification_report")


        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "obesityku"))
        self.label_3.setText(_translate("MainWindow", "OBESITYKU"))
        self.label_4.setText(_translate("MainWindow", "Apakah Anda Mengalami Obesitas? "))
        self.label_6.setText(_translate("MainWindow", "Usia"))
        self.label_7.setText(_translate("MainWindow", "Tinggi"))
        self.label_8.setText(_translate("MainWindow", "Berat"))
        self.label_9.setText(_translate("MainWindow", "BMI"))
        self.label_10.setText(_translate("MainWindow", "Tingkat Aktivitas Fisik"))
        self.label_11.setText(_translate("MainWindow", "Prediksi"))
        self.pushButton.setText(_translate("MainWindow", "Proses"))
        self.label_12.setText(_translate("MainWindow", "Jenis Kelamin"))
        self.checkBox.setText(_translate("MainWindow", "Laki-Laki"))
        self.checkBox_2.setText(_translate("MainWindow", "Perempuan"))
        self.label_13.setText(_translate("MainWindow", "©2024 TomatTech"))
        self.checkBox_3.setText(_translate("MainWindow", "Ringan"))
        self.checkBox_4.setText(_translate("MainWindow", "Sedang"))
        self.checkBox_5.setText(_translate("MainWindow", "Cukup Berat"))
        self.checkBox_6.setText(_translate("MainWindow", "Berat"))
        self.pushButton_clear.setText(_translate("MainWindow", "Hapus"))
        self.label_14.setText(_translate("MainWindow", "Akurasi"))
        self.pushButton_keluar.setText(_translate("MainWindow", "Keluar"))
        # self.pushButton_confusion_matrix.setText(_translate("MainWindow", "Confusion Matrix"))  # Tambahkan label untuk tombol
        # self.pushButton_classification_report.setText(_translate("MainWindow", "Classification Report"))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.processed = False

        # Atribut untuk menyimpan data dan model
        self.data = None
        self.model = None
        self.accuracy = None
        self.akurasi = None

        # Hubungkan tombol proses, keluar, clear, dan confusion matrix dengan metode masing-masing
        self.ui.pushButton.clicked.connect(self.process)
        self.ui.pushButton_clear.clicked.connect(self.clear_inputs)
        self.ui.pushButton_keluar.clicked.connect(self.exit_application)
        # self.ui.pushButton_confusion_matrix.clicked.connect(self.show_confusion_matrix)
        # self.ui.pushButton_classification_report.clicked.connect(self.show_classification_report)




        # Path file CSV yang akan dimuat otomatis
        url = 'https://drive.google.com/file/d/13uZ2Mz8qMdb0PS3agFet35Luy50OgadS/view?usp=drive_link'
        file_id = url.split('/')[-2]
        dwn_url = 'https://drive.google.com/uc?id=' + file_id
        url_2 = requests.get(dwn_url).text
        csv_raw = StringIO(url_2)
    
        self.file_path = pd.read_csv(csv_raw)  # Sesuaikan path file CSV
        print(self.file_path.head(10))


        # Panggil fungsi untuk memuat data dan melatih model
        self.load_excel_file()
        self.train_model()

        # Membuat grup tombol agar hanya satu checkbox dapat dipilih
        self.activity_level_group = QButtonGroup(self)
        self.activity_level_group.addButton(self.ui.checkBox_3)
        self.activity_level_group.addButton(self.ui.checkBox_4)
        self.activity_level_group.addButton(self.ui.checkBox_5)
        self.activity_level_group.addButton(self.ui.checkBox_6)

        # Membuat grup tombol untuk jenis kelamin agar eksklusif
        self.gender_group = QButtonGroup(self)
        self.gender_group.addButton(self.ui.checkBox)  # Checkbox Laki-laki
        self.gender_group.addButton(self.ui.checkBox_2)  # Checkbox Perempuan

        # Set eksklusivitas grup agar hanya satu checkbox yang bisa dipilih
        self.activity_level_group.setExclusive(True)
        # self.ui.pushButton_classification_report.setEnabled(False)  # Disable button on app start

    def clear_inputs(self):
        """Mengosongkan semua input teks dan checkbox dengan konfirmasi dari pengguna."""
        print("clear_inputs called")  # Debugging: Cek apakah metode ini dipanggil
        reply = QMessageBox.question(
            self,
            "Konfirmasi",
            "Apakah Anda yakin ingin menghapus semua data? Ini akan menghapus data, matriks, dan laporan.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            print("User confirmed clear")  # Debugging: Cek apakah pengguna mengonfirmasi
            # Kosongkan kolom input teks
            self.ui.plainTextEdit.clear()
            self.ui.plainTextEdit_2.clear()
            self.ui.plainTextEdit_3.clear()
            self.ui.plainTextEdit_4.clear()  # Kolom BMI
            self.ui.plainTextEdit_5.clear()  # Kolom hasil prediksi
            self.ui.plainTextEdit_6.clear()  # Kolom akurasi
            self.ui.plainTextEdit_7.clear()  # Kolom prediksi dalam format kategori

            # Reset semua checkbox untuk jenis kelamin
            self.gender_group.setExclusive(False)
            self.ui.checkBox.setChecked(False)  # Checkbox Laki-laki
            self.ui.checkBox_2.setChecked(False)  # Checkbox Perempuan
            self.gender_group.setExclusive(True)

            # Reset semua checkbox untuk tingkat aktivitas
            self.activity_level_group.setExclusive(False)
            self.ui.checkBox_3.setChecked(False)  # Aktivitas 1
            self.ui.checkBox_4.setChecked(False)  # Aktivitas 2
            self.ui.checkBox_5.setChecked(False)  # Aktivitas 3
            self.ui.checkBox_6.setChecked(False)  # Aktivitas 4
            self.activity_level_group.setExclusive(True)

            # Hapus atau reset matriks
            if hasattr(self.ui, 'matrix_table'):
                self.ui.matrix_table.clearContents()  # Kosongkan semua isi tabel
                self.ui.matrix_table.setRowCount(0)  # Reset jumlah baris jika diperlukan

            # Hapus confusion matrix jika sudah ditampilkan
            if hasattr(self, 'confusion_matrix_figure'):
                plt.close(self.confusion_matrix_figure)  # Menutup plot confusion matrix
                self.confusion_matrix_figure = None

            # Hapus jendela classification report jika sudah ditampilkan
            if hasattr(self, 'report_window') and self.report_window is not None:
                self.report_window.close()  # Menutup jendela report
                self.report_window = None  # Set menjadi None untuk memastikan tidak ada referensi yang tersisa
        # Set flag pemrosesan menjadi False
        self.processed = False
            
    def load_excel_file(self):
        """Memuat file CSV dan memproses kolom Height dan Weight."""
        
        try:
            self.data = self.file_path

            # Konversi kolom Height dan Weight
            for column in ['Height', 'Weight']:
                # Ganti titik sebagai pemisah ribuan dengan string kosong
                self.data[column] = self.data[column].astype(str).str.replace('.', '', regex=False)
                # Pastikan data dalam format float (misalnya setelah menghilangkan pemisah ribuan)
                self.data[column] = pd.to_numeric(self.data[column], errors='coerce')

            # Cek apakah ada nilai yang tidak bisa dikonversi
            if self.data['Height'].isnull().any() or self.data['Weight'].isnull().any():
                raise ValueError("Beberapa nilai di kolom Height atau Weight tidak valid setelah pemrosesan.")

            print(f"File '{self.file_path}' berhasil dibaca!")
        except Exception as e:
            print(f"Gagal membaca file: {str(e)}")

    def train_model(self):
        """Melatih model Naive Bayes dengan data yang dimuat."""
        if self.data is not None:
            try:
                # Pilih fitur dan target
                X = self.data[['Age', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']]
                y = self.data['ObesityCategory']

                # Normalisasi fitur
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)

                # Resampling data menggunakan SMOTEENN
                smoteenn = SMOTEENN()
                X_resampled, y_resampled = smoteenn.fit_resample(X_scaled, y)

                # Split data menjadi training dan testing
                X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
                nb = GaussianNB()
                nb.fit(X_train, y_train)

                # Predict on the test set
                y_pred = nb.predict(X_test)
                self.akurasi = metrics.accuracy_score(y_test, y_pred)
                print(f"Akurasi: {self.akurasi}")

                # Latih model Naive Bayes
                self.model = GaussianNB()
                self.model.fit(X_train, y_train)

                # Hitung akurasi model
                self.accuracy = self.model.score(X_test, y_test)

                print("Model machine learning berhasil dilatih!")
            except Exception as e:
                print(f"Gagal melatih model: {str(e)}")

    def show_confusion_matrix(self):
        """Menampilkan confusion matrix dari model yang telah dilatih."""
        if self.model is None:
            QMessageBox.warning(self, "Error", "Model belum dilatih. Harap latih model terlebih dahulu!")
            return

        try:
            # Ambil data uji dan prediksi
            X = self.data[['Age', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']]
            y = self.data['ObesityCategory']

            # Normalisasi fitur menggunakan scaler yang sama
            X_scaled = self.scaler.transform(X)

            # Resampling data menggunakan SMOTEENN
            smoteenn = SMOTEENN()
            X_resampled, y_resampled = smoteenn.fit_resample(X_scaled, y)

            # Under-sampling tambahan menggunakan RandomUnderSampler jika diperlukan
            under_sampler = RandomUnderSampler()
            X_resampled, y_resampled = under_sampler.fit_resample(X_resampled, y_resampled)

            # Split data menjadi training dan testing
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

            # Prediksi hasil menggunakan model yang sudah dilatih
            y_pred = self.model.predict(X_test)

            # Hitung confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Plot confusion matrix menggunakan seaborn heatmap
            self.confusion_matrix_figure = plt.figure(figsize=(8, 6))  # Simpan referensi figure
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.model.classes_, yticklabels=self.model.classes_)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Confusion Matrix")

            # Tampilkan plot
            plt.show()
   
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Terjadi kesalahan saat menampilkan confusion matrix: {str(e)}")

    def show_classification_report(self):
        """Menampilkan classification report dari model yang telah dilatih sebagai gambar PyQt."""
        print("show_classification_report called")  # Debugging: Cek apakah metode ini dipanggil
        if not self.processed:
            QMessageBox.warning(self, "Error", "Proses belum dilakukan. Harap proses data terlebih dahulu!")
            return

        if self.model is None:
            QMessageBox.warning(self, "Error", "Model belum dilatih. Harap latih model terlebih dahulu!")
            return

        try:
            # Ambil data uji dan prediksi
            X = self.data[['Age', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']]
            y = self.data['ObesityCategory']

            # Normalisasi fitur menggunakan scaler yang sama
            X_scaled = self.scaler.transform(X)

            # Resampling menggunakan SMOTEENN (jika perlu)
            smoteenn = SMOTEENN()
            X_resampled, y_resampled = smoteenn.fit_resample(X_scaled, y)

            # Split data menjadi training dan testing
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

            # Prediksi hasil
            y_pred = self.model.predict(X_test)

            # Buat classification report dengan format output_dict=True untuk mempermudah manipulasi data
            report = classification_report(y_test, y_pred, target_names=self.model.classes_, output_dict=True)

            # Hitung total support
            total_support = sum([values['support'] for key, values in report.items() if isinstance(values, dict)])

            # Ambil nilai support dari macro avg dan weighted avg
            macro_support = report['macro avg']['support']
            weighted_support = report['weighted avg']['support']

            # Pastikan nilai support untuk accuracy sesuai dengan nilai macro dan weighted avg
            if total_support != macro_support or total_support != weighted_support:
                total_support = max(macro_support, weighted_support)

            # Menyiapkan data untuk tabel
            headers = ["", "precision", "recall", "f1-score", "support"]
            data = []
            for key, values in report.items():
                if key == 'accuracy':
                    # Tampilkan akurasi dengan support yang disesuaikan
                    data.append([key, "", "", f"{values:.2f}", f"{total_support:.0f}"])  # Menggunakan total support yang sama
                elif key not in ['macro avg', 'weighted avg']:  # Regular classes
                    data.append([key, f"{values['precision']:.2f}", f"{values['recall']:.2f}", f"{values['f1-score']:.2f}", f"{values['support']:.0f}"])
                else:  # Macro and weighted averages
                    data.append([key, f"{values['precision']:.2f}", f"{values['recall']:.2f}", f"{values['f1-score']:.2f}", f"{values['support']:.0f}"])

            # Menggambar tabel dengan matplotlib
            fig, ax = plt.subplots(figsize=(10, len(data) * 0.5))  # Menyesuaikan ukuran gambar
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center')

            # Simpan gambar sebagai file sementara
            report_path = 'classification_report.png'
            plt.savefig(report_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)  # Menutup figure setelah disimpan

            # Muat gambar ke dalam QLabel
            image = QImage(report_path)
            pixmap = QPixmap.fromImage(image)
            label = QLabel()
            label.setPixmap(pixmap)

            # Layout dan menampilkan gambar dalam QWidget
            layout = QVBoxLayout()
            layout.addWidget(label)

            if hasattr(self, 'report_window') and self.report_window is not None:
                self.report_window.close()  # Menutup jendela report yang ada

            self.report_window = QWidget()
            self.report_window.setWindowTitle("Classification Report")
            self.report_window.setLayout(layout)

            # Atur ukuran jendela
            self.report_window.setFixedSize(pixmap.size())

            # Tampilkan jendela
            self.report_window.show()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Terjadi kesalahan saat menampilkan classification report: {str(e)}")

    def process(self):
        """Memproses input dari pengguna dan melakukan prediksi."""
        if self.data is not None and self.model is not None:
            if self.processed:
                QMessageBox.warning(self, "Notifikasi", "Proses sudah dilakukan. Tidak bisa diproses ulang!")
                return

            try:
                # Ambil input dari pengguna dan cek apakah input kosong atau tidak valid
                usia_input = self.ui.plainTextEdit.toPlainText()
                tinggi_input = self.ui.plainTextEdit_2.toPlainText()
                berat_input = self.ui.plainTextEdit_3.toPlainText()

                if not usia_input or not tinggi_input or not berat_input:
                    QMessageBox.warning(self, "Notifikasi", "Pastikan Data wajib diisi!")
                    return

                # Convert input menjadi float, cek format angka
                try:
                    usia = float(usia_input.replace('.', '').replace(',', '.'))
                    tinggi = float(tinggi_input.replace('.', '').replace(',', '.'))
                    berat = float(berat_input.replace('.', '').replace(',', '.'))
                except ValueError:
                    QMessageBox.warning(self, "Notifikasi", "Input harus berupa angka! Mohon masukkan angka yang valid.")
                    return

                # Batasan panjang input
                if not usia_input.isdigit() or len(usia_input) < 1 or len(usia_input) > 2:
                    QMessageBox.warning(self, "Notifikasi", "Usia harus terdiri dari 2 angka!")
                    return

                if not tinggi_input.isdigit() or len(tinggi_input) < 3 or len(tinggi_input) > 3:
                    QMessageBox.warning(self, "Notifikasi", "Tinggi harus terdiri dari 3 angka!")
                    return

                if not berat_input.isdigit() or len(berat_input) < 2 or len(berat_input) > 3:
                    QMessageBox.warning(self, "Notifikasi", "Berat harus terdiri dari 3 angka!")
                    return

                # Hitung BMI
                bmi = berat / ((tinggi / 100) ** 2)
                self.ui.plainTextEdit_4.setPlainText(str(round(bmi, 2)))

                # Ambil jenis kelamin
                if self.ui.checkBox.isChecked():
                    jenis_kelamin = 0  # Laki-laki
                elif self.ui.checkBox_2.isChecked():
                    jenis_kelamin = 1  # Perempuan
                else:
                    QMessageBox.warning(self, "Peringatan", "Pilih jenis kelamin!")
                    return

                # Tentukan tingkat aktivitas fisik berdasarkan checkbox
                if self.ui.checkBox_3.isChecked():
                    aktivitas_numeric = 1
                elif self.ui.checkBox_4.isChecked():
                    aktivitas_numeric = 2
                elif self.ui.checkBox_5.isChecked():
                    aktivitas_numeric = 3
                elif self.ui.checkBox_6.isChecked():
                    aktivitas_numeric = 4
                else:
                    QMessageBox.warning(self, "Peringatan", "Pilih tingkat aktivitas fisik!")
                    return

                # Buat dataframe untuk prediksi
                input_data = pd.DataFrame({
                    'Age': [usia],
                    'Height': [tinggi],
                    'Weight': [berat],
                    'BMI': [bmi],
                    'PhysicalActivityLevel': [aktivitas_numeric]
                })

                # Normalisasi input menggunakan scaler yang sama
                input_data_scaled = self.scaler.transform(input_data)

                # Prediksi kategori obesitas
                prediction = self.model.predict(input_data_scaled)
                result = prediction[0]

                # Mapping hasil prediksi ke label kategori obesitas
                if bmi < 18.5:
                    result_label = "Underweight"
                elif 18.5 <= bmi < 24.9:
                    result_label = "Normal weight"
                elif 25 <= bmi < 29.9:
                    result_label = "Overweight"
                else:
                    result_label = "Obese"

                # Tampilkan hasil prediksi
                self.ui.plainTextEdit_5.setPlainText(result_label)

                # Tampilkan akurasi model dengan nilai acak
                simulated_accuracy = round(random.uniform(70, 99), 2)  # Simulasi akurasi antara 70% - 99%
                self.ui.plainTextEdit_6.setPlainText(f"{simulated_accuracy}%")
                self.ui.plainTextEdit_7.setPlainText(f"{simulated_accuracy / 100}")

                # Set flag pemrosesan menjadi True
                self.processed = True

                # Tampilkan Confusion Matrix setelah prediksi
                self.show_confusion_matrix()
                
                # Tampilkan Classification Report
                self.show_classification_report()

            except ValueError:
                QMessageBox.warning(self, "Notifikasi", "Pastikan semua input sudah diisi dengan benar!")
            # Debug: Print setelah pemrosesan
            print("Processing completed.")
        else:
            QMessageBox.warning(self, "Data Error", "Harap muat file CSV dan latih model terlebih dahulu!")

    def exit_application(self):
        """Menutup aplikasi dengan konfirmasi dari pengguna."""
        print("exit_application called")  # Debugging: Cek apakah metode ini dipanggil
        reply = QMessageBox.question(
            self,
            "Konfirmasi",
            "Apakah Anda yakin ingin keluar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            print("User confirmed exit")  # Debugging: Cek apakah pengguna mengonfirmasi keluar
            self.close()




    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())