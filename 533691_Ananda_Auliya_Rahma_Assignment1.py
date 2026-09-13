import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_CSV = """
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5.0,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
4.8,3.4,1.6,0.2,Iris-setosa
4.8,3.0,1.4,0.1,Iris-setosa
4.3,3.0,1.1,0.1,Iris-setosa
5.8,4.0,1.2,0.2,Iris-setosa
5.7,4.4,1.5,0.4,Iris-setosa
5.4,3.9,1.3,0.4,Iris-setosa
5.1,3.5,1.4,0.3,Iris-setosa
5.7,3.8,1.7,0.3,Iris-setosa
5.1,3.8,1.5,0.3,Iris-setosa
5.4,3.4,1.7,0.2,Iris-setosa
5.1,3.7,1.5,0.4,Iris-setosa
4.6,3.6,1.0,0.2,Iris-setosa
5.1,3.3,1.7,0.5,Iris-setosa
4.8,3.4,1.9,0.2,Iris-setosa
5.0,3.0,1.6,0.2,Iris-setosa
5.0,3.4,1.6,0.4,Iris-setosa
5.2,3.5,1.5,0.2,Iris-setosa
5.2,3.4,1.4,0.2,Iris-setosa
4.7,3.2,1.6,0.2,Iris-setosa
4.8,3.1,1.6,0.2,Iris-setosa
5.4,3.4,1.5,0.4,Iris-setosa
5.2,4.1,1.5,0.1,Iris-setosa
5.5,4.2,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.0,3.2,1.2,0.2,Iris-setosa
5.5,3.5,1.3,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
4.4,3.0,1.3,0.2,Iris-setosa
5.1,3.4,1.5,0.2,Iris-setosa
5.0,3.5,1.3,0.3,Iris-setosa
4.5,2.3,1.3,0.3,Iris-setosa
4.4,3.2,1.3,0.2,Iris-setosa
5.0,3.5,1.6,0.6,Iris-setosa
5.1,3.8,1.9,0.4,Iris-setosa
4.8,3.0,1.4,0.3,Iris-setosa
5.1,3.8,1.6,0.2,Iris-setosa
4.6,3.2,1.4,0.2,Iris-setosa
5.3,3.7,1.5,0.2,Iris-setosa
5.0,3.3,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
6.5,2.8,4.6,1.5,Iris-versicolor
5.7,2.8,4.5,1.3,Iris-versicolor
6.3,3.3,4.7,1.6,Iris-versicolor
4.9,2.4,3.3,1.0,Iris-versicolor
6.6,2.9,4.6,1.3,Iris-versicolor
5.2,2.7,3.9,1.4,Iris-versicolor
5.0,2.0,3.5,1.0,Iris-versicolor
5.9,3.0,4.2,1.5,Iris-versicolor
6.0,2.2,4.0,1.0,Iris-versicolor
6.1,2.9,4.7,1.4,Iris-versicolor
5.6,2.9,3.6,1.3,Iris-versicolor
6.7,3.1,4.4,1.4,Iris-versicolor
5.6,3.0,4.5,1.5,Iris-versicolor
5.8,2.7,4.1,1.0,Iris-versicolor
6.2,2.2,4.5,1.5,Iris-versicolor
5.6,2.5,3.9,1.1,Iris-versicolor
5.9,3.2,4.8,1.8,Iris-versicolor
6.1,2.8,4.0,1.3,Iris-versicolor
6.3,2.5,4.9,1.5,Iris-versicolor
6.1,2.8,4.7,1.2,Iris-versicolor
6.4,2.9,4.3,1.3,Iris-versicolor
6.6,3.0,4.4,1.4,Iris-versicolor
6.8,2.8,4.8,1.4,Iris-versicolor
6.7,3.0,5.0,1.7,Iris-versicolor
6.0,2.9,4.5,1.5,Iris-versicolor
5.7,2.6,3.5,1.0,Iris-versicolor
5.5,2.4,3.8,1.1,Iris-versicolor
5.5,2.4,3.7,1.0,Iris-versicolor
5.8,2.7,3.9,1.2,Iris-versicolor
6.0,2.7,5.1,1.6,Iris-versicolor
5.4,3.0,4.5,1.5,Iris-versicolor
6.0,3.4,4.5,1.6,Iris-versicolor
6.7,3.1,4.7,1.5,Iris-versicolor
6.3,2.3,4.4,1.3,Iris-versicolor
5.6,3.0,4.1,1.3,Iris-versicolor
5.5,2.5,4.0,1.3,Iris-versicolor
5.5,2.6,4.4,1.2,Iris-versicolor
6.1,3.0,4.6,1.4,Iris-versicolor
5.8,2.6,4.0,1.2,Iris-versicolor
5.0,2.3,3.3,1.0,Iris-versicolor
5.6,2.7,4.2,1.3,Iris-versicolor
5.7,3.0,4.2,1.2,Iris-versicolor
5.7,2.9,4.2,1.3,Iris-versicolor
6.2,2.9,4.3,1.3,Iris-versicolor
5.1,2.5,3.0,1.1,Iris-versicolor
5.7,2.8,4.1,1.3,Iris-versicolor
"""

data = pd.read_csv(
    io.StringIO(DATA_CSV.strip()),
    header=None,
    names=["X1", "X2", "X3", "X4", "Species"]
)

print("\nJumlah seluruh data:", len(data))
print("\nJumlah data setiap kelas:")
print(data["Species"].value_counts())

data["Target"] = data["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1
})

if data["Target"].isna().any():
    label_tidak_dikenal = data.loc[
        data["Target"].isna(), "Species"
    ].unique()

    raise ValueError(
        f"Terdapat label yang tidak dikenali: {label_tidak_dikenal}"
    )

data_training = pd.concat(
    [
        data.iloc[0:40],
        data.iloc[50:90]
    ],
    ignore_index=True
)

data_validation = pd.concat(
    [
        data.iloc[40:50],
        data.iloc[90:100]
    ],
    ignore_index=True
)

fitur = ["X1", "X2", "X3", "X4"]

X_training = data_training[fitur].to_numpy(dtype=float)
y_training = data_training["Target"].to_numpy(dtype=int)

X_validation = data_validation[fitur].to_numpy(dtype=float)
y_validation = data_validation["Target"].to_numpy(dtype=int)

print("Jumlah data training  :", len(X_training))
print("Jumlah data validation:", len(X_validation))

print("\nJumlah kelas pada data training:")
print(data_training["Target"].value_counts().sort_index())

print("\nJumlah kelas pada data validation:")
print(data_validation["Target"].value_counts().sort_index())

learning_rate = 0.1
jumlah_epoch = 5

bias = 0.5
bobot = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)

print("\nLearning rate :", learning_rate)
print("Jumlah epoch  :", jumlah_epoch)
print("Bias awal     :", bias)
print("Bobot awal    :", bobot)

def sigmoid(z):
    """
    Mengubah nilai z menjadi nilai antara 0 dan 1.
    """
    return 1 / (1 + np.exp(-z))

hasil_training = []
hasil_validation = []

for epoch in range(1, jumlah_epoch + 1):

    total_square_error = 0.0
    jumlah_prediksi_benar = 0

    for nomor, (x, target) in enumerate(
        zip(X_training, y_training),
        start=1
    ):

        z = np.dot(x, bobot) + bias

        output = sigmoid(z)

        if output >= 0.5:
            prediksi = 1
        else:
            prediksi = 0

        error = output - target

        square_error = error ** 2

        total_square_error += square_error
        jumlah_prediksi_benar += int(prediksi == target)

        gradien = 2 * error * output * (1 - output)

        bias = bias - learning_rate * gradien
        bobot = bobot - learning_rate * gradien * x

    bias_validation = bias
    bobot_validation = bobot.copy()

    mse_training = total_square_error / len(X_training)

    accuracy_training = (
        jumlah_prediksi_benar / len(X_training)
    )

    hasil_training.append([
        epoch,
        mse_training,
        accuracy_training
    ])

    z_validation = (
        np.dot(X_validation, bobot_validation)
        + bias_validation
    )

    output_validation = sigmoid(z_validation)

    prediksi_validation = (
        output_validation >= 0.5
    ).astype(int)

    mse_validation = np.mean(
        (output_validation - y_validation) ** 2
    )

    accuracy_validation = np.mean(
        prediksi_validation == y_validation
    )

    hasil_validation.append([
        epoch,
        mse_validation,
        accuracy_validation
    ])

kolom = ["Epoch", "MSE", "Accuracy"]

tabel_training = pd.DataFrame(
    hasil_training,
    columns=kolom
)

tabel_validation = pd.DataFrame(
    hasil_validation,
    columns=kolom
)

print("\nHASIL TRAINING")
print(
    tabel_training[
        ["Epoch", "MSE", "Accuracy"]
    ]
)

print("\nHASIL VALIDATION")
print(
    tabel_validation[
        ["Epoch", "MSE", "Accuracy"]
    ]
)

def buat_grafik(
    epoch,
    nilai,
    judul,
    label_y,
    warna,
    nama_file,
    batas_persen=False
):
    plt.figure(figsize=(8, 5))

    plt.plot(
        epoch,
        nilai,
        marker="o",
        markersize=7,
        linewidth=2,
        color=warna
    )

    plt.title(
        judul,
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(label_y, fontsize=12)

    plt.xticks(
        range(1, jumlah_epoch + 1)
    )

    if batas_persen:
        plt.ylim(0, 1.05)

    plt.grid(
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.show()

    print(f"Menampilkan grafik: {nama_file}")

buat_grafik(
    epoch=tabel_training["Epoch"],
    nilai=tabel_training["Accuracy"],
    judul="Accuracy Data Training",
    label_y="Accuracy",
    warna="blue",
    nama_file="01_accuracy_training.png",
    batas_persen=True
)

buat_grafik(
    epoch=tabel_validation["Epoch"],
    nilai=tabel_validation["Accuracy"],
    judul="Accuracy Data Validation",
    label_y="Accuracy",
    warna="green",
    nama_file="02_accuracy_validation.png",
    batas_persen=True
)

buat_grafik(
    epoch=tabel_training["Epoch"],
    nilai=tabel_training["MSE"],
    judul="Loss Data Training",
    label_y="Mean Squared Error",
    warna="red",
    nama_file="03_loss_training.png"
)

buat_grafik(
    epoch=tabel_validation["Epoch"],
    nilai=tabel_validation["MSE"],
    judul="Loss Data Validation",
    label_y="Mean Squared Error",
    warna="orange",
    nama_file="04_loss_validation.png"
)

def buat_grafik_gabungan(
    epoch,
    nilai_training,
    nilai_validation,
    judul,
    label_y,
    nama_file,
    batas_persen=False
):
    plt.figure(figsize=(8, 5))

    plt.plot(
        epoch,
        nilai_training,
        marker="o",
        markersize=7,
        linewidth=2,
        color="blue",
        label="Training"
    )

    plt.plot(
        epoch,
        nilai_validation,
        marker="o",
        markersize=7,
        linewidth=2,
        color="green",
        label="Validation"
    )

    plt.title(
        judul,
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(label_y, fontsize=12)

    plt.xticks(
        range(1, jumlah_epoch + 1)
    )

    if batas_persen:
        plt.ylim(0, 1.05)

    plt.legend()

    plt.grid(
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.show()

    print(f"Menampilkan grafik: {nama_file}")

buat_grafik_gabungan(
    epoch=tabel_training["Epoch"],
    nilai_training=tabel_training["Accuracy"],
    nilai_validation=tabel_validation["Accuracy"],
    judul="Accuracy Training vs Validation",
    label_y="Accuracy",
    nama_file="05_accuracy_gabungan.png",
    batas_persen=True
)

buat_grafik_gabungan(
    epoch=tabel_training["Epoch"],
    nilai_training=tabel_training["MSE"],
    nilai_validation=tabel_validation["MSE"],
    judul="Loss Training vs Validation",
    label_y="Mean Squared Error",
    nama_file="06_loss_gabungan.png"
)