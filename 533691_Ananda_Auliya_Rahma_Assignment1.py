import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NAMA_FILE = "iris.csv"

data = pd.read_csv(
    NAMA_FILE,
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

data_validation = data.groupby("Species").tail(10)

data_training = data.drop(data_validation.index)

data_training = data_training.reset_index(drop=True)
data_validation = data_validation.reset_index(drop=True)

print("\nJumlah data training  :", len(data_training))
print("Jumlah data validation:", len(data_validation))

print("\nJumlah kelas pada data training:")
print(data_training["Target"].value_counts().sort_index())

print("\nJumlah kelas pada data validation:")
print(data_validation["Target"].value_counts().sort_index())

fitur = ["X1", "X2", "X3", "X4"]

X_training = data_training[fitur].to_numpy(dtype=float)
y_training = data_training["Target"].to_numpy(dtype=int)

X_validation = data_validation[fitur].to_numpy(dtype=float)
y_validation = data_validation["Target"].to_numpy(dtype=int)

learning_rate = 0.1
jumlah_epoch = 5

bias_awal = 0.5

bobot_awal = np.array(
    [0.5, 0.5, 0.5, 0.5],
    dtype=float
)

print("\nParameter Model")
print("Learning rate :", learning_rate)
print("Jumlah epoch  :", jumlah_epoch)
print("Bias awal     :", bias_awal)
print("Bobot awal    :", bobot_awal)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hitung_output(x, bobot, bias):
    z = np.dot(x, bobot) + bias
    return sigmoid(z)

def tentukan_prediksi(output):
    if output >= 0.5:
        return 1
    else:
        return 0

def hitung_error(output, target):
    return output - target

def hitung_square_error(error):
    return error ** 2

def hitung_gradien(error, output):
    return 2 * error * output * (1 - output)

def update_parameter(
    bobot,
    bias,
    gradien,
    x,
    learning_rate
):

    bias_baru = bias - learning_rate * gradien

    bobot_baru = (
        bobot
        - learning_rate * gradien * x
    )

    return bobot_baru, bias_baru

def evaluasi_validation(
    X_validation,
    y_validation,
    bobot,
    bias
):

    z_validation = (
        np.dot(X_validation, bobot)
        + bias
    )

    output_validation = sigmoid(
        z_validation
    )

    prediksi_validation = (
        output_validation >= 0.5
    ).astype(int)

    loss_validation = np.mean(
        (output_validation - y_validation) ** 2
    )

    accuracy_validation = np.mean(
        prediksi_validation == y_validation
    )

    return loss_validation, accuracy_validation

def train_model(
    X_training,
    y_training,
    X_validation,
    y_validation,
    learning_rate,
    jumlah_epoch,
    bobot_awal,
    bias_awal
):

    bobot = bobot_awal.copy()
    bias = bias_awal

    hasil_training = []
    hasil_validation = []

    for epoch in range(1, jumlah_epoch + 1):

        total_loss = 0.0
        jumlah_prediksi_benar = 0

        for x, target in zip(
            X_training,
            y_training
        ):

            output = hitung_output(
                x,
                bobot,
                bias
            )

            prediksi = tentukan_prediksi(
                output
            )

            error = hitung_error(
                output,
                target
            )

            square_error = hitung_square_error(
                error
            )

            total_loss += square_error

            jumlah_prediksi_benar += int(
                prediksi == target
            )

            gradien = hitung_gradien(
                error,
                output
            )

            bobot, bias = update_parameter(
                bobot,
                bias,
                gradien,
                x,
                learning_rate
            )

        loss_training = (
            total_loss
            / len(X_training)
        )

        accuracy_training = (
            jumlah_prediksi_benar
            / len(X_training)
        )

        hasil_training.append([
            epoch,
            loss_training,
            accuracy_training
        ])

        loss_validation, accuracy_validation = (
            evaluasi_validation(
                X_validation,
                y_validation,
                bobot,
                bias
            )
        )

        hasil_validation.append([
            epoch,
            loss_validation,
            accuracy_validation
        ])

    kolom = [
        "Epoch",
        "Loss",
        "Accuracy"
    ]

    tabel_training = pd.DataFrame(
        hasil_training,
        columns=kolom
    )

    tabel_validation = pd.DataFrame(
        hasil_validation,
        columns=kolom
    )

    return (
        tabel_training,
        tabel_validation,
        bobot,
        bias
    )

(
    tabel_training,
    tabel_validation,
    bobot_akhir,
    bias_akhir
) = train_model(
    X_training,
    y_training,
    X_validation,
    y_validation,
    learning_rate,
    jumlah_epoch,
    bobot_awal,
    bias_awal
)

print("\nHASIL TRAINING")

print(
    tabel_training[
        ["Epoch", "Loss", "Accuracy"]
    ]
)

print("\nHASIL VALIDATION")

print(
    tabel_validation[
        ["Epoch", "Loss", "Accuracy"]
    ]
)

print("\nPARAMETER AKHIR")

print("Bobot akhir:")
print(bobot_akhir)

print("\nBias akhir:")
print(bias_akhir)

def buat_grafik(
    epoch,
    nilai_training,
    nilai_validation,
    judul,
    label_y,
    batas_persen=False
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        epoch,
        nilai_training,
        marker="o",
        markersize=7,
        linewidth=2,
        label="Training"
    )

    plt.plot(
        epoch,
        nilai_validation,
        marker="o",
        markersize=7,
        linewidth=2,
        label="Validation"
    )

    plt.title(
        judul,
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel(
        "Epoch",
        fontsize=12
    )

    plt.ylabel(
        label_y,
        fontsize=12
    )

    plt.xticks(
        range(1, jumlah_epoch + 1)
    )

    if batas_persen:
        plt.ylim(0, 1)
        plt.yticks([0, 0.25, 0.5, 0.75, 1])

    plt.legend()

    plt.grid(
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.show()

buat_grafik(
    epoch=tabel_training["Epoch"],
    nilai_training=tabel_training["Accuracy"],
    nilai_validation=tabel_validation["Accuracy"],
    judul="Accuracy Training vs Validation",
    label_y="Accuracy",
    batas_persen=True
)

buat_grafik(
    epoch=tabel_training["Epoch"],
    nilai_training=tabel_training["Loss"],
    nilai_validation=tabel_validation["Loss"],
    judul="Loss Training vs Validation",
    label_y="Loss"
)
