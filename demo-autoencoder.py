from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.colors

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


def print_img(array1, array2, score, name=""):
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle(f"{name}: Error={score:.4f}", fontsize=20)
    colors = ['white', 'black']
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('name', colors)
    norm = plt.Normalize(0, 1)
    ax1.imshow(array1, cmap=cmap, norm=norm, interpolation="none")
    ax1.axis('off')
    ax2.imshow(array2, cmap=cmap, norm=norm, interpolation="none")
    ax2.axis('off')
    plt.savefig(f"imgs/demo-{name}.png", format="png")


def print_imgs(encoded_imgs, decoded_imgs, scores, names, prefix=""):
    for i in range(len(encoded_imgs)):
        print_img(encoded_imgs[i], decoded_imgs[i], scores[i], prefix+names[i])


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation="relu")
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(15, activation="sigmoid"),
            layers.Reshape((5, 3))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_demo_imgs():
    r1 = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ])

    r1b = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ])

    r2 = np.array([
        [1, 1, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1]
    ])

    r2b = np.array([
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ])

    r3 = np.array([
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ])

    r3b = np.array([
        [1, 1, 1],
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ])

    r4 = np.array([
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1]
    ])

    r5 = np.array([
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ])

    r6 = np.array([
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    r7 = np.array([
        [1, 1, 1],
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1]
    ])

    r7b = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ])

    r8 = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    r9 = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1]
    ])

    rU = np.array([
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    rP = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ])

    rPb = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 0, 0],
        [1, 0, 0]
    ])

    rX = np.array([
        [1, 0, 1],
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
        [1, 0, 1]
    ])

    rY = np.array([
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 0]
    ])

    rJ = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    rC = np.array([
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1]
    ])

    rK = np.array([
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [1, 0, 1]
    ])

    rF = np.array([
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ])

    rFb = np.array([
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 0],
        [1, 0, 0]
    ])

    ru = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0]
    ])

    rH = np.array([
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1]
    ])

    r0 = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    return {
        "One_a": r1,
        "One_b": r1b,
        "Two_a": r2,
        "Two_b": r2b,
        "Three_a": r3,
        "Three_b": r3b,
        "Four_a": r4,
        "Five": r5,
        "Six": r6,
        "Seven_a": r7,
        "Seven_b": r7b,
        "Eight": r8,
        "Nine": r9,
        "Zero": r0,
        "U": rU,
        "U_lower": ru,
        "P_a": rP,
        "P_b": rPb,
        "X": rX,
        "Y": rY,
        "J": rJ,
        "C": rC,
        "K": rK,
        "F_a": rF,
        "F_b": rFb,
        "H": rH,
    }


def basic_encode(array):
    r0 = sum(array[0])/len(array[0])
    r1 = sum(array[1])/len(array[1])
    r2 = sum(array[2])/len(array[1])
    r3 = sum(array[3])/len(array[1])
    r4 = sum(array[4])/len(array[1])

    l = len(array)

    c0 = sum([array[i][0] for i in range(l)])/l
    c1 = sum([array[i][1] for i in range(l)])/l
    c2 = sum([array[i][2] for i in range(l)])/l

    return {"r": [r0, r1, r2, r3, r4], "c": [c0, c1, c2]}


def basic_decode(encoding):
    encoded = [[(encoding["r"][r] + encoding["c"][c])/2 for c in range(3)] for r in range(5)]
    new_arr = np.array(encoded)
    return new_arr


def get_mse(base, new):
    #     Assumes same shape
    return (
            sum([sum([(base[r][c] - new[r][c])**2 for r in range(len(base))]) for c in range(len(base[0]))]) /
           (len(base[0])*len(base))
    )


def demo_entrypoint():

    imgs = get_demo_imgs()

    demo_encoded_1 = basic_encode(imgs["One_b"])
    print(f"Demo encoded 1: {demo_encoded_1}")
    demo_decoded_1 = basic_decode(demo_encoded_1)
    print(f"Demo 1: {demo_decoded_1}")

    demo_encoded_2 = basic_encode(imgs["Two_b"])
    demo_decoded_2 = basic_decode(demo_encoded_2)

    demo_normalised_1 = demo_decoded_1 * 1/np.amax(demo_decoded_1)
    demo_normalised_2 = demo_decoded_2 * 1/np.amax(demo_decoded_2)

    print(f"Demo 1: {demo_decoded_1}")
    print(f"Demo 2: {demo_decoded_2}")

    print_img(imgs["One_b"], demo_normalised_1, get_mse(imgs["One_b"], demo_decoded_1), name="basic_1b")
    print_img(imgs["Two_b"], demo_normalised_2, get_mse(imgs["Two_b"], demo_decoded_2), name="basic_2b")

    autoencoder = Autoencoder(latent_dim=8)
    autoencoder.compile(optimizer='Adam', loss=losses.MeanSquaredError())

    proportion_train = 0.7
    shuffled_keys = list(imgs.keys())
    shuffle(shuffled_keys)

    lim_train = int(len(shuffled_keys) * proportion_train)
    train_keys = shuffled_keys[0:lim_train]
    test_keys = shuffled_keys[lim_train+1:]

    x_train = np.array([imgs[x] for x in train_keys])
    x_test = np.array([imgs[x] for x in test_keys])

    autoencoder.fit(x_train, x_train, epochs=2000, shuffle=True, validation_data=(x_test, x_test))

    train_images = autoencoder.encoder(x_train).numpy()
    decoded_train_images = autoencoder.decoder(train_images).numpy()

    train_scores = [get_mse(x_train[i], decoded_train_images[i]) for i in range(len(x_train))]
    print_imgs(x_train, decoded_train_images, train_scores, train_keys, prefix="train")

    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    scores = [get_mse(x_test[i], decoded_imgs[i]) for i in range(len(encoded_imgs))]

    net_mse = sum(scores)/len(scores)
    print(f"net_mse: {net_mse}")

    print_imgs(x_test, decoded_imgs, scores, test_keys)


if __name__ == "__main__":
    demo_entrypoint()
