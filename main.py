import matplotlib.pyplot as plt
import numpy as np
import copy

#1,2
my_img = plt.imread("C:/Users/kozlo/OneDrive/Изображения/jyZFO6vZB54.jpg")
plt.figure(figsize=(8, 8))
plt.imshow(my_img)
plt.title("Исходное изображение")
plt.show()

#3
r_channel = my_img[:, :, 0]
g_channel = my_img[:, :, 1]
b_channel = my_img[:, :, 2]

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
reg_img = my_img.copy()
reg_img[:, :, 1] = 0
reg_img[:, :, 2] = 0
plt.imshow(reg_img, cmap='Reds')
plt.title("Красный канал")

plt.subplot(1, 3, 2)
reg_img = my_img.copy()
reg_img[:, :, 0] = 0
reg_img[:, :, 2] = 0
plt.imshow(reg_img, cmap='Greens')
plt.title("Зеленый канал")

plt.subplot(1, 3, 3)
reg_img = my_img.copy()
reg_img[:, :, 1] = 0
reg_img[:, :, 0] = 0
plt.imshow(reg_img, cmap='Blues')
plt.title("Синий канал")

plt.tight_layout()
plt.show()

plt.show()

#4
def rgb_to_ycbcr(im):
    r = im[:, :, 0]
    g = im[:, :, 1]
    b = im[:, :, 2]
    # Y
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y)
    cr = 0.713 * (r - y)
    return y, cb, cr
Y, Cb, Cr = rgb_to_ycbcr(my_img)

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(Y, cmap='gray')
plt.title("Яркостная компонента Y")

plt.subplot(1, 3, 2)
plt.imshow(Cb, cmap='gray')
plt.title("Цветоразностная компонента Cb")

plt.subplot(1, 3, 3)
plt.imshow(Cr, cmap='gray')
plt.title("Цветоразностная компонента Cr")

plt.tight_layout()
plt.show()




#5
img_dwns = Y
img_dwns = img_dwns[::5, ::2]#colom/row

fig = plt.figure(figsize=(5, 5))

plt.imshow(img_dwns, cmap='gray')

plt.show()

#6

Y_scaled = (Y - np.min(Y)) / (np.max(Y) - np.min(Y)) * 255
Y_scaled = Y_scaled.astype(np.uint8)#norm

bits = np.zeros((8, Y_scaled.shape[0], Y_scaled.shape[1]), dtype=np.uint8)

masks = [1, 2, 4, 8, 16, 32, 64, 128]
for i in range(8):
    bits[i] = (Y_scaled & masks[i]) >> i

titles = ["bit 0", "bit 1", "bit 2", "bit 3", "bit 4", "bit 5", "bit 6", "bit 7"]

fig, axs = plt.subplots(2, 4, figsize=(12, 6))

for i in range(8):
    row = i // 4
    col = i % 4
    ax = axs[row, col]

    ax.imshow(bits[7 - i], cmap='gray')
    ax.set_title(titles[i])

plt.tight_layout()
plt.show()

# #task 7
#
# bit_count_x = [{"00": 0, "01": 0, "10": 0, "11": 0} for k in range(8)]
#
# k = 0
# for k in range(8):
#
#     for j in range(len(bits[k][1, :]) - 1):
#
#         for i in range(len(bits[k][:, 1])):
#
#             if bits[k][i, j] == 0 and bits[k][i, j+1] == 0:
#                 bit_count_x[k]["00"] = bit_count_x[k]["00"] + 1
#             elif bits[k][i, j] == 0 and bits[k][i, j+1] == 1:
#                 bit_count_x[k]["01"] = bit_count_x[k]["01"] + 1
#             elif bits[k][i, j] == 1 and bits[k][i, j+1] == 0:
#                 bit_count_x[k]["10"] = bit_count_x[k]["10"] + 1
#             elif bits[k][i, j] == 1 and bits[k][i, j+1] == 1:
#                 bit_count_x[k]["11"] = bit_count_x[k]["11"] + 1
#
# fig, ax = plt.subplots(2, 4, figsize = (20, 10))
#
# ax[0][0].stem(["00", "01", "10", "11"], [bit_count_x[0]["00"],bit_count_x[0]["01"], bit_count_x[0]["10"],
# bit_count_x[0]["11"]])
# ax[0][0].set_title("7 bit")
# ax[0][1].stem(["00", "01", "10", "11"], [bit_count_x[1]["00"],bit_count_x[1]["01"], bit_count_x[1]["10"],
# bit_count_x[1]["11"]])
# ax[0][1].set_title("6 bit")
# ax[0][2].stem(["00", "01", "10", "11"], [bit_count_x[2]["00"],bit_count_x[2]["01"], bit_count_x[2]["10"],
# bit_count_x[2]["11"]])
# ax[0][2].set_title("5 bit")
# ax[0][3].stem(["00", "01", "10", "11"], [bit_count_x[3]["00"],bit_count_x[3]["01"], bit_count_x[3]["10"],
# bit_count_x[3]["11"]])
# ax[0][3].set_title("4 bit")
# ax[1][0].stem(["00", "01", "10", "11"], [bit_count_x[4]["00"],bit_count_x[4]["01"], bit_count_x[4]["10"],
# bit_count_x[4]["11"]])
# ax[1][0].set_title("3 bit")
# ax[1][1].stem(["00", "01", "10", "11"], [bit_count_x[5]["00"],bit_count_x[5]["01"], bit_count_x[5]["10"],
# bit_count_x[5]["11"]])
# ax[1][1].set_title("2 bit")
# ax[1][2].stem(["00", "01", "10", "11"], [bit_count_x[6]["00"],bit_count_x[6]["01"], bit_count_x[6]["10"],
# bit_count_x[6]["11"]])
# ax[1][2].set_title("1 bit")
# ax[1][3].stem(["00", "01", "10", "11"], [bit_count_x[7]["00"],bit_count_x[7]["01"], bit_count_x[7]["10"],
# bit_count_x[7]["11"]])
# ax[1][3].set_title("0 bit")
# plt.show()
# plt.tight_layout()

Cr_scaled = (Cr - np.min(Cr)) / (np.max(Cr) - np.min(Cr)) * 255
Cr_scaled = Cr_scaled.astype(np.uint8)

bits_cr = np.zeros((8, Cr_scaled.shape[0], Cr_scaled.shape[1]), dtype=np.uint8)

masks = [1, 2, 4, 8, 16, 32, 64, 128]
for i in range(8):
    bits_cr[i] = (Cr_scaled & masks[i]) >> i

titles = ["bit 0", "bit 1", "bit 2", "bit 3", "bit 4", "bit 5", "bit 6", "bit 7"]

fig, axs = plt.subplots(2, 4, figsize=(12, 6))

for i in range(8):
    row = i // 4
    col = i % 4
    ax = axs[row, col]

    ax.imshow(bits_cr[7 - i], cmap='gray')
    ax.set_title(titles[i])

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(bits[6 + i], cmap='gray')
    plt.title(f"Bit {6 + i} (Y)")

plt.figure(figsize=(8, 8))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(bits_cr[6 + i], cmap='gray')
    plt.title(f"Bit {6 + i} (Cr)")
plt.show()


Y = bits[7]  # Используем 2 старших бита из Y
Cr = bits_cr[7]  # Используем 2 старших бита из Cr


R = np.zeros(Y.shape, dtype=np.uint8)
G = np.zeros(Y.shape, dtype=np.uint8)
B = np.zeros(Y.shape, dtype=np.uint8)

# Восстановление компонент R, G и B
R = Y + 1.13983 * Cr
G = Y - 0.39465 * Cb - 0.58060 * Cr
B = Y + 2.03211 * Cb



R = np.clip(R, 0, 255).astype(np.uint8)
G = np.clip(G, 0, 255).astype(np.uint8)
B = np.clip(B, 0, 255).astype(np.uint8)

restored_rgb_image = np.dstack((R, G, B))


plt.figure(figsize=(6, 6))
plt.imshow(restored_rgb_image)
plt.title("Восстановленное RGB-изображение из 2 старших битов Y и Cr")
plt.show()


print("Значения битовых плоскостей Y:")
for i in range(2):
    print(f"Bit {i} (Y):")
    print(bits[i])

print("Значения битовых плоскостей Cr:")
for i in range(2):
    print(f"Bit {i} (Cr):")
    print(bits_cr[i])
