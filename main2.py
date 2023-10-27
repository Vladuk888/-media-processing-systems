import numpy as np
import math
import matplotlib.pyplot as plt
from math import pi, sin, cos, sqrt
from scipy.integrate import quad
import soundfile as sf
from PIL import Image

T = 8/1000
k_max = 10
ak, bk = [], []
A_k, phi = np.zeros(k_max), np.zeros(k_max)


def calculate_bk(k):
    def integrand1(t):
        return 0

    def integrand2(t):
        return (2/T) * (-t+0.5) * np.sin((2*np.pi*k*t)/T)

    def integrand3(t):
        return 0

    result1, _ = quad(integrand1, 0, T/4)
    result2, _ = quad(integrand2, T/4, 3*T/4)
    result3, _ = quad(integrand3, 3*T/4, T)

    return result1 + result2 + result3


def calculate_ak(k):
    def integrand1(t):
        return 0

    def integrand2(t):
        return (2/T) * (-t+0.5) * np.cos((2*np.pi*k*t)/T)

    def integrand3(t):
        return 0

    result1, _ = quad(integrand1, 0, T/4)
    result2, _ = quad(integrand2, T/4, 3*T/4)
    result3, _ = quad(integrand3, 3*T/4, T)

    return result1 + result2 + result3


def calculate_a0():
    def integrand1(t):
        return 0

    def integrand2(t):
        return (-t+0.5)

    def integrand3(t):
        return 0

    result1, _ = quad(integrand1, 0, T/4)
    result2, _ = quad(integrand2, T/4, 3*T/4)
    result3, _ = quad(integrand3, 3*T/4, T)

    return (2/T) * (result1 + result2 + result3)


a0 = calculate_a0()

fs = 44100
x_t = np.zeros((int(fs*0.01)))

for k in range(1, k_max):
    a_k = calculate_ak(k)
    b_k = calculate_bk(k)
    ak.append(a_k)
    bk.append(b_k)

for n in range(len(x_t)):
    t = n / fs
    x_t[n] = a0 / 2
    for k in range(1, k_max):
        x_t[n] = x_t[n] + (ak[k-1]*cos((2*pi*k) * t / T) + bk[k-1]*sin((2*pi*k) * t / T))

sf.write('before.wav', x_t, 44100)

for l in range(1, k_max):
    A_k[l] = np.sqrt(ak[l-1] ** 2 + bk[l-1] ** 2)
    phi[l] = np.random.random()

x_t2 = np.zeros((int(fs * 0.01)))

for z in range(len(x_t2)):
    t = z / fs
    x_t2[z] = a0 / 2
    for k in range(k_max):
        x_t2[z] = x_t2[z] + A_k[k] * np.cos((2 * pi * k) * t / T + phi[k])

filename2 = "output2.wav"

sf.write(filename2, x_t2, int(fs))


n = len(x_t)
part_size = n // 3
part1 = x_t[:part_size]
part2 = x_t[part_size:2*part_size]
part3 = x_t[2*part_size:]

x_t_modified = np.concatenate([part3, part2, part1])


plt.plot(x_t_modified)
plt.title("Сигнал после перемены местами первой и третей частей")
plt.xlabel("Время")
plt.ylabel("Значение")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(x_t)
plt.title("График x_t")
plt.xlabel("Отсчеты")
plt.ylabel("Значение")
plt.grid(True)

# Второй график
plt.figure(figsize=(8, 4))
plt.plot(x_t2)
plt.title("График x_t2")
plt.xlabel("Отсчеты")
plt.ylabel("Значение")
plt.grid(True)

plt.show()

k_values = np.arange(1, k_max)

A_k_values = np.zeros(k_max)
phi_values = np.zeros(k_max)

for k in k_values:
    A_k_values[k] = A_k[k]
    phi_values[k] = phi[k]

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(k_values, A_k_values[1:], marker='o', linestyle='-')
plt.title('График зависимости A_k от k')
plt.xlabel('k')
plt.ylabel('ak')

plt.subplot(2, 1, 2)
plt.plot(k_values, phi_values[1:], marker='o', linestyle='-')
plt.title('График зависимости phi от k')
plt.xlabel('k')
plt.ylabel('bk')

plt.tight_layout()
plt.show()