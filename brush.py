from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


def hat(x, sharpness, func=erf):
    return (func((1.-np.abs(x*2.5))*sharpness)*0.5+0.5)


def hat_img(power, smoothness, start, end, func=erf):
    w = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    h = 2
    vector1 = [1, 0]
    vector2 = end-start
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)

    theta = np.arccos(dot_product)
    radius = 100
    size = int(radius)*4  # image size
    center = start/radius  # center (for subpixel rendering)
    x = np.linspace((-size / 2) / radius, (size / 2) / radius, 1280)
    print(x.shape)
    y = np.linspace((-size / 2) / radius, (size / 2) /
                    radius, 720).reshape(-1, 1)
    print("y:", y.shape)

    # rotation
    ct, st = np.cos(theta), np.sin(theta)
    xr = x * ct - y * st - center[0]
    yr = x * st + y * ct - center[1]
    # scale
    x = xr / (float(w)/radius)
    y = yr / (float(h) / radius)
    img = np.sqrt(abs(x ** power) + abs(y ** power))
    img = hat(img, smoothness, func)
    return img


plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
img2 = hat_img(8.0, 2, np.array([0, 0]), np.array([0, 20]))
plt.imshow(img2 * 255)
plt.subplot(1, 2, 2)
img = hat_img(8.0, 2, np.array([50, 0]), np.array([100, 100]))
print(img2)

plt.imshow(img*255)
plt.show()
