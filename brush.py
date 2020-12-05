from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


def hat(x, sharpness, func=erf):
    return (func((1.-np.abs(x*2.5))*sharpness)*0.5+0.5)


def radial_hat(radius, smoothness, center=np.array([0, 0]), func=erf):
    width = 1279  # image size
    height = 719
    center = np.array(center)/radius  # center (for subpixel rendering)
    x = np.linspace((-width/2)/radius, (width/2)/radius, width+1)
    y = np.linspace((-height / 2) / radius, (height / 2) /
                    radius, height + 1).reshape(-1, 1)

    img = (x**2 + y**2)
    img = hat(img, smoothness, func)  # scurve)
    return img


def hat_img(power, smoothness, theta, center=np.array([0, 0]), func=erf):
    radius = 10
    width = 1279  # image size
    height = 719
    center = np.array(center) / radius

    x = np.linspace((-width/2)/radius, (width/2)/radius, width+1)
    y = np.linspace((-height / 2) / radius, (height / 2) /
                    radius, height + 1).reshape(-1, 1)

    # rotation
    ct, st = np.cos(theta), np.sin(theta)
    xr = x * ct - y * st - center[0]
    yr = x * st + y * ct - center[1]
    # scale
    x = xr / (float(width)/radius)
    y = yr / (float(height) / radius)

    # superellipse distance
    img = np.sqrt(abs(x ** power) + abs(y ** power))
    img = hat(img, smoothness, func)
    return img


plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
img2 = hat_img(8.0, 2, 0.5, np.array([0, 0]))
plt.imshow(img2 * 255)
plt.subplot(1, 2, 2)
img = hat_img(8.0, 2, 0.5, np.array([50, 0]))
print(img2)

plt.imshow(img*255)
plt.show()
