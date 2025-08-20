import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.table import Table
from math import exp

def gaussian_kernel_1d(kernel_size=7, sigma=1.0):
    assert kernel_size % 2 == 1
    half = kernel_size // 2
    kernel = np.array([exp(-(x**2)/(2*sigma**2)) for x in range(-half, half+1)])
    kernel /= kernel.sum()
    return kernel.astype(np.float32)



def custom_gaussian_blur(image, kernel_size=7, sigma=1.0):
    kernel = gaussian_kernel_1d(kernel_size, sigma)
    height, width, channels = image.shape
    temp = np.zeros_like(image, dtype=np.float32)
    output = np.zeros_like(image, dtype=np.float32)
    for c in range(channels):
        for i in range(height):
            temp[i, :, c] = np.convolve(image[i, :, c], kernel, mode='same')
    for c in range(channels):
        for j in range(width):
            output[:, j, c] = np.convolve(temp[:, j, c], kernel, mode='same')
    return output.astype(np.uint8)

def gauss_opencv2_separable(image, kernel_size=7, sigma=1.0):
    kernel_1d = gaussian_kernel_1d(kernel_size, sigma)
    return cv2.sepFilter2D(image, -1, kernel_1d, kernel_1d)

def gauss_opencv2_filter2d(image, kernel_size=7, sigma=1.0):
    kernel_1d = gaussian_kernel_1d(kernel_size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return cv2.filter2D(image, -1, kernel_2d)

def gauss_opencv_double(image, kernel_size=7, sigma=1.0):
    kernel_1d = gaussian_kernel_1d(kernel_size, sigma).reshape(-1, 1)
    return cv2.filter2D(cv2.filter2D(image, -1, kernel_1d), -1, kernel_1d.T)

def plot_timing_table(data, headers, title="Gaussian Blur Time Comparison"):
    fig, ax = plt.subplots(figsize=(18, len(data) * 0.4 + 2))
    ax.set_axis_off()
    table = Table(ax)
    n_rows = len(data) + 1
    n_cols = len(headers)
    width, height = 1.0 / n_cols, 1.0 / n_rows
    for i, header in enumerate(headers):
        cell = table.add_cell(0, i, width, height, text=header, loc='center', facecolor='#40466e')
        cell.get_text().set_color('white')
    for row_idx, row in enumerate(data, start=1):
        for col_idx, cell_val in enumerate(row):
            cell = table.add_cell(row_idx, col_idx, width, height, text=str(cell_val), loc='center',
                                  facecolor='#f1f1f2' if row_idx % 2 == 0 else 'white')
            cell.get_text().set_fontsize(10)
    ax.add_table(table)
    plt.title(title, fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()

image_sizes = [512, 640, 720, 1080, 1440, 1920]
kernel_sizes = [3, 7, 15, 33, 65, 99]
num_runs = 5

timing_table = []
headers = ["Image Size", "Kernel", "cv2.GaussianBlur", "Custom", "sepFilter2D", "filter2D", "DoubleFilter2D"]

for size in image_sizes:
    for ksize in kernel_sizes:
        img = (np.random.rand(size, size, 3) * 255).astype(np.uint8)

        def time_avg(func):
            start = time.time()
            for _ in range(num_runs):
                _ = func()
            end = time.time()
            return (end - start) / num_runs

        t_cv = time_avg(lambda: cv2.GaussianBlur(img, (ksize, ksize), sigmaX=1))
        t_custom = time_avg(lambda: custom_gaussian_blur(img, ksize, 1))
        t_sep = time_avg(lambda: gauss_opencv2_separable(img, ksize, 1))
        t_f2d = time_avg(lambda: gauss_opencv2_filter2d(img, ksize, 1))
        t_double = time_avg(lambda: gauss_opencv_double(img, ksize, 1))

        times = [t_cv, t_custom, t_sep, t_f2d, t_double]
        min_time = min(times)

        def format_time(t):
            ratio = round(t / min_time, 2)
            return f"{t:.4f}s (x{ratio})"

        row = [f"{size}x{size}", ksize] + [format_time(t) for t in times]
        timing_table.append(row)

        print(f"[{size}x{size}, kernel={ksize}] => "
              f"cv2: {format_time(t_cv)}, custom: {format_time(t_custom)}, "
              f"sep: {format_time(t_sep)}, f2d: {format_time(t_f2d)}, double: {format_time(t_double)}")

plot_timing_table(timing_table, headers)
