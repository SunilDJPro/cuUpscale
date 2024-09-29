# cuUpscale

![CUDA 12](https://img.shields.io/badge/CUDA-12-brightgreen)
![CPU_BUILD](https://img.shields.io/badge/CPU_BUILD-passing-green)
![GitHub repo size](https://img.shields.io/github/repo-size/SunilDJPro/cuUpscale)
![GitHub language count](https://img.shields.io/github/languages/count/SunilDJPro/cuUpscale)
![GitHub top language](https://img.shields.io/github/languages/top/SunilDJPro/cuUpscale)

###       POWERED BY
![POWERED BY CUDA](https://upload.wikimedia.org/wikipedia/en/thumb/b/b9/Nvidia_CUDA_Logo.jpg/220px-Nvidia_CUDA_Logo.jpg)


cuUpscale is a high-performance image upscaling tool that supports both CPU and CUDA-based GPU upscaling techniques. This project is designed to handle large resolutions with efficient parallelization using OpenMP for CPUs and CUDA for GPUs.

---

## ✨ Features

### 🚀 **cpuUpscale**
- Performs **Bicubic**, **Lanczos**, and **Edge Directional Interpolation (EDI)** upscaling with **multi-threading** enabled.
- Supports up to **x8 scaling** from the original image with a maximum resolution of **8K**.
- Utilizes **OpenMP** to parallelize loops at the CPU level, speeding up the upscaling process.

### ⚡ **cuUpscale**
- Performs **Lanczos** and **Edge Directed Interpolation (EDI)** upscaling using **CUDA C/C++** with support for **CUDA 12**. (Updated from original 11.8 build)
- Capable of scaling images up to **x16** and handling input/output images with resolutions up to **32K** without overflow issues.
- Quadratic loop complexity (**O(n^3)** worst case), but efficiently parallelized across **Streaming Multiprocessors**.
  - Tested on **RTX 3080Ti** with **80 SMs**, successfully upscaling from **4K to 16K** with high performance.

---

## 🛠️ Technologies Used

- **CUDA 12** for GPU acceleration.
- **OpenMP** for CPU parallelism.


---

## 🚀 Getting Started

To use this tool, you can choose between the **cpuUpscale** or **cuUpscale** methods depending on your system's capabilities and the required resolution. 

---

### 📝 License

This project is licensed under the MIT License.
