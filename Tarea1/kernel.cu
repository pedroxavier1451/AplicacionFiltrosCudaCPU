#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // Para cargar imágenes PNG o JPEG
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // Para guardar imágenes
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono> // Para medir el tiempo de ejecución

#define PI 3.14159265358979323846

// Función para crear el kernel de Gabor
std::vector<std::vector<double>> createGaborKernel(int kernel_size, double sigma, double theta, double lambda, double gamma, double psi) {
    std::vector<std::vector<double>> kernel(kernel_size, std::vector<double>(kernel_size));

    int half_size = kernel_size / 2;

    for (int y = -half_size; y <= half_size; ++y) {
        for (int x = -half_size; x <= half_size; ++x) {
            double x_theta = x * cos(theta) + y * sin(theta);
            double y_theta = -x * sin(theta) + y * cos(theta);
            double gauss = exp(-(x_theta * x_theta + gamma * gamma * y_theta * y_theta) / (2 * sigma * sigma));
            double sinusoid = cos(2 * PI * x_theta / lambda + psi);
            kernel[y + half_size][x + half_size] = gauss * sinusoid;
        }
    }

    return kernel;
}

// Función para crear el kernel Emboss
std::vector<std::vector<double>> createEmbossKernel() {
    return {
        {-2, -1, 0},
        {-1,  1, 1},
        { 0,  1, 2}
    };
}

// Función para crear un kernel Gaussiano
std::vector<std::vector<double>> createGaussianKernel(int kernel_size, double sigma) {
    std::vector<std::vector<double>> kernel(kernel_size, std::vector<double>(kernel_size));
    int half_size = kernel_size / 2;
    double sum = 0.0;

    for (int y = -half_size; y <= half_size; ++y) {
        for (int x = -half_size; x <= half_size; ++x) {
            double value = (1 / (2 * PI * sigma * sigma)) * exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[y + half_size][x + half_size] = value;
            sum += value;
        }
    }

    // Normalizar el kernel
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            kernel[y][x] /= sum;
        }
    }

    return kernel;
}

// Función para aplicar la convolución
std::vector<std::vector<double>> applyConvolution(const std::vector<std::vector<unsigned char>>& image, const std::vector<std::vector<double>>& kernel, int width, int height, int kernel_size) {
    int half_kernel = kernel_size / 2;
    std::vector<std::vector<double>> result(height, std::vector<double>(width, 0));

    for (int y = half_kernel; y < height - half_kernel; ++y) {
        for (int x = half_kernel; x < width - half_kernel; ++x) {
            double sum = 0.0;
            for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
                for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
                    int pixel_value = image[y + ky][x + kx];
                    sum += pixel_value * kernel[ky + half_kernel][kx + half_kernel];
                }
            }
            result[y][x] = sum;
        }
    }
    return result;
}


int main()
{
    // Variables para la imagen
    int width, height, channels;

    // Cargar la imagen usando stb_image
    unsigned char* h_imageData = stbi_load("imagen1.jpg", &width, &height, &channels, 0);
    if (!h_imageData) {
        printf("Error al cargar la imagen\n");
        return -1;
    }

    if (channels < 3) {
        printf("La imagen no tiene suficientes canales (RGB)\n");
        stbi_image_free(h_imageData);
        return -1;
    }

    // Convertir la imagen a una matriz 2D (escala de grises)
    std::vector<std::vector<unsigned char>> grayImage(height, std::vector<unsigned char>(width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = (y * width + x) * channels;
            unsigned char r = h_imageData[index];
            unsigned char g = h_imageData[index + 1];
            unsigned char b = h_imageData[index + 2];
            // Convertir a escala de grises
            grayImage[y][x] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }

    // Parámetros del filtro de Gabor
    double sigma = 1.0;
    double theta = PI / 4;  // 45 grados
    double lambda = 10.0;   // Longitud de onda
    double gamma = 0.5;     // Relación de aspecto
    double psi = 0;         // Fase
    int kernel_size = 21;   // Tamaño del kernel

    // Crear el kernel de Gabor
    std::vector<std::vector<double>> gaborKernel = createGaborKernel(kernel_size, sigma, theta, lambda, gamma, psi);

    // Medir el tiempo de ejecución del filtro Gabor
    auto start_gabor = std::chrono::high_resolution_clock::now();

    // Aplicar la convolución
    std::vector<std::vector<double>> gaborResult = applyConvolution(grayImage, gaborKernel, width, height, kernel_size);

    auto end_gabor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gabor_time = end_gabor - start_gabor;
    std::cout << "Tiempo de ejecución del filtro Gabor: " << gabor_time.count() << " segundos." << std::endl;

    // Crear el kernel Emboss
    std::vector<std::vector<double>> embossKernel = createEmbossKernel();

    // Medir el tiempo de ejecución del filtro Emboss
    auto start_emboss = std::chrono::high_resolution_clock::now();

    // Aplicar la convolución de Emboss
    std::vector<std::vector<double>> embossResult = applyConvolution(grayImage, embossKernel, width, height, 3); // Kernel de 3x3

    auto end_emboss = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> emboss_time = end_emboss - start_emboss;
    std::cout << "Tiempo de ejecución del filtro Emboss: " << emboss_time.count() << " segundos." << std::endl;

    // Aplicar el filtro Diferencia de Gaussiana (DoG)
    double sigma1 = 1.0;
    double sigma2 = 2.0; // Dos sigmas diferentes para la diferencia
    kernel_size = 21;    // Tamaño del kernel Gaussiano

    // Crear los kernels gaussianos
    std::vector<std::vector<double>> gaussKernel1 = createGaussianKernel(kernel_size, sigma1);
    std::vector<std::vector<double>> gaussKernel2 = createGaussianKernel(kernel_size, sigma2);

    auto start_dog = std::chrono::high_resolution_clock::now();

    // Aplicar convolución usando la función ya existente
    std::vector<std::vector<double>> result1 = applyConvolution(grayImage, gaussKernel1, width, height, kernel_size);
    std::vector<std::vector<double>> result2 = applyConvolution(grayImage, gaussKernel2, width, height, kernel_size);

    // Restar las dos convoluciones (DoG)
    std::vector<std::vector<double>> dogResult(height, std::vector<double>(width, 0));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            dogResult[y][x] = result1[y][x] - result2[y][x];
        }
    }

    auto end_dog = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dog_time = end_dog - start_dog;
    std::cout << "Tiempo de ejecución del filtro Diferencia de Gaussiana (DoG): " << dog_time.count() << " segundos." << std::endl;

    // Guardar las imágenes de salida para cada filtro (Gabor, Emboss y DoG)
    auto saveImage = [&](const std::vector<std::vector<double>>& result, const char* filename) {
        std::vector<unsigned char> outputImage(height * width);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double value = result[y][x];
                value = std::max(0.0, std::min(255.0, value));
                outputImage[y * width + x] = static_cast<unsigned char>(value);
            }
        } 
        stbi_write_jpg(filename, width, height, 1, outputImage.data(), 100); // 100 es la calidad del JPEG
        };

    saveImage(gaborResult, "resultado_gabor.jpg");
    saveImage(embossResult, "resultado_emboss.jpg");
    saveImage(dogResult, "resultado_dog.jpg");

    // Liberar la memoria asignada por stb_image
    stbi_image_free(h_imageData);

    return 0;
}