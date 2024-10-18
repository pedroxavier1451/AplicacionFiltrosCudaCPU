#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // Para cargar imágenes PNG o JPEG
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // Para guardar imágenes
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono> // Para medir el tiempo de ejecución

#define PI 3.14159265358979323846

// Función para crear un kernel de Emboss en formato unidimensional
std::vector<double> createEmbossKernel(int kernel_size) {
    std::vector<double> kernel(kernel_size * kernel_size, 0);  // Inicializar con ceros
    int half_size = kernel_size / 2;

    // Crear el efecto de Emboss con un desplazamiento diagonal
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            if (x < half_size && y < half_size) {
                kernel[y * kernel_size + x] = -1;  // Parte negativa
            }
            else if (x > half_size && y > half_size) {
                kernel[y * kernel_size + x] = 1;   // Parte positiva
            }
            else if (x == half_size && y == half_size) {
                kernel[y * kernel_size + x] = 1;   // Valor central
            }
        }
    }

    return kernel;
}

// Función para crear el kernel de Gabor en formato unidimensional
std::vector<double> createGaborKernel(int kernel_size, double sigma, double theta, double lambda, double gamma, double psi) {
    std::vector<double> kernel(kernel_size * kernel_size);  // Unidimensional
    int half_size = kernel_size / 2;

    for (int y = -half_size; y <= half_size; ++y) {
        for (int x = -half_size; x <= half_size; ++x) {
            double x_theta = x * cos(theta) + y * sin(theta);
            double y_theta = -x * sin(theta) + y * cos(theta);
            double gauss = exp(-(x_theta * x_theta + gamma * gamma * y_theta * y_theta) / (2 * sigma * sigma));
            double sinusoid = cos(2 * PI * x_theta / lambda + psi);
            kernel[(y + half_size) * kernel_size + (x + half_size)] = gauss * sinusoid;
        }
    }

    return kernel;
}

// Función para crear el kernel High-Boost en formato unidimensional
std::vector<double> createHighBoostKernel(int kernel_size, double A) {
    std::vector<double> kernel(kernel_size * kernel_size, -1);  // Inicializar con -1 (unidimensional)
    int half_size = kernel_size / 2;
    kernel[half_size * kernel_size + half_size] = A + (kernel_size * kernel_size) - 1;
    return kernel;
}

// Función para aplicar la convolución de manera secuencial en la CPU
std::vector<double> applyConvolutionCPU(const std::vector<unsigned char>& image, const std::vector<double>& kernel, int width, int height, int kernel_size) {
    int half_kernel = kernel_size / 2;
    std::vector<double> result(width * height, 0);

    for (int y = half_kernel; y < height - half_kernel; ++y) {
        for (int x = half_kernel; x < width - half_kernel; ++x) {
            double sum = 0.0;
            for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
                for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
                    int pixel_value = image[(y + ky) * width + (x + kx)];
                    sum += pixel_value * kernel[(ky + half_kernel) * kernel_size + (kx + half_kernel)];
                }
            }
            result[y * width + x] = sum;
        }
    }
    return result;
}


int main()
{
    // Variables para la imagen
    int width, height, channels;

    // Cargar la imagen usando stb_image
    unsigned char* h_imageData = stbi_load("imagen4.jpg", &width, &height, &channels, 0);
    if (!h_imageData) {
        printf("Error al cargar la imagen\n");
        return -1;
    }

    if (channels < 3) {
        printf("La imagen no tiene suficientes canales (RGB)\n");
        stbi_image_free(h_imageData);
        return -1;
    }

    // Convertir la imagen a escala de grises
    std::vector<unsigned char> grayImage(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = (y * width + x) * channels;
            unsigned char r = h_imageData[index];
            unsigned char g = h_imageData[index + 1];
            unsigned char b = h_imageData[index + 2];
            grayImage[y * width + x] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }

    int kernel_size = 21;  // Tamaño del kernel

    // Parámetros del filtro de Gabor
    double sigma = 4.0;
    double theta = 0;  // 90 grados
    double lambda = 10.0;   // Longitud de onda
    double gamma = 0.5;     // Relación de aspecto
    double psi = 0;         // Fase

    // Parámetros del filtro High-Boost
    double A = 10.0;  // Factor de realce ajustable

    // Crear el kernel de Gabor
    // Crear el kernel de Emboss
    std::vector<double> embossKernel = createEmbossKernel(kernel_size);
    std::vector<double> gaborKernel = createGaborKernel(kernel_size, sigma, theta, lambda, gamma, psi);
    std::vector<double> highBoostKernel = createHighBoostKernel(kernel_size, A);

    // Medir el tiempo de ejecución del filtro Emboss
    auto start_emboss = std::chrono::high_resolution_clock::now();

    // Aplicar la convolución usando el kernel Emboss
    std::vector<double> result_emboss = applyConvolutionCPU(grayImage, embossKernel, width, height, kernel_size);

    auto end_emboss = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> emboss_time = end_emboss - start_emboss;
    std::cout << "Tiempo de ejecución del filtro Emboss: " << emboss_time.count() << " segundos." << std::endl;

    // Medir el tiempo de ejecución del filtro Gabor
    auto start_gabor = std::chrono::high_resolution_clock::now();

    // Aplicar la convolución
    std::vector<double> result_gabor = applyConvolutionCPU(grayImage, gaborKernel, width, height, kernel_size);

    auto end_gabor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gabor_time = end_gabor - start_gabor;
    std::cout << "Tiempo de ejecución del filtro Gabor: " << gabor_time.count() << " segundos." << std::endl;

    // Medir el tiempo de ejecución del filtro High-Boost
    auto start_highboost = std::chrono::high_resolution_clock::now();

    // Aplicar la convolución usando el kernel High-Boost
    std::vector<double> result_highboost = applyConvolutionCPU(grayImage, highBoostKernel, width, height, kernel_size);

    auto end_highboost = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> highboost_time = end_highboost - start_highboost;
    std::cout << "Tiempo de ejecución del filtro High-Boost: " << highboost_time.count() << " segundos." << std::endl;

    // Guardar las imágenes de salida para cada filtro (Gabor, High-Boost y DoG)
    auto saveImage = [&](const std::vector<double>& result, const char* filename) {
        std::vector<unsigned char> outputImage(width * height);
        double min_val = *std::min_element(result.begin(), result.end());
        double max_val = *std::max_element(result.begin(), result.end());
        for (int i = 0; i < width * height; ++i) {
            double normalized_value = 255.0 * (result[i] - min_val) / (max_val - min_val);
            outputImage[i] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, normalized_value)));
        }
        stbi_write_jpg(filename, width, height, 1, outputImage.data(), 100);
    };

    saveImage(result_emboss, "resultado_emboss_cpu.jpg");
    saveImage(result_gabor, "resultado_gabor_cpu.jpg");
    saveImage(result_highboost, "resultado_highboost_cpu.jpg");
    
    // Liberar la memoria asignada por stb_image
    stbi_image_free(h_imageData);

    return 0;
}