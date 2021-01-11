import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as compiler
import numpy as np
import math
import sys
import timeit
import cv2
from PIL import Image

try:
    input_image = str(sys.argv[1])
    output_image = str(sys.argv[2])
    numSigma = str(sys.argv[3])
except IndexError:
    sys.exit("No input/output image")

try:
    #img = Image.open(input_image)
    img = cv2.imread(input_image)    
    imagenGris=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#convirtiendo en scala de gris
    input_array = np.array(imagenGris)
    #Image.fromarray(input_array).save(output_image)
    oneCanal = input_array[:, :].copy()#copiamos todas las columnas y filas de la imagen 
except FileNotFoundError:
    sys.exit("Cannot load image file")


# ######################################## #
# generate gaussian kernel (size of N * N) #
# ######################################## #
sigma = int(numSigma)  # standard deviation of the distribution
kernel_width = int(3 * sigma)
if kernel_width % 2 == 0: #En caso de obtener un kernel par ajustarlo a impar
    kernel_width = kernel_width - 1  # make sure kernel width only sth 3,5,7 etc

# create empty matrix for the gaussian kernel #
kernel_matrix = np.empty((kernel_width, kernel_width), np.float32)
kernel_half_width = kernel_width // 2
for i in range(-kernel_half_width, kernel_half_width + 1):
    for j in range(-kernel_half_width, kernel_half_width + 1):
        kernel_matrix[i + kernel_half_width][j + kernel_half_width] = (
                np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
                / (2 * np.pi * sigma ** 2)
        )
gaussian_kernel = kernel_matrix / kernel_matrix.sum()


# #################################################################### #
#calculo hilos bloques y grids segun la imagen ingresada, parametros enviados a CUDA
# #################################################################### #
height, width = input_array.shape[:2]
dim_block = 32
dim_grid_x = math.ceil(width / dim_block)
dim_grid_y = math.ceil(height / dim_block)

# load CUDA code
mod = compiler.SourceModule(open('drive/MyDrive/ComputoParalelo/practica/gaussian_blur.cu').read())
apply_filter = mod.get_function('applyFilter')
#tiempos de ejecución
time_started = timeit.default_timer()
apply_filter(
        drv.In(oneCanal),
        drv.Out(oneCanal),
        np.uint32(width),
        np.uint32(height),
        drv.In(gaussian_kernel),
        np.uint32(kernel_width),
        block=(dim_block, dim_block, 1),
        grid=(dim_grid_x, dim_grid_y))
time_ended = timeit.default_timer()

output_array = np.empty_like(input_array)
output_array[:,:] = oneCanal
Image.fromarray(output_array).save(output_image[:-4]+numSigma+output_image[len(output_image)-4:])
print('Tiempo Ejecución: ', time_ended - time_started, 's')

print('Proceso Terminado...')