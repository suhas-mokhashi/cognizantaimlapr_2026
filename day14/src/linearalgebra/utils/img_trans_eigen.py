#create eigen image transformation matrix
import numpy as np
import cv2
from linearalgebra.configurations.conf import Config
def create_eigen_image_transformation_matrix(img_path):
    # Load the image and convert it to grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #display the image
    cv2.imshow('Original Image', img)
    # add wait key to display the image
    cv2.waitKey(0)
    #resize the image to 256x256
    #eigen require square matrix
    resized_img = cv2.resize(img, (256, 256))
    #display the image
    cv2.imshow('Resized Image', resized_img)
    # add wait key to display the image
    cv2.waitKey(0)
    #convert the image to float32
    resized_img = resized_img.astype(np.float32)
    print("Resized Image: ", resized_img)
    #calculate the mean of the image
    mean = np.mean(resized_img,axis=0)
    print("Mean of the image: ", mean)
    #center the image by subtracting the mean
    #translate the image to the origin
    centered_img = resized_img - mean
    print("Centered Image: ", centered_img)
    #display the image
    cv2.imshow('Centered Image', centered_img)
    # add wait key to display the image
    cv2.waitKey(0)
    #calculate the covariance matrix
    cov_matrix = np.cov(centered_img, rowvar=False)
    print("Covariance Matrix: ", cov_matrix)
    #calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    print("Eigenvalues: ", eigenvalues) 
    print("Eigenvectors: ", eigenvectors)
    #sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    print("Sorted Eigenvalues: ", sorted_eigenvalues)
    print("Sorted Eigenvectors: ", sorted_eigenvectors)


if __name__ == "__main__":
    config = Config()
    img_path = config.image_path
    create_eigen_image_transformation_matrix(img_path)
    