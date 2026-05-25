#compute eigen values and eigen vectors of a matrix
from cv2 import eigen
import numpy as np
def eigen_mat_transform(A):
    #transform the matrix and compute the eigenvalues and eigenvectors using numpy
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    return eigenvalues, eigenvectors

def eigen_mat_scale(A, scale):
    #scale the matrix and compute the eigenvalues and eigenvectors using numpy
    A_scaled = A * scale
    eigenvalues, eigenvectors = np.linalg.eig(A_scaled)
    return eigenvalues, eigenvectors    

if __name__ == "__main__":
    #create matrix using numpy 2,2 
    image_matrix = np.array([[1,2],[3,4]])
    print("Image Matrix: \n", image_matrix)
    eigenvalues, eigenvectors = eigen_mat_transform(image_matrix)
    print("Eigenvalues: \n", eigenvalues)
    print("Eigenvectors: \n", eigenvectors)
    scale = 2
    eigenvalues_scaled, eigenvectors_scaled = eigen_mat_scale(image_matrix, scale)
    print("Scaled Eigenvalues: \n", eigenvalues_scaled)
    print("Scaled Eigenvectors: \n", eigenvectors_scaled)