#inverse image matrix using logo.webp
import numpy as np
import cv2
from linearalgebra.configurations.conf import Config
def show_image_matrix(image_path):
    #read the image using opencv    
    img = cv2.imread(image_path)
    #convert the image to matrix
    img_matrix = np.array(img)
    print("Original Image Matrix:")
    print(img_matrix.shape)
    #display the image
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def inverse_image(image_path):
    img=cv2.imread(image_path)

    # Resize image
    img = cv2.resize(img, (256, 256))

    # Convert color image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Original Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Convert to float
    img_matrix = gray.astype(np.float64)

    # Add small value to diagonal to avoid singular matrix
    img_matrix = img_matrix + np.eye(256) * 0.001

    # Inverse matrix
    inv_matrix = np.linalg.inv(img_matrix)
    print("Inverse Image Matrix:")
    print(inv_matrix.shape)
    #display the image
    cv2.imshow("Original Image", inv_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    show_image_matrix(Config.image_path)
    inverse_image(Config.image_path)