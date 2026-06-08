#create pca for tshirt data in csv file
#consider target columns QaulityClass, Rest all inputs

import pandas as pd
from sklearn.decomposition import PCA   
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from supervisedmlapp.configurations.conf import T_SHIRT_FILE_PATH
from sklearn.preprocessing import LabelEncoder
def pca_analysis():
    # Load the dataset
    data = pd.read_csv(T_SHIRT_FILE_PATH)

    #dataset information
    print("Dataset Information:")
    print(data.info())


    #apply label encoding to the target variable
   
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(data['QualityClass'])

    #label encoding to defects, size and stitiching
    data['Defects'] = label_encoder.fit_transform(data['Defects'])
    data['Size'] = label_encoder.fit_transform(data['Size'])
    data['Stitching'] = label_encoder.fit_transform(data['Stitching'])

    #apply one hot encoding to color,fabric,brand
    data = pd.get_dummies(data, columns=['Color', 'Fabric', 'Brand'])   
  
   
    # Separate features and target variable
    X = data.drop('QualityClass', axis=1)  # Features
    
    #print x and y
    print("Features (X):")
    print(X.head())
    print("\nTarget variable (y):")
    print(y[:5])

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2 principal components for visualization
    X_pca = pca.fit_transform(X_scaled) 
    #print principal components
    #computing correlated values
    print("Principal Components:")
    print(X_pca[:5])
    #need how to arrive at the principal components from the original features    

    print("PCA Components column name and values:") 
    for i, component in enumerate(pca.components_):
        print(f"Principal Component {i+1}:")
        for feature, value in zip(X.columns, component):
            print(f"  {feature}: {value}") 




   


if __name__ == "__main__":
    pca_analysis()