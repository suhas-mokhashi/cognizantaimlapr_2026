#apply l2 normalization to the cyber security text file
import numpy as np
from linearalgebra.configurations.conf import Config
def l2_normalize(file_path):
    with open(file_path, 'r',encoding='utf-8') as f:
        lines = f.read()
    #create the documents by splitting the lines using \n
    documents=[p.strip() for p in lines.split('\n') if p.strip()]
    print("Number of documents:", len(documents))
    #apply l2 normalization to the documents
    normalized_documents = []
    for doc in documents:
        #split the document into words
        words = doc.split()
        #calculate the l2 norm of the document
        l2_norm = np.sqrt(sum([len(word)**2 for word in words]))
        #normalize the document by dividing each word by the l2 norm
        normalized_doc = [len(word)/l2_norm for word in words]
        normalized_documents.append(normalized_doc)
    print("Normalized documents:")
    for doc in documents:
        print(doc)
        print(f"Normalized: {normalized_documents[documents.index(doc)]}\n")
if __name__ == "__main__":
    file_path = Config.cyber_path
    l2_normalize(file_path)