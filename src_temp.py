from scipy.sparse.linalg import lsmr
from numpy.linalg import norm
import numpy as np
import pickle
import json
import os
def get_residual_per_class(class_train_matrix, test_vector):

    solution_vector = lsmr(class_train_matrix, test_vector, damp=10, atol=1.0e-12, btol=1.0e-12, conlim=1.0e+8, show=False)
    #norm_of_residual = solution_vector[]
    Ax = np.dot(class_train_matrix,solution_vector[0])
    norm_of_residual = norm(Ax-test_vector)
    return norm_of_residual

def normalize_data(data_matrix):
    norm_data_as_list = []
    t_data  = data_matrix.T
    for col in t_data:
        temp_list = [ele/norm(col) for ele in col]
        norm_data_as_list.append(temp_list)
    norm_mat = np.asarray(norm_data_as_list)
    required_normalized_matrix = norm_mat.T
    return required_normalized_matrix

def lsmr_sparse_representaion_classifier(train_matrix, test_vector,labels):
    unique_labels = sorted(list(set(labels)))
    residual_per_class = list()
    for class_label in unique_labels:
        required_train_point_indices = [i for i, x in enumerate(labels) if x == class_label]
        required_train_matrix = train_matrix[:, required_train_point_indices]
        residual_per_class.append(get_residual_per_class(required_train_matrix,test_vector))
    predicted_label = unique_labels[residual_per_class.index(min(residual_per_class))]
    return predicted_label,residual_per_class

def controller(train_matrix, test_matrix, labels):
    #train_matrix = np.genfromtxt('tamocr_train.csv',delimiter = ",")
    #test_matrix = np.genfromtxt('tamocr_test.csv',delimiter = ",")
    #labels = [j for j in range(15) for i in range(8)]
    normalized_train_matrix = train_matrix #normalize_data(train_matrix)
    normalized_test_matrix = test_matrix #norbmalize_data(test_matrix)
    print(normalized_test_matrix.shape)
    print(normalized_test_matrix.T.shape)
    '''
    print(normalized_train_matrix.shape)
    print(normalized_test_matrix.shape)
    print(normalized_test_matrix.T.shape)
    print(normalized_test_matrix.T[0].shape)
    
    trns_mat = normalized_test_matrix.T[0]
    mat = trns_mat.T
    print(mat.shape)
    '''
    predicted_class=[]
    if len(normalized_test_matrix.shape) == 1:
        predicted_label, residual =lsmr_sparse_representaion_classifier( normalized_train_matrix, normalized_test_matrix, labels )
        predicted_class.append(predicted_label)
        print(residual)
    else:
        for test_vector in normalized_test_matrix.T:
            predicted_label, residual =lsmr_sparse_representaion_classifier( normalized_train_matrix, test_vector.T, labels )
            predicted_class.append(predicted_label)
            print(residual)
    print("yhe label is ")
    print(predicted_class)
    return predicted_class

def main():
    train_matrix = np.genfromtxt('train.csv',delimiter = ",")
    test_matrix = np.genfromtxt('test.csv',delimiter = ",")
    json_data=open('train_labels').read()
    train_labels = json.loads(json_data)
    json_data=open('test_labels').read()
    test_labels = json.loads(json_data)
    result = controller(train_matrix, test_matrix, train_labels)
    matches = [i for i, j in zip(result, test_labels) if i == j]
    print(result)
    print(test_labels)
    print(len(test_labels))
    print(len(matches))
if __name__ == '__main__':     
    main()
    
