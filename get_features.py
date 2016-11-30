import operator
import sys, os
sys.path.append("../..")
# import facerec modules
#from facerec.feature import PCA as pca
#from facerec.feature import Fisherfaces
#from facerec.feature import SpatialHistogram as sh
#from facerec.distance import EuclideanDistance
#from facerec.classifier import NearestNeighbor
#from facerec.model import PredictableModel
#from facerec.validation import KFoldCrossValidation
#from facerec.visual import subplot
#from facerec.util import minmax_normalize
# import numpy, matplotlib and logging
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import logging

import os
import pickle
import json
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
#import src_temp as src
def read_images(subject_path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    train_images_names = sorted(os.listdir(subject_path))
    label_list = [int(name.split('_')[0]) for name in train_images_names]
    for filename in os.listdir(subject_path):
	try:
	    im = Image.open(os.path.join(subject_path, filename))
	    im = im.convert("L")
	    # resize to given size (if given)
	    if (sz is not None):
		im = im.resize(sz, Image.ANTIALIAS)
	    X.append(np.asarray(im, dtype=np.uint8))
	    y.append(c)
	except IOError, (errno, strerror):
	    print "I/O error({0}): {1}".format(errno, strerror)
	except:
	    print "Unexpected error:", sys.exc_info()[0]
	    raise
    c = c+1
    return X,label_list

    


def get_data(folder):
    train_images_names = sorted(os.listdir(folder))
    label_list = [name.split('_')[0] for name in train_images_names]
    unique_faces = sorted(list(set(label_list)))
    name_number_map = { name:unique_faces.index(name) for name in unique_faces }
    
    
    train_images_names_for_every_face_id = []
    for face_id in unique_faces:
        locations_of_current_face_id = [i for i,x in enumerate(label_list) if x == face_id]
        train_image_names_for_current_face_id = [train_images_names[location] for location in locations_of_current_face_id]
        train_images_names_for_every_face_id.append(train_image_names_for_current_face_id)
    
    #sub_val =map(operator.sub,[int(i) for i in h_ver.tolist()],train_images_names_for_every_face_id[-1])
    flag_for_dummy_creation = 0
    new_class_id_flag = 0
    class_id = []
    #dummy=cv2.imread('inp_train/1_01_subject01happy.jpg',0)
    col = 256 #dummy.shape[1]
    row = 256 #dummy.shape[0]
    all_images_as_vec = np.empty((row*col,0), dtype=object)
    for image_list_per_class in train_images_names_for_every_face_id:
        class_id.append(new_class_id_flag)
        for image_name in image_list_per_class:
            current_image = cv2.imread(folder+'/'+image_name,0)
            edge = cv2.Canny(current_image,threshold1= 200, threshold2=300)
            
           
            
            current_image_as_vec = edge.reshape(col*row,1)
            #print all_images_as_vec.shape
            #print current_image_as_vec.shape
            all_images_as_vec = np.hstack ((all_images_as_vec,current_image_as_vec))
            new_class_id_flag = new_class_id_flag + 1

    return all_images_as_vec,label_list

def main():
    
    mode = 'inp_'+sys.argv[1]
    train_data,y = get_data(mode)
    #test_data,test_label = get_data('inp_test')
    
    np.savetxt(sys.argv[1]+'.csv', train_data , delimiter=',')
    #np.savetxt('test.csv', test_data,delimiter=',')
    with open(sys.argv[1]+'_labels', 'w') as outfile:
        json.dump(y, outfile)
    #with open('test_labels.txt', 'w') as outfile:
        #json.dump(test_label, outfile)
    '''
    print train_data[0].shape
    pca_obj = pca()
    eigen_face = pca_obj.compute(train_data[0],y)
    print len(eigen_face)
    ''' 
    
    '''
    fisher_feature = Fisherfaces()
    transform = fisher_feature.compute(train_data[0],y,1,[])
    rows = transform[0].shape[0]
    train_matrix = np.empty([rows, 0], dtype=float)
    
    for i in transform:
        i = i.astype('float')
        train_matrix = np.hstack((train_matrix,i))
     
    print train_matrix.shape
    test_data,test_label = read_images('test')
    
    transform_test = fisher_feature.compute(train_data[0],y,0,test_data[0]) 
    row_test = transform_test[0].shape[0]
    test_matrix = np.empty([row_test, 0], dtype=float)
    print len(transform_test)
    for i in transform_test:
        i = i.astype('float')
        test_matrix = np.hstack((test_matrix,i))
    print test_matrix.shape
    np.savetxt('train.csv', train_matrix , delimiter=',')
    np.savetxt('test.csv', test_matrix,delimiter=',')
    with open('train_labels.txt', 'w') as outfile:
        json.dump(y, outfile)
    with open('test_labels.txt', 'w') as outfile:
        json.dump(test_label, outfile)    
    '''
    '''
    train_matrix,train_label = get_data('train')
    print train_matrix.shape
    print train_label
    print len(train_label)
    #isomap_obj = Isomap(n_neighbors=35, n_components=1000)
    #projected_train_matrix = isomap_obj.fit_transform(train_matrix)
    pca = PCA(n_components=100)
    proj_obj = pca.fit(train_matrix)
    projected_train_matrix= proj_obj.transform(train_matrix) 
    print projected_train_matrix.shape
    
    test_matrix,test_label = get_data('test')
    print test_matrix.shape
    #proj_obj = isomap_obj.fit_transform(test_matrix)
    #projected_test_matrix = proj_obj.transform(test_matrix) 
    #print projected_test_matrix.shape
    np.savetxt('train.csv', train_matrix.T , delimiter=',')
    np.savetxt('test.csv', test_matrix.T,delimiter=',')
    with open('train_labels.txt', 'w') as outfile:
        json.dump(train_label, outfile)
    with open('test_labels.txt', 'w') as outfile:
        json.dump(test_label, outfile)
    #src.controller(train_matrix,test_matrix,labels)
    '''
main()


