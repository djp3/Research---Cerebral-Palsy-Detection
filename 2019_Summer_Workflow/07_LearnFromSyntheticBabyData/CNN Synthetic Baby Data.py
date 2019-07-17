#!/usr/bin/env python
# coding: utf-8


#djp3 - This is automatically generated from the Jupyter notebook, so don't
# make permanent changes here do it in the notebook with the same name

# In[19]:


#Import from the Keras library
from keras import models
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,  MaxPooling2D
from keras import optimizers 
from keras import utils
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
#from secret import credentials

#This allows for Keras models to be saved. 
import h5py
#Other import statements 
import h5py
import random
import numpy as np
import pandas as pd
import pathlib
import cv2
import pymysql
import os

#limit the data to a subset for testing
#Make empty string to have no limit
limit = "ORDER BY RAND() LIMIT 3000"
#limit = "ORDER BY RAND() LIMIT 7000"

epochs = 1
batch_size = 32

#directory where data from database is stored
cache_path = 'cache'

#File name for the statistics to be save in. Must include .txt at the end
statistics_output_file = 'statistics.output.txt'

#Must have the h5py package installed or the model will not save. This should be the path of the location you would like
#To save the model
model_file_name = 'model.output'


# In[20]:


#Secrets shouldn't be in the repository
from secrets import credentials
# Of the form
#credentials = {
#        'db_host' : 'something.us-east-1.rds.amazonaws.com'
#        'db_port' : 3306
#        'db_name' : 'name',
#        'db_username' : 'something',
#        'db_password' : 'secret'
#        }


def connect(): 
    db_host = credentials['db_host'];
    db_port = credentials['db_port'];
    db_name = credentials['db_name'];
    db_username = credentials['db_username']
    db_password = credentials['db_password']
    
    conn = pymysql.connect(db_host, user=db_username, port=db_port, passwd=db_password, db=db_name)
    return conn


# In[21]:


def import_data(cache_path, conn): 

    #Create the cache directory if it doesn't exist
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        
    cursor = conn.cursor()

    try:
        recording_ids = {}
        xy = {}    
        
        image_query = "SELECT r.id, r.recording_id, r.isCSGM FROM nicu.Video_Raw AS r JOIN nicu.Video_Generated AS g ON r.id=g.raw_id  WHERE (r.recording_id>1) AND (g.RGB_Optical_Flow IS NOT NULL) " +limit
        try:
            cursor.execute(image_query) #(list(recording_ids.keys())))
            for row in cursor.fetchall():
                raw_id = row[0]
                rec_id = row[1]
                csgm = row[2]
                if rec_id in recording_ids:
                    recording_ids.get(rec_id).append(raw_id)
                else:
                    recording_ids.update({rec_id:[raw_id]})
                xy.update({raw_id:[csgm]})
        except Exception as e:
            print("Error retrieving ID's", e)
            conn.rollback()
            raise e
            
        print("Collecting images for processing (o = source image in cache, ⇣ = source image fetched from db, x = source image not in db)")
        for rec_id in recording_ids:
            print("")
            print("Analyzing recording_id:",rec_id,": ",end="")
            raw_id_list = recording_ids.get(rec_id)
            for raw_id in raw_id_list:
                current_input = cache_path+'/'+str(raw_id)+".oflow.png"
                if not os.path.exists(current_input):
                    cursor2 = conn.cursor()
                    try:
                        image_query = "SELECT RGB_Optical_Flow from Video_Generated WHERE (raw_id=%s)"
                        cursor2.execute(image_query, (str(raw_id)))
                        for row in cursor2.fetchall():
                            db_img = row[0]
                            if db_img is not None:
                                img=cv2.imdecode(np.asarray(bytearray(db_img),dtype=np.uint8),cv2.IMREAD_UNCHANGED)
                                cv2.imwrite(current_input,img)
                                print("⇣",end="",flush=True)
                            else:
                                print("x",end="",flush=True)
                    except Exception as e:
                        print("Error retrieving Optical Flow frame",e)
                        raise e
                    finally:
                        cursor2.close()     
                else:
                    print("o",end="",flush=True)

                #Resizing the image
                img = cv2.imread(current_input)
                scale_percent = 1
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                cv2.resize(img,(width,height), interpolation=cv2.INTER_CUBIC)
                xy.get(raw_id).insert(0,img)
        print("")
        return recording_ids, xy
    finally:
        cursor.close()
    


# In[22]:


def create_array(raw_ids, xy):    
    image_list=[]
    csgm_list=[]
    
    random.shuffle(raw_ids)
    
    for i in raw_ids:
        #if not xy.get(i)[0] == None:
        image_list.append(xy.get(i)[0])
        csgm_list.append(xy.get(i)[1])
    x = np.array(image_list)
    y = np.array(csgm_list)
    return x, y
        


# In[23]:


def create_cnn(x_train, filter_info={0:[32,3]}, dropout={0:0.25}, pooling={0:2}, activation='relu', loss='mean_squared_error', final_activation='sigmoid'):    
    
    model = models.Sequential() 
    str_model = "Overview of Model Architecture: /n"
    
    filter_size = 0
    
    for i in filter_info: 
        filter_size = filter_info.get(i)[1]
        num_filters = filter_info.get(i)[0]
        
        if i == 0: 
            model.add(Conv2D(num_filters, (filter_size,filter_size), activation = 'relu', input_shape=x_train.shape[1:]))
        else: 
            model.add(Conv2D(num_filters, (filter_size,filter_size), activation= 'relu'))
        
        str_model += ("2D Convolution Layer with %d filters the size of (%d,%d) and %s activation \n" %(num_filters, filter_size, filter_size, activation))
        
        model.add(Conv2D(num_filters, (filter_size,filter_size), activation= 'relu'))
        str_model += ("2D Convolution Layer with %d filters the size of (%d,%d) and %s activation \n" %(num_filters, filter_size, filter_size, activation))
        
        if i in pooling:           
            pool_filter_size = pooling.get(i)
            model.add(MaxPooling2D(pool_size=(pool_filter_size, pool_filter_size)))
            str_model += ('2D Pooling Max Pooling Layer with filter size (%d,%d)\n' %(pool_filter_size,pool_filter_size))
            
                 
        if i in dropout: 
            drop_rate = dropout.get(i)
            model.add(Dropout(drop_rate))
            str_model += ('Droput Layer with with a rate of %f \n' %(drop_rate))


    
    #These will be added to the end of every model no matter what
    model.add(Flatten())
    str_model += ('Flatten\n')
    model.add(Dense(256,activation=activation))
    str_model += ('Dense layer with %s activation\n' %(activation))
    model.add(Dropout(0.5))
    str_model += ('Droput Layer with with a rate of 0.5 \n')
    
    #Sigmoid activiation is employed in the final step because the output is binary. 
    model.add(Dense(1, activation=final_activation)) 
    str_model += ('Dense layer with %s activation\n' %(final_activation))
    
    model.compile(loss=loss, 
              optimizer=optimizers.SGD(lr=1e-4),
              metrics=['acc']) 
    str_model += ('Loss: %s' %(loss))
                      
    print(str_model)

    return model, str_model

                    


# In[24]:


def confusion_matrix(exp_values, predicted_values):
    """
    This creates a confusion matrix with the predicted accuracy of the model.
    
    exp_values must be in the format of a list and predicted values is expected to come in the format of the ouput 
    of Keras's model.predict()
    
    The ouput is a pandas dataframe that displays a confusion matrix indicitive of the accuracy of the model along 
        with a number score which is the accuracy of the model.
    """
    predicted_values = convert_predictions(predicted_values)
    
    
    
    #Creates a DataFrame of zeros
    matrix = pd.DataFrame(np.zeros((2,2)) , ['P0','P1'], ['E0','E1'])
   
    #Caculates whether the score was right or wrong and updates the confusion matrix 
    for i in range(len(exp_values)):
        if exp_values[i] == predicted_values[i]:
            matrix.iloc[[predicted_values[i]],[predicted_values[i]]] += 1
        else:
            matrix.iloc[[predicted_values[i]],[exp_values[i]]] += 1
   
    #Calculate diagonal sum and the accuracy of the model
    #Precision (TP/TP+FPos)      Recall TP(TP+FNegative)
    diagonal_sum = 0
    for i in range(2):
        diagonal_sum += matrix.iloc[i][i]
    
    score = diagonal_sum/len(exp_values)
    
  
    return  matrix, score
    
    
            


# In[25]:


def convert_predictions(predictions): 
    """
    Converts predictions outputted by a keras model into a list with 1 represented the predicted output and zero 
    in other classes. 
    """
    l =[]
    for p in predictions: 
        if p >= 0.5:
            l.append(1)
        else:
            l.append(0)
    return l


# In[26]:


def runTest(pooling, dropout, filter_info, loss, activation, final_activation, file_name='model.txt', model_name='model', save_model=False, epochs=5, batch_size=32):
    
    conn = connect()
    try:
        recording_ids_dict, xy = import_data(cache_path,conn)
    finally:
        conn.close()

    matrices = {}
    scores = {}
    model_scores = {}
    str_model =''

    for i in recording_ids_dict:
        print('Testing on ' + str(i))
        train_ids= []
        test_ids = []
        
        for j in recording_ids_dict:
            if j == i:
                test_ids = recording_ids_dict[j]
            else: 
                train_ids.extend(list(recording_ids_dict[j]))
        
        x_train, y_train = create_array(train_ids, xy)
        x_test, y_test = create_array(test_ids, xy)
        
        #Scaling the values to a value between 0 and 1
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        
        
        model,str_model = create_cnn(x_train,
                                     filter_info=filter_info,
                                     dropout=dropout,
                                     pooling=pooling,
                                     loss=loss,
                                     final_activation=final_activation,
                                     activation=activation)
        
        #Fit the model
        model.fit(x_train, y_train, epochs = epochs)
        
        #Create predictions and evaluate to find loss and accuaracy
        predict = model.predict(x_test)
        model_score = model.evaluate(x_test, y_test)
        print('Model was ' + str(model_score[1]) + '% accurate and exhibited an average loss of ' + str(model_score[0]) + '.')
        
        matrix,score = confusion_matrix(y_test, predict)
        
        matrices.update({i : matrix})
        print(str(matrix) + '\n')
        scores.update({i: score})
        print(str(score) + '\n')
        model_scores.update({i:model_score})
   
    with open(file_name, 'w') as f:
        for key in matrices:
            f.write("Baby %s\n" % key)
            f.write("%s\n" % str_model)
            f.write("%s\n" % matrices[key])
            f.write("%s\n" % scores[key])
            f.write("%s\n" % model_scores[key])
        
            
    if save_model : 
        model.save(model)
    #Add a final matrix 


# In[27]:


filter_info={0:[32,3],1:[64,3],2:[128,3]}
dropout={0:0.25,1:0.25,2:0.25}
pooling={0:2,0:2,0:2}


runTest(file_name=statistics_output_file, 
        filter_info=filter_info, 
        dropout=dropout, 
        pooling=pooling, 
        loss='mean_squared_error', 
        activation='relu',
        epochs=epochs, 
        batch_size=batch_size,
        final_activation='sigmoid')


# In[ ]:




