import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model,load_model



# Plot the evaluation metrics for all the epochs 
def plot_histories(histories):
    
    # Input: histories obtained from training using fit_generator
    for h, history in enumerate(histories):
        keys = history.history.keys()
        fig, axs = plt.subplots(1, len(keys)//2, figsize = (25, 5))
        fig.suptitle('No. ' + str(h+1) + ' Fold Training Results' , fontsize=30)

        for k, key in enumerate(list(keys)[:len(keys)//2]):
            training = history.history[key]
            validation = history.history['val_' + key]

            epoch_count = range(1, len(training) + 1)

            axs[k].plot(epoch_count, training, 'r--')
            axs[k].plot(epoch_count, validation, 'b-')
            axs[k].legend(['Training ' + key, 'Validation ' + key])
    
    with open(str(h+1) + '_trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)



# Predicting results using test dataset (from pre-trained model) 
def pred_results_tl(TLmodel, test_df, number_of_img, height, width):
    for i in range(number_of_img):
        index = np.random.randint(0,len(test_df.index))
        print("Testing image", i+1, ":", index)

        img = cv2.imread(test_df['filename'].iloc[index])
        img = cv2.resize(img, (height, width))
        img = img[np.newaxis, :, :, :]
        img = img / 255
        tl_pred = TLmodel.predict(img)

        plt.figure(figsize=(10,10))
        plt.subplot(1,3,1)
        plt.imshow(np.squeeze(img))
        plt.title('Original Image')

        plt.subplot(1,3,2)     
        plt.imshow(np.squeeze(cv2.resize(cv2.imread(test_df['mask'].iloc[index]), (height, width))))
        plt.title('Original Mask')

        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(tl_pred) > .5)
        plt.title('Prediction (TL)')

        plt.show()

 

# Predicting results using test dataset (from fine-tuned model, and compare with pre-trained model) 
def pred_results_ft(TLmodel, FTmodel, test_df, number_of_img, height, width):
    for i in range(number_of_img):
        index = np.random.randint(0,len(test_df.index))
        print("Testing image", i+1, ":", index)

        img = cv2.imread(test_df['filename'].iloc[index])
        img = cv2.resize(img, (height, width))
        img = img[np.newaxis, :, :, :]
        img = img / 255
        tl_pred = TLmodel.predict(img)
        ft_pred = FTmodel.predict(img)

        plt.figure(figsize=(8,8))
        plt.subplot(1,4,1)
        plt.imshow(np.squeeze(img))
        plt.title('Original Image')

        plt.subplot(1,4,2)     
        plt.imshow(np.squeeze(cv2.resize(cv2.imread(test_df['mask'].iloc[index]), (height, width))))
        plt.title('Original Mask')

        plt.subplot(1,4,3)
        plt.imshow(np.squeeze(tl_pred) > .5)
        plt.title('Prediction (Model 1)')

        plt.subplot(1,4,4)
        plt.imshow(np.squeeze(ft_pred) > .5)
        plt.title('Prediction (Model 2)')

        plt.show()


# Display prediction results on top of the original images
def display_pred(TLmodel, FTmodel, test_df, number_of_img, height, width):
    
    for i in range(number_of_img):
        index = np.random.randint(0,len(test_df.index))
        print("Testing image", i+1, ":", index)

        img = cv2.imread(test_df['filename'].iloc[index])
        img = cv2.resize(img, (height, width))
        img = img[np.newaxis, :, :, :]
        img = img / 255
        tl_pred = TLmodel.predict(img)
        ft_pred = FTmodel.predict(img)

        plt.figure(figsize=(8,8))
        plt.subplot(1,3,1)
        plt.imshow(np.squeeze(img))
        plt.imshow(np.squeeze(cv2.resize(cv2.imread(test_df['mask'].iloc[index]), (height, width))), cmap='Reds', alpha=0.4)
        plt.title('Original')

        plt.subplot(1,3,2)     
        plt.imshow(np.squeeze(img))
        plt.imshow(np.squeeze(tl_pred) > .5, cmap='Reds', alpha=0.5)
        plt.title('Prediction (Model 1)')

        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(img))
        plt.imshow(np.squeeze(ft_pred) > .5, cmap='Reds', alpha=0.5)
        plt.title('Prediction (Model 2)')

        plt.show()




        


      
