# snap_chat_filter
This project put sunglasses on detected face by finding 30 facial keypoints.These keypoints mark important areas of the face — the eyes, corners of the mouth, the nose, etc.
This dataset on Kaggle is used to train model to detect the facial keypoints given an image with a face.
https://www.kaggle.com/c/facial-keypoints-detection
Each datapoint in the dataset contains space separated pixel values of the images in a sequential order and the last 30 values of the datapoint represent 15 pairs of coordinates of the keypoints on the faces.
So we just have to train a CNN model to solve a classic deep learning regression problem. 


