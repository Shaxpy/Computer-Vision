#visit the links
#download the image
#Open image in OPENCV and resize 100x100
#COnvert to gray and save
import urllib.request
import cv2
import numpy as np
import os
def store_raw_images():
    neg_images_link='http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04555897'
    neg_image_urls=urllib.request.urlopen(neg_images_link).read().decode()

    if not os.path.exists('neg'):
        os.makedirs('neg')

    pic_num=1
    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i,'neg/'+str(pic_num)+'.jpg')
            img=cv2.imread('neg/'+str(pic_num)+'.jpg',cv2.IMREAD_GRAYSCALE)
            resized_image=cv2.resize(img,(100,100))
            cv2.imwrite('neg/'+str(pic_num)+'.jpg',resized_image)
            pic_num +=1
        except Exception as e:
            print(str(e))
def find():
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('ugli'):
                try:
                    current_image_path=str(file_type)+'/'+str(img)
                    ugly=cv2.imread('ugli/'+str(ugly))
                    ques=cv2.imread(current_image_path)

                    if ugly.shape==ques.shape and not(np.bitwise_xor(ugly,ques).any()):
                        print('Baddddd')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))

def pos_n_neg():
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            if file_type=='neg':
                line=file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)
            # elif file_type == 'pos':
            #     line = file_type+'/'+img+' 1 0 0 50 50\n'
            #     with open('info.dat','a') as f:
            #         f.write(line)
#store_raw_images()---meant to store images from the Imagenet link
#find()----find the bad images and store them differently in Ugli folder
pos_n_neg()#---make descrip for the neg dir- images (since pos is already made) 
#FInally we make samples of positive images 

#opencv_createsamples -img 111.jpeg -bg bg.txt -info info/info.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1650

# Thus, these are  "positive" images, created from otherwise "negative" images, and that 
# negative image will also be used in training. Now that we have positive images, we now need to
# create the vector file, which is basically where we 
# stitch all of our positive images together. We will actually be using opencv_createsamples again

#opencv_createsamples -info info/info.lst -num 1650 -w 20 -h 20 -vec positives.vec

#Finally cascade classifier is made  
#opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 1600 -numNeg 800 -numStages 10 -w 20 -h 20


