# Importing necessary library 
#import Augmentor 
# Passing the path of the image directory 
#p = Augmentor.Pipeline("C:/Users/Vedant/Desktop/introvert/") 
  
# Defining augmentation parameters and generating 5 samples 
#p.flip_left_right(0.5) 
#p.black_and_white(0.1) 
#p.rotate(0.3, 10, 10) 
#p.skew(0.4, 0.5) 
#p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5) 
#p.sample(5) 

 #Importing necessary functions 
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img 
   
# Initialising the ImageDataGenerator class. 
# We will pass in the augmentation parameters in the constructor. 
datagen = ImageDataGenerator( 
        rotation_range = 40, 
        #shear_range = 0.2, 
        #zoom_range = 0.2, 
        horizontal_flip = True) 
        #brightness_range = (0.8, 1.2)) 
    
# Loading a sample image  
path='C:/Users/Vedant/Desktop/wthres/'
j=1
for j in range(1,148):
	img = load_img(path+"train_"+str(j)+".jpg") 
	# Converting the input sample image to an array 
	x = img_to_array(img) 
	# Reshaping the input image 
	x = x.reshape((1, ) + x.shape)  

	# Generating and saving 5 augmented samples  
	# using the above defined parameters.  
	i = 1
	for batch in datagen.flow(x, batch_size = 1,save_to_dir ='C:/Users/Vedant/Desktop/w2',save_prefix ='image', save_format ='jpeg'):
		j+=1
		i += 1
		if i > 2:
			break