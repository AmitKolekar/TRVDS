import cv2
import os

# specify the folder containing the images
folder_path = "C:/Users/HP/Desktop/project/presentation/TRVDS_v3/TRVDS_v3/TrafficRecord/bike/3prrp"

# loop through all the files in the folder
for filename in os.listdir(folder_path):
    # check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # load the image using OpenCV
        img = cv2.imread(os.path.join(folder_path, filename))

        # display the image
        cv2.imshow(filename, img)

# wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
