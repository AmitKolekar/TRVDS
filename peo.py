import cv2

# Load the video file
cap = cv2.VideoCapture('"C:\Users\HP\Desktop\project\presentation\TRVDS_v3\TRVDS_v3\Videos\final_traffic.mp4"')

# Load the pre-trained person detection model
person_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Set up variables for counting people on the bike
bike_count = 0
person_count = 0

# Loop through each frame of the video
while True:
    # Read the next frame
    ret, frame = cap.read()
    
    # If there are no more frames, break out of the loop
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect people in the frame
    people = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop through each person and check if they are on a bike
    for (x, y, w, h) in people:
        # If the person is at the top of the frame, assume they are on a bike
        if y < frame.shape[0] // 2:
            bike_count += 1
        
        # Increment the person count
        person_count += 1
    
    # Display the frame with bounding boxes around detected people
    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    
    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close the window
cap.release()
cv2.destroyAllWindows()

# Print the results
print(f'Number of people detected: {person_count}')
print(f'Number of people on bikes: {bike_count}')
