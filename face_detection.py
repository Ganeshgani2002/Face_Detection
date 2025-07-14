import cv2  # Make sure to use lowercase 'cv2'

# Load the Haar cascade classifier
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open the default camera (0)
cam = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=4)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Face Detection", img)

    # Break the loop on 'Esc' key press (ASCII 27)
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the camera and close windows
cam.release()
cv2.destroyAllWindows()
