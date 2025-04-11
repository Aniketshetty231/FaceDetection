import cv2

# Load the pre-trained Haar cascade face detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame-by-frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale (Haar cascades work better on gray images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the live video with rectangles
    cv2.imshow("Face Detection", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
