from imutils import paths
import numpy as np
import pickle
import imutils
import cv2 
import time
import os

# Face detection model paths
ProtoPath = "face_detection_model\deploy.prototxt"
ModelPath = "face_detection_model\res10_300x300_ssd_iter_140000.caffemodel"
EmbedderPath = "face_detection_model\openface_nn4.small2.v1.t7"

# Load face detection and embedding models
detector = cv2.dnn.readNetFromCaffe(ProtoPath, ModelPath) 
embedding = cv2.dnn.readNetFromTorch(EmbedderPath)

# Load user-password mapping
user_password_mapping = {
    "Samyukta": "password0",
    "Aryan": "password1",
    "Shivangi": "password3",
    "Vinithra": "password4",
    # Add more users as needed
}

# Function to verify username
def verify_username(username):
    # Verify if the entered username is in the mapping
    if username in user_password_mapping:
        return True
    else:
        print("User Not Registered.")
        print("Please Try Again.")
        print()
        return False
    
# Function to verify password
def verify_password(username, password):
    # Verify password for the given username
    if user_password_mapping.get(username) == password:
        return True
    else:
        print("Incorrect Password.")
        print("Please Try Again.")
        print()
        return False

# Function to capture user's face and create embeddings
def capture_face_embeddings():
    names = []
    embeddings = []

    # Read images from dataset directory
    imagePaths = list(paths.list_images("./dataset"))

    for (_, imagePath) in enumerate(imagePaths):
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(str(imagePath))

        (h, w) = image.shape[:2] 
        image_blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(image_blob)
        detections = detector.forward()

        index = np.argmax(detections[0,0,:,2])
        box = detections[0,0,index,3:7] * np.array([w,h,w,h]) 
        x1, y1, x2, y2 = box.astype(int)
        face = image[y1:y2, x1:x2]

        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
            (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedding.setInput(faceBlob)
        embedding_val = embedding.forward()
        embeddings.append(embedding_val.flatten())
        names.append(name)

    data = {"embeddings" : embeddings ,"names" : names}

    f = open("./pickle/embeddings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close() 

# Function to authenticate using face
def authenticate_face():
    # Load face embeddings from pickle file
    data = pickle.loads(open("./pickle/embeddings.pickle", "rb").read())
    embeddings = data["embeddings"]
    names = data["names"]

    # Capture video from webcam
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = vs.read()
        (h, w) = frame.shape[:2]

        image_blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(image_blob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                face = frame[y1:y2, x1:x2]

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedding.setInput(faceBlob)
                vec = embedding.forward()

                # Loop over the embeddings
                for (i, embedding_vec) in enumerate(embeddings):
                    # Compute the Euclidean distance between the current embedding
                    # and the embedding of the image
                    distance = np.linalg.norm(vec - embedding_vec)
                    if distance < 0.5:
                        return names[i]

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Step 1: Username verification
    while True:
        username = input("Enter Username: ")
        if verify_username(username):
            break

    # Step 2: Password verification
    while True:
        password = input("Enter Password: ")
        if verify_password(username, password):
            break

    # Step 3: Face authentication
    print("Please Look at the Camera for Face Authentication...")
    authenticated_user = authenticate_face()
    if authenticated_user == username:
        print("Face Authentication Successful!")
        print("Welcome,", username)
    else:
        print("Face Authentication Failed!")
        print("Access Denied.")

# Execute the main function
if __name__ == "__main__":
    main()
