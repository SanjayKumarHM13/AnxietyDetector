import cv2
import numpy as np
import tensorflow as tf

def load_model():
    return tf.keras.models.load_model('models/anxiety_model.h5')

def detect_anxiety():
    model = load_model()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    
    # Get the number of output classes from the model
    num_classes = model.output_shape[-1]
    print(f"Number of classes in the model: {num_classes}")
    
    # Define emotions based on the number of classes
    emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][:num_classes]
    print(f"Emotions being used: {emotions}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            
            prediction = model.predict(roi_gray)[0]
            emotion_index = np.argmax(prediction)
            
            if emotion_index < len(emotions):
                emotion = emotions[emotion_index]
            else:
                emotion = f"Unknown ({emotion_index})"
            
            confidence = prediction[emotion_index]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            if emotion in ['Sad', 'Fear']:
                cv2.putText(frame, 'Possible Anxiety Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Anxiety Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_anxiety()