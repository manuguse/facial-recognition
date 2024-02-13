import cv2, face_recognition
from os import listdir

def encode_image(img_path):
    img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_encoding = face_recognition.face_encodings(rgb_img)[0]
    return img_encoding

def load_images(folder):
    images = {}
    for file in listdir(folder):
        file_path = f'{folder}/{file}'
        img_encoding = encode_image(file_path)
        person_name = file.replace(".jpg", '').replace("_", ' ') # only works for .jpg files for now
        images[person_name] = img_encoding
    return images

count = 0
images = load_images('images')
camera = cv2.VideoCapture(0)
found_person = False
found_face = False
rec_color = (0, 0, 255)
items = []

while True:
    ret, frame = camera.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if count == 10:
        items = []
        count = 0
    if count == 0:
        for index, face in enumerate(face_locations):
            top, right, bottom, left = face
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[index]
            currentItem = ((0, 0, 255), 'unknown', face)
            for person_name, encoding in images.items():
                result = face_recognition.compare_faces([encoding], face_encoding)
                if result[0]:
                    currentItem = ((0, 255, 0), person_name, face)
                    break
            items.append(currentItem)
    for item in items:
        top, right, bottom, left = item[2]
        text_position = (left, bottom + 20)
        cv2.putText(frame, item[1], text_position, cv2.FONT_HERSHEY_DUPLEX, 0.5, item[0], 1)
        cv2.rectangle(frame, (left, top), (right, bottom), item[0], 2)
    
    cv2.imshow('video', frame)
    count += 1
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()