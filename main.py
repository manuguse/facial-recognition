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

noface_position = (0, 0)

while True:
    ret, frame = camera.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) != 0:
        found_face = True
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), rec_color, 2)
            if count % 10 == 0:
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                for person_name, encoding in images.items():
                    result = face_recognition.compare_faces([encoding], face_encoding)
                    if result[0]:
                        found_person = True
                        break
                    else:
                        found_person = False
    else:
        found_person = False
        found_face = False

    if found_person:
        text_position = (left, bottom + 20)
        cv2.putText(frame, person_name, text_position, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
        rec_color = (0, 255, 0)
    else:
        rec_color = (0, 0, 255)
        if found_face:
            text_position = (left, bottom + 20)
            cv2.putText(frame, 'unknown', text_position, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
        # else:
        #     text_width, _ = cv2.getTextSize('no face found'.upper(), cv2.FONT_HERSHEY_DUPLEX, 1, 2)
        #     text_y = (frame.shape[0] - text_width[1])
        #     text_x = (frame.shape[1] - text_width[0]) // 2
        #     text_position = (text_x, text_y)
        #     cv2.putText(frame, 'no face found'.upper(), text_position, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('video', frame)
    
    count += 1
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()