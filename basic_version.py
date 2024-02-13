import cv2, face_recognition
from os import listdir

def encode_image(img_path):
    img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_encoding = face_recognition.face_encodings(rgb_img)[0]
    return img_encoding

def load_images(folder):
    images = {}
    print(f'{len(listdir(folder))} images found in the folder "{folder}"')
    for file in listdir(folder):
        file_path = f'{folder}/{file}'
        img_encoding = encode_image(file_path)
        person_name = file.replace(".jpg", '').replace("_", ' ') # only works for .jpg files for now
        images[person_name] = img_encoding
    return images

test_img = encode_image('test_images/olivia2_test.jpg')

folder_name = 'images'
found_person = False
for person_name, encoding in load_images(folder_name).items():
    result = face_recognition.compare_faces([test_img], encoding)
    if result[0]:
        print(f'{person_name} is a match!')
        found_person = True
        break

if not found_person:
    print('no match found')
