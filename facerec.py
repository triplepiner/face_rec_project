#!/usr/bin/env python
# coding: utf-8

# # Find faces in pictures
# We start by loading an example picture using Python imaging library. Go ahead and **run the cell bellow** to see the image.

# In[1]:

import os
from PIL import Image, ImageDraw
from IPython.display import display

# The program we will be finding faces on the example below
pil_im = Image.open('Jaroslav_and_Makar.jpeg')
display(pil_im)

# # Learning from example
# Now we show the library two different faces ([Joe Biden](/biden.jpg), [Barack Obama](/obama.jpg)) and generate the encodings for them. Encoding is simply a low dimensional representation of a face that can be easily compared with other faces the library will recognize in the future.

# In[2]:


import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
import PIL


# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.

makar_image = face_recognition.load_image_file("Makar_test.jpeg")
makar_face_encoding = face_recognition.face_encodings(makar_image)[0]

# Load a second sample picture and learn how to recognize it.
yaroslav_image = face_recognition.load_image_file("Jaroslav_test.jpeg")
yaroslav_face_encoding = face_recognition.face_encodings(yaroslav_image)[0]



# Create arrays of known face encodings and their names
known_face_encodings = [
    makar_face_encoding,
    yaroslav_face_encoding,
]
known_face_names = [
    "Makar",
    "Yaroslav"
]
print('Learned encoding for', len(known_face_encodings), 'images.')

# # Showtime
# Finally, we load the image we looked at in the first cell, find the faces in the image and compare them with the encodings the library generated in the previous step. We can see that library now correctly recognizes Barack and Joe in the input.

# In[3]:


# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("Jaroslav_and_Makar.jpeg")

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    f = open('name.txt', 'w')
    f.write(f'{name} ')
    f.close()
    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    #f = open('name.txt', 'w')
    #f.write(f'{name}')
    #f.close()

# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image

display(pil_image)
pil_image.save("result.png")
#name.save("name.txt")
#f=open('name.txt', 'w')
#f.write(f'{name} ')
#f.close()
# In[3]:


