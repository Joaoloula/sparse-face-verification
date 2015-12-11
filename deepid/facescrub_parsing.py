# Parser for saving images in facescrub.txt in cropped 32x32 format
# we use requests to access the urls as it's more consistent than urllib

import pickle
from PIL import Image
import requests
from StringIO import StringIO
from random import shuffle
import signal

# Path to read and write directories
rdir_path = '/home/loula/Programming/python/face_verification/deepid/facescrub/'
wdir_path = ('/home/loula/Programming/python/face_verification/deepid/'
             'fs_labeled/')


# Define alarm class and exception for timeout
class AlarmException(Exception):
    pass


def alarmHandler(signum, frame):
    raise AlarmException


def download_image(URL):
    # Downloads image from an url
    signal.signal(signal.SIGALRM, alarmHandler)
    signal.alarm(1)
    response = requests.get(URL)
    img = Image.open(StringIO(response.content))
    return img


def create_url_vector(path):
    # Creates array with [[name], [url], [bbox], [sha256-key]]
    # from facescrub txt file
    with open(path, 'r') as txt:
        scrub_txt = txt.read()
        # Split on divisions, start at first image
        parsed = scrub_txt[37:].split('\n')
        structured_urls = []
        for i in xrange(1, len(parsed)-1):
            infos = parsed[i].split('\t')
            structured_urls.append([infos[0], infos[3], infos[4], infos[5]])
        return structured_urls

labeled_list = (create_url_vector(rdir_path+'facescrub_actors.txt') +
                create_url_vector(rdir_path+'facescrub_actresses.txt'))
image_labels = []  # List where we'll put filenames and labels
failed_images = []  # List of images where download failed
current_name = 'Aaron Eckhart'  # First name
current_num = 0  # First label
total_img_number = len(labeled_list)

for i in xrange(total_img_number):
    if i % 1000 == 0 and i != 0:
        print ('Progress: %0.2f, error rate = %0.2f' %
               (float(i)/float(total_img_number),
                float(len(failed_images))/float(i))
               )
    try:
        image = download_image(labeled_list[i][1])
        # Get bbox dimensions
        bbox_string = labeled_list[i][2].split(',')
        bbox = map(int, bbox_string)
        # Crop and resize to 32x32
        image = image.crop(bbox).resize([32, 32], Image.ANTIALIAS)
        name = labeled_list[i][0]
        if name != current_name:
            current_name = name
            current_num += 1
        # Save image to /fs_labeled/image_number.jpg
        image_path = wdir_path + str(i) + '.jpg'
        image.save(image_path)
        image_labels.append([image_path, current_num])
    except (IOError, AlarmException):
        # Requests failed to access the url or timed out
        failed_images.append(i)

# Shuffle the dataset and split it into train and test
shuffle(image_labels)
[train, test] = [image_labels[:80000], image_labels[80001:]]

# Save dataset
pickle.dump([train, test], open('facescrub_labeled', 'w'))
pickle.dump([failed_images], open('facescrub_fails', 'w'))
