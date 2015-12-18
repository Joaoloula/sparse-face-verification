import pickle

_, test = pickle.load(open("facescrub_labeled", "rb"))
people_number = 530
ordered_test = [[] for i in range(people_number)]

for identity in range(people_number):
    for image in test:
        if image[1] == identity:
            ordered_test[identity].append(image[0])

pickle.dump(ordered_test, open('ordered_test', 'w'))
