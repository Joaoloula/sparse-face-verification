def read_database(database_number):
    assert 1 <= database_number <= 9

    if database_number == 10:
        dataset = '10'
    else:
        dataset = '0'+str(database_number)

    # Open datsasets and save the filenames, while associating the right labels
    with open('/home/loula/Programming/Python_Scripts/CNN_LFW/lfwcrop_grey/'
              'lists/'+dataset+'_train_diff.txt') as f:
        train_mismatches_ = f.readlines()
    train_mismatches = [[string.split(), 0] for string in train_mismatches_]
    with open('/home/loula/Programming/Python_Scripts/CNN_LFW/lfwcrop_grey/'
              'lists/'+dataset+'_train_same.txt') as f:
        train_matches_ = f.readlines()
    train_matches = [[string.split(), 1] for string in train_matches_]

    train_set = train_mismatches+train_matches

    with open('/home/loula/Programming/Python_Scripts/CNN_LFW/lfwcrop_grey/'
              'lists/'+dataset+'_test_diff.txt') as f:
        test_mismatches_ = f.readlines()
    test_mismatches = [[string.split(), 0] for string in test_mismatches_]
    with open('/home/loula/Programming/Python_Scripts/CNN_LFW/lfwcrop_grey/'
              'lists/'+dataset+'_test_same.txt') as f:
        test_matches_ = f.readlines()
    test_matches = [[string.split(), 1] for string in test_matches_]

    test_set = test_mismatches+test_matches

    return (train_set, test_set)
