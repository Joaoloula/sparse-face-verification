function label = label_identify(person, file_vector)
    %Creates label with 1 for person of interest and 0 for the rest
    label = 1-2*cellfun('isempty', strfind(file_vector, person));
end