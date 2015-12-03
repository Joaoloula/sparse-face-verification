function fullFileName = Create_NameFile_Vector(directory)
    %Creates cell with filenames for the database
    %Works for directories with subfolders that contain .jpg images
    filePattern=fullfile(directory, '*\*.jpg');
    images=rdir(filePattern);
    size = length(images);
    fullFileName=cell(size, 1);
    for k = 1 : size
        baseFileName = images(k).name;
        fullFileName{k} = fullfile(baseFileName);
    end
end
