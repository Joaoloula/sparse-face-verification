function [split1, split2] = split(matrix, split_position)
    shuffled_matrix=matrix(:,randperm(size(matrix,2)));
    split1 = shuffled_matrix(:,1: split_position);
    split2 = shuffled_matrix(:,split_position+1:end);
end

