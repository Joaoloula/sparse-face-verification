function vectorized = vectorizer(cell)
    lines=length(cell);
    columns=length(cell{1}(:));
    vectorized = zeros(lines, columns);
    for n=1:lines
        vectorized(n, :)=cell{n}(:)';
    end
    vectorized=vectorized';
end
        