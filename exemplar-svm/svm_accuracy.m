function accuracy = svm_accuracy(w, b, test_set)
    predicted_scores=test_set(1:end-1,:)'*w+b+0.2;
    predicted_labels = (predicted_scores>0)-(predicted_scores<=0);
    georges = length(intersect(find(predicted_labels>0), find(test_set(end,:)'>0)))
    total = sum(predicted_labels+1)/2
    accuracy = norm(test_set(end,:)'-predicted_labels, 1)/13233;
end