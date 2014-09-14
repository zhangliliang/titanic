%tic;
[heart_scale_label, heart_scale_inst] = libsvmread('./liblinear-1.94/heart_scale');
model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1');
[predict_label, accuracy, dec_values] = svmpredict(heart_scale_label, heart_scale_inst, model); % test the training data
%toc;