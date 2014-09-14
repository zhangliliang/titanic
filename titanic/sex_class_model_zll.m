%% read train.csv
%1 PassengerId
%2 Survived
%3 Pclass
%4 Name
%5 Sex
%6 Age
%7 SibSp
%8 Parch
%9 Ticket
%10 Fare
%11 Cabin
%12 Embarked
fid = fopen('train.csv');
traindata = textscan(fid,'%d %d %d %q %s %f %d %d %s %f %s %s','Delimiter', ',','HeaderLines',1);
fclose(fid);

%% get label(0 or 1), sex(0 as male, 1 as female), class(1,2,3)
label = traindata{2};
sex = strcmp('female', traindata{5});
class = traindata{3};
age = traindata{6};
fare = traindata{10};

%% train a SVM model
X = sparse(double([sex, class]));
y = double(label);
model = train(y, X, '-s 3 -B 1 -v 5 -c 1');

% %% read the test.csv
% %1 PassengerId
% %2 Pclass
% %3 Name
% %4 Sex
% %5 Age
% %6 SibSp
% %7 Parch
% %8 Ticket
% %9 Fare
% %10 Cabin
% %11 Embarked
% fid = fopen('test.csv');
% testdata = textscan(fid,'%d %d %q %s %f %d %d %s %f %s %s','Delimiter', ',','HeaderLines',1);
% fclose(fid);
% 
% %% get sex
% sex = strcmp('female', testdata{4});
% 
% %% predict by SVM
% X = double(sparse(sex));
% predict_labels = predict(zeros(size(X,1), 1), X, model);
% 
% %% write to svm
% ofile = fopen('sex.predict.csv', 'w');
% fprintf(ofile, 'PassengerId,Survived\n');
% for i = 1:numel(testdata{1})
%     pid = testdata{1}(i);
%     fprintf(ofile, '%d,%d\n', pid, predict_labels(i));
% end
% fclose(ofile);




