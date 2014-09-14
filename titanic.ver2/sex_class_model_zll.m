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

%% scale the data to [0, 1]
X = double([sex class age fare]);
X(isnan(X)) = 0;
maxX = max(X);
minX = min(X);
Xscale = bsxfun(@rdivide, bsxfun(@minus, X, minX), maxX-minX);

%% train a SVM model
X = sparse(double(Xscale));
y = double(label);
model = svmtrain(y, X, '');

%% read the test.csv
%1 PassengerId
%2 Pclass
%3 Name
%4 Sex
%5 Age
%6 SibSp
%7 Parch
%8 Ticket
%9 Fare
%10 Cabin
%11 Embarked
fid = fopen('test.csv');
testdata = textscan(fid,'%d %d %q %s %f %d %d %s %f %s %s','Delimiter', ',','HeaderLines',1);
fclose(fid);

%% get sex
sex = strcmp('female', testdata{4});
class = testdata{2};
age = testdata{7};
fare = testdata{9};

%% scale the data to [0, 1]
X = double([sex class age fare]);
X(isnan(X)) = 0;
Xscale = bsxfun(@rdivide, bsxfun(@minus, X, minX), maxX-minX);

%% predict by SVM
X = sparse(double(Xscale));
predict_labels = svmpredict(zeros(size(X,1), 1), X, model);

%% write to svm
ofile = fopen('sex.class.age.fare.csv', 'w');
fprintf(ofile, 'PassengerId,Survived\n');
for i = 1:numel(testdata{1})
    pid = testdata{1}(i);
    fprintf(ofile, '%d,%d\n', pid, predict_labels(i));
end
fclose(ofile);




