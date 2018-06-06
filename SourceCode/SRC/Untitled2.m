train_num=10;
R1=randperm(2414,train_num*38);
R2=randperm(2414,2414-train_num*38);
Train_set = fea(R1,:);
train_label = gnd(R1,:);
Test_set=fea(R2,:);
Test_label = gnd(R2,:);

predictions=src(Train_set,train_label,Test_set,0.3);

accuracy=length(find(Test_label-predictions==0))/(2414-train_num*38);
