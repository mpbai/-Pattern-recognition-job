# -Pattern-recognition-job
clear 
clc
% load iris_dataset
% data=irisInputs';
 data= load('data.dat');
total=10560;
train_num=350;
test_num=130;%每一类480个 52维 22类
class_num=22;
variable_num=52;
datapred=zeros(total,variable_num);%预测数据
class_correctnum=zeros(class_num,1);
color=[0 0 0;0 0 1;0 1 0;0 1 1; 1 0 0; 1 1 0;1 0 1;...                        
            0.5 0.5 0.5;0.5 0.5 1;0.5 1 0.5;0.5 1 1; 1 0.5 0.5; 1 1 0.5;1 0.5 1;...
              0.5 0.1 0.5;0.5 0.1 1;0.5 1 0.8; 0.8 0.2 0.8; 1 0.9 0.2;0.7 0.1 1;0.3 0.5 0.7;0.9 0.5 0.4 ];
for i = 1 : class_num
tempdata{i} = data((i-1)*480+1:i*480,:);
datatrain{i} = tempdata{i}(1:train_num,:); %训练数据
datapred((i-1)*test_num+1:i*test_num,:) =tempdata{i}(train_num+1:480,:); %预测数据
for j = 1 : variable_num
 [mu(i,j),sigma(i,j)]=normfit(datatrain{i}(:,j));%求训练数据各变量均值和方差 
end
end
post = zeros(test_num*class_num,class_num);%后验概率设初值为0 
for i = 1 : test_num*class_num
for j = 1 : class_num
prodt = ones(class_num,1);
for k = 1 : variable_num 
 prodt(j) = prodt(j) * normpdf(datapred(i,k),mu(j,k),sigma(j,k)); %求朴素贝叶斯法的似然函数
end
post(i,j) = prodt(j); %1.先验相同，2.归一化因子相同，所以后验等于likelihood
end
[~, Ind] = max(post(i,:));%选择最大后验概率对应的类别
label(i) = Ind;
end
mybar=zeros(class_num,class_num);
for i=1:class_num  
    for j=1:class_num
        mybar(i,j)=length(find(label((i-1)*test_num+1:i*test_num)==j));
    end
    correct(i)= mybar(i,i)/test_num;
end
fprintf('总的正确率为: ')
meancorrect=mean(correct)

figure(1)
h=bar(mybar,'stacked') ;
for p=1:class_num
set(h(p),'facecolor',color(p,:))
end
legend('类别1','类别2','类别3','类别4','类别5','类别6','类别7','类别8','类别9','类别10','类别11','类别12','类别13','类别14','类别15','类别16',...
'类别17','类别18','类别19','类别20','类别21','类别22');
ylabel('样本数');
xlabel('类别');
hold off
figure(2)
correct=([correct,meancorrect]);
correctbar=bar( correct) ;
 xlabel('类别编号');
ylabel('正确率');
