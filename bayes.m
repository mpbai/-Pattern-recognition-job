clear 
clc
% load iris_dataset
% data=irisInputs';
 data= load('data.dat');
total=10560;
train_num=350;
test_num=130;%ÿһ��480�� 52ά 22��
class_num=22;
variable_num=52;
datapred=zeros(total,variable_num);%Ԥ������
class_correctnum=zeros(class_num,1);
color=[0 0 0;0 0 1;0 1 0;0 1 1; 1 0 0; 1 1 0;1 0 1;...                        
            0.5 0.5 0.5;0.5 0.5 1;0.5 1 0.5;0.5 1 1; 1 0.5 0.5; 1 1 0.5;1 0.5 1;...
              0.5 0.1 0.5;0.5 0.1 1;0.5 1 0.8; 0.8 0.2 0.8; 1 0.9 0.2;0.7 0.1 1;0.3 0.5 0.7;0.9 0.5 0.4 ];
for i = 1 : class_num
tempdata{i} = data((i-1)*480+1:i*480,:);
datatrain{i} = tempdata{i}(1:train_num,:); %ѵ������
datapred((i-1)*test_num+1:i*test_num,:) =tempdata{i}(train_num+1:480,:); %Ԥ������
for j = 1 : variable_num
 [mu(i,j),sigma(i,j)]=normfit(datatrain{i}(:,j));%��ѵ�����ݸ�������ֵ�ͷ��� 
end
end
post = zeros(test_num*class_num,class_num);%����������ֵΪ0 
for i = 1 : test_num*class_num
for j = 1 : class_num
prodt = ones(class_num,1);
for k = 1 : variable_num 
 prodt(j) = prodt(j) * normpdf(datapred(i,k),mu(j,k),sigma(j,k)); %�����ر�Ҷ˹������Ȼ����
end
post(i,j) = prodt(j); %1.������ͬ��2.��һ��������ͬ�����Ժ������likelihood
end
[~, Ind] = max(post(i,:));%ѡ����������ʶ�Ӧ�����
label(i) = Ind;
end
mybar=zeros(class_num,class_num);
for i=1:class_num  
    for j=1:class_num
        mybar(i,j)=length(find(label((i-1)*test_num+1:i*test_num)==j));
    end
    correct(i)= mybar(i,i)/test_num;
end
fprintf('�ܵ���ȷ��Ϊ: ')
meancorrect=mean(correct)

figure(1)
h=bar(mybar,'stacked') ;
for p=1:class_num
set(h(p),'facecolor',color(p,:))
end
legend('���1','���2','���3','���4','���5','���6','���7','���8','���9','���10','���11','���12','���13','���14','���15','���16',...
'���17','���18','���19','���20','���21','���22');
ylabel('������');
xlabel('���');
hold off
figure(2)
correct=([correct,meancorrect]);
correctbar=bar( correct) ;
 xlabel('�����');
ylabel('��ȷ��');
  
  
  
  