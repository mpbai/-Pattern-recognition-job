
clear;
clc;

%% 测试数据故障发生时间
 happen=160;
 
 %% 读取数据 d00: 正常数据；d08：故障数据
 d00=importdata('d00.dat');
 d08=importdata('d05_te.dat');
 X=d00';
 XT=d08;
 
 %% 数据规范化处理，包括测试数据（测试数据按训练数据得到的均值、方差进行规范化）
[X,mean,std]=zscore(X); %数据预处理，并返回训练集的均值和标准差
XT=(XT-ones(size(XT,1),1)*mean)./(ones(size(XT,1),1)*std);%新的故障数据集在标准化时作为减数的均值和作为除数的标准差都取自训练集


%% 主分量分析
[COEFF, SCORE, LATENT] = pca(X);
%[COEFF, SCORE, LATENT] = princomp(X); %LATENT is the eigenvalues of the covariance matrix of X, COEFF is the loadings, SCORE is the principal component scores, TSQUARED is the Hotelling's T-squared statistic for each observation in X.
percent = 0.85; %input('确定方差贡献率限(0～1）：')       %the predetermined contribution rate, usually 85%
beta=0.99;      %input('确定统计阈值置信度(0～1）：')    %beta is the inspection level
k=0;
for i=1:size(LATENT,1)      %根据方差贡献率确定主元个数k（与第i个负荷向量相关联的方差等于对应的特征值）； %% choose first k principal components
    alpha(i)=sum(LATENT(1:i))/sum(LATENT);
    if alpha(i)>=percent  
        k=i;
        break;  
    end 
end

P=COEFF(:,1:k);               %P为负荷矩阵（投影矩阵）

%% 计算t2, SPE统计量
for i=1:size(X,1)
    t2(i)=X(i,:)*P*inv(diag(LATENT(1:k)))*P'*X(i,:)'; %计算t2统计量   
    SPE(i)=X(i,:)*COEFF*(X(i,:)*COEFF)'-X(i,:)*P*(X(i,:)*P)';
end

%% t2统计量的控制限 （按t2分布计算）
T2knbeta=k*(size(X,1)-1)*(size(X,1)+1)/(size(X,1)-k)/size(X,1)*finv(beta,k,(size(X,1)-k)); %t2统计量的控制限

%% spe统计量的控制限
a=sum(SPE)/size(SPE,2);
b=var(SPE);
SPEbeta=b/(2*a)*chi2inv(beta,2*a^2/b);

%% 计算测试数据的t2,SPE统计量 （happen前数据正常，后面异常）
for i=1:size(XT,1);
    XTt2(i)=XT(i,:)*P*inv(diag(LATENT(1:k)))*P'*XT(i,:)';
    XTSPE(i)=XT(i,:)*COEFF*(XT(i,:)*COEFF)'-XT(i,:)*P*(XT(i,:)*P)';
end

%% 作图显示结果
figure(11)
subplot(2,1,1);
plot(1:happen,XTt2(1:happen),'b',happen+1:size(XTt2,2),XTt2(happen+1:end),'r');
hold on;
TS=T2knbeta*ones(size(XT,1),1);
plot(TS,'k--');
title('PCA-T2 for TE data');
xlabel('Sample');
ylabel('T2');
hold off;
subplot(2,1,2);
plot(1:happen,XTSPE(1:happen),'b',happen+1:size(XTSPE,2),XTSPE(happen+1:end),'r');
hold on;
S=SPEbeta*ones(size(XT,1),1);
plot(S,'k--');
title('PCA-SPE for TE data');
xlabel('Sample');
ylabel('SPE');
hold off;

%% False alarm rate 虚警率（误报率）
falseT2=0;
falseSPE=0;
for wi=1:happen
    if XTt2(wi)>T2knbeta
        falseT2=falseT2+1;
    end
    falserate_pca_T2=100*falseT2/happen;
    if XTSPE(wi)>SPEbeta
        falseSPE=falseSPE+1;
    end
    falserate_pca_SPE=100*falseSPE/happen;
end

%% Miss rate 漏报率
missT2=0;
missSPE=0;
for wi=happen+1:size(XTt2,2)
    if XTt2(wi)<T2knbeta
        missT2=missT2+1;
    end
    if XTSPE(wi)<SPEbeta
        missSPE=missSPE+1;
    end 
end
missrate_pca_T2=100*missT2/(size(XTt2,2)-happen);
missrate_pca_SPE=100*missSPE/(size(XTt2,2)-happen);

%% 结果输出：考虑连续几个样本的情况，
 disp('----PCA--False alarm rate----');
falserate_pca=[falserate_pca_T2 falserate_pca_SPE]
 disp('----PCA--Miss  rate----');
missrate_pca=[missrate_pca_T2 missrate_pca_SPE]
