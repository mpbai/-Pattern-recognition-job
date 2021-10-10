
clear;
clc;

%% �������ݹ��Ϸ���ʱ��
 happen=160;
 
 %% ��ȡ���� d00: �������ݣ�d08����������
 d00=importdata('d00.dat');
 d08=importdata('d05_te.dat');
 X=d00';
 XT=d08;
 
 %% ���ݹ淶�����������������ݣ��������ݰ�ѵ�����ݵõ��ľ�ֵ��������й淶����
[X,mean,std]=zscore(X); %����Ԥ����������ѵ�����ľ�ֵ�ͱ�׼��
XT=(XT-ones(size(XT,1),1)*mean)./(ones(size(XT,1),1)*std);%�µĹ������ݼ��ڱ�׼��ʱ��Ϊ�����ľ�ֵ����Ϊ�����ı�׼�ȡ��ѵ����


%% ����������
[COEFF, SCORE, LATENT] = pca(X);
%[COEFF, SCORE, LATENT] = princomp(X); %LATENT is the eigenvalues of the covariance matrix of X, COEFF is the loadings, SCORE is the principal component scores, TSQUARED is the Hotelling's T-squared statistic for each observation in X.
percent = 0.85; %input('ȷ�����������(0��1����')       %the predetermined contribution rate, usually 85%
beta=0.99;      %input('ȷ��ͳ����ֵ���Ŷ�(0��1����')    %beta is the inspection level
k=0;
for i=1:size(LATENT,1)      %���ݷ������ȷ����Ԫ����k�����i����������������ķ�����ڶ�Ӧ������ֵ���� %% choose first k principal components
    alpha(i)=sum(LATENT(1:i))/sum(LATENT);
    if alpha(i)>=percent  
        k=i;
        break;  
    end 
end

P=COEFF(:,1:k);               %PΪ���ɾ���ͶӰ����

%% ����t2, SPEͳ����
for i=1:size(X,1)
    t2(i)=X(i,:)*P*inv(diag(LATENT(1:k)))*P'*X(i,:)'; %����t2ͳ����   
    SPE(i)=X(i,:)*COEFF*(X(i,:)*COEFF)'-X(i,:)*P*(X(i,:)*P)';
end

%% t2ͳ�����Ŀ����� ����t2�ֲ����㣩
T2knbeta=k*(size(X,1)-1)*(size(X,1)+1)/(size(X,1)-k)/size(X,1)*finv(beta,k,(size(X,1)-k)); %t2ͳ�����Ŀ�����

%% speͳ�����Ŀ�����
a=sum(SPE)/size(SPE,2);
b=var(SPE);
SPEbeta=b/(2*a)*chi2inv(beta,2*a^2/b);

%% ����������ݵ�t2,SPEͳ���� ��happenǰ���������������쳣��
for i=1:size(XT,1);
    XTt2(i)=XT(i,:)*P*inv(diag(LATENT(1:k)))*P'*XT(i,:)';
    XTSPE(i)=XT(i,:)*COEFF*(XT(i,:)*COEFF)'-XT(i,:)*P*(XT(i,:)*P)';
end

%% ��ͼ��ʾ���
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

%% False alarm rate �龯�ʣ����ʣ�
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

%% Miss rate ©����
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

%% �������������������������������
 disp('----PCA--False alarm rate----');
falserate_pca=[falserate_pca_T2 falserate_pca_SPE]
 disp('----PCA--Miss  rate----');
missrate_pca=[missrate_pca_T2 missrate_pca_SPE]
