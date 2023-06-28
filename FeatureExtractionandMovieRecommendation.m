clear all
clc

Data=importdata('ratings.csv').data;% importing dataset
Data= Data(randperm(size(Data, 1)), :);% shuffling dataset

% dividing the data to training and testing
DataTesting=Data(round(4*size(Data,1)/5):size(Data,1),:);% Testing data
Data=Data(1:round(4*size(Data,1)/5),:); %Training data

% getting the users and movies
Users=Data(:,1);
Movies=Data(:,2);

% what users and movies present in our data
Usersused=unique(sort(Users));
Moviesused=unique(sort(Movies));

Moviesindex=(1:size(Moviesused,1))';% defining an ID for each used movie

m=size(Usersused,1);
n=size(Moviesused,1);

iterations=1000000; %number of iterations
r=9; %number of features
alpha=0.03; %step size
lambda=0.1; %regularization weight

U=rand(m,r);
V=rand(n,r);

plotindex=0;

for i=1:iterations

%generating the stochastic index
stochasticindex=round((size(Data,1)-1)*rand())+1;
iindex=Users(stochasticindex);
jindex=find(Moviesused==Data(stochasticindex,2));

% alpha=1/sqrt(0.5*i);
%update step
U(iindex,:)=U(iindex,:)-alpha*((U(iindex,:)*V(jindex,:)'-Data(stochasticindex,3))*V(jindex,:)+lambda*U(iindex,:));
V(jindex,:)=V(jindex,:)-alpha*((U(iindex,:)*V(jindex,:)'-Data(stochasticindex,3))*U(iindex,:)+lambda*V(jindex,:));

% calculating error
if(i==1 || rem(i,10000)==0)
    plotindex=plotindex+1;
    error=[];
    for j=1:size(Data,1)
        iindex=Users(j);
        jindex=find(Moviesused==Data(j,2));
        error=[error (U(iindex,:)*V(jindex,:)'-Data(j,3))^2];
    end
    RMSE(plotindex)=sqrt((1/size(Data,1))*sum(error));
end

end


plot(RMSE)

%create the prediction M matrix
Mpred=zeros(m,n);
for i=1:m
    for j=1:n
        Mpred(i,j)=U(i,:)*V(j,:)';
    end
end
%end

%create the real M matrix - zeros mean no data (in the actual data, their is no zeros)
Mreal=zeros(m,n);
Mvec=[];
for i=1:size(Data,1)
    iindex=Users(i);
    jindex=find(Moviesused==Data(i,2));
    Mreal(iindex,jindex)=Data(i,3);
    Mvec=[Mvec Mpred(iindex,jindex)];
end
%end

%feature extraction
[GC,GR] = groupcounts(Data(:,2));
[mt20]=find(GC>=20);
vec=ismember(Data(:,2), GR(mt20) );

ind=[];
for i=1:size(vec)
    if(vec(i)==1)
    ind=[ind i];
    end
end
Dataabove20=Data(ind,:);

jindex=[];
for s=1:size(ind,2)
jindex=[jindex find(Moviesused==Data(ind(s),2))];
end
jindex=unique(jindex);
Vabove20=V(jindex,:);

feature=5;
Vabove20sorted=sortrows(Vabove20,[-feature]);
list=[];
for i=1:20
list=[list find(V(:,feature)==Vabove20sorted(i,feature))];
end
Moviesused(list)

% Recommending Similar Movies
similarity=[];
firstmovie=20;
for i=1:size(Vabove20,1)
similarity=[similarity (Vabove20(firstmovie,:)*Vabove20(i,:)')/(norm(Vabove20(firstmovie,:))*norm(Vabove20(i,:)))];
end
similaritysorted=sort(similarity,'descend');
similarindex=[];
for i=1:6
    similarindex=[similarindex find(similarity==similaritysorted(i))];
end
similarlist=[];
for i=1:size(similarindex,2)
similarlist=[similarlist find(V(:,1)==Vabove20(similarindex(i),1))];
end

Moviesused(similarlist)


   

