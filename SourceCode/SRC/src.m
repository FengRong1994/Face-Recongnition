function [predictions,src_scores,uniqlabels]=src(Traindata,Trainlabels,Testdata,sp_level)
%src
A=sparse_represent(Testdata,Traindata,sp_level);
uniqlabels=unique(Trainlabels);
c=max(size(uniqlabels));
for i=1:c
    R=Testdata-A(:,find(Trainlabels==uniqlabels(i)))*Traindata(find(Trainlabels==uniqlabels(i)),:);
    src_scores(:,i)=sqrt(sum(R.*R,2));
end
[maxval,indices]=min(src_scores');
predictions=uniqlabels(indices);
%sparse_represent
function [A]=sparse_represent(Test,Train,sp_level)

[n,p]=size(Train);
[k,pt]=size(Test);
if(p~=pt)
    sprintf('training data and test data must have the same dimensionality')
else
    X=Train;
    K_Tr=X*X';
    A=[];
    for i=1:k
        y=Test(i,:);
        K_te=X*y';
        K_y=y*y';
        [min_val,min_index]=min(K_te);
        a=sparse_represent_kernelized(K_y,K_Tr,K_te,sp_level);
        A=[A;a];
    end    
end
%sparse_represent_kernelized
function [a]=sparse_represent_kernelized(K_y,K_Tr,K_te,sp_level)


n=max(size(K_Tr));

;

N=sp_level*n;
cvx_begin
  cvx_quiet(true);
  variable a(1,n);
  minimize a*K_Tr*transpose(a)+K_y-transpose(K_te)*transpose(a)-a*K_te;
  norm(a,1)<=N;
cvx_end

