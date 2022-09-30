function [V,gap,acc]=autosc_gd(X,k,theta,lambda,Label)
if isempty(theta)
    theta=[5:15];
end
if isempty(lambda)
    lambda=[0.01 0.1 1];
end  
theta
lambda
X=normalizeL2(X);
V=[];
disp('Linear regression...')
[gap{1},V_L]=compute_gap(X,k,lambda,theta,'linear');
disp('Kernel regression...')
[gap{2},V_K]=compute_gap(X,k,lambda,theta,'kernel');
mg_1=max(gap{1}(:));
mg_2=max(gap{2}(:));
if mg_1>mg_2
    disp('Use linear regression!')
    [i,j]=find(gap{1}==mg_1);
    z=real(normalizeL2(V_L{i}{j}')');
    disp(['lambda=' num2str(lambda(i))])
    disp(['theta=' num2str(theta(j))])
else
    disp('Use kernel regression!')
    [i,j]=find(gap{2}==mg_2);
    z=real(normalizeL2(V_K{i}{j}')');
    disp(['lambda=' num2str(lambda(i))])
    disp(['theta=' num2str(theta(j))])
end
disp('Running k-means...')
L = kmeans(z,k,'maxiter',1000,'replicates',20,'EmptyAction','singleton');
L=bestMap(Label(:),L(:));
acc=cluster_accuracy(Label,L);
end
%%
function [gap,V]=compute_gap(X,k,lambda,theta,method)
[d,n]=size(X);
switch method
    case 'linear'
        XX=X'*X;
        XX2=X*X';
    case 'kernel'
        c=1;
        XX=sum(X.*X,1);
        dist=repmat(XX,n,1) + repmat(XX',1,n) - 2*X'*X;
        sigma2=(mean(real(dist(:).^0.5))*c)^2;
        KK=kernel(X,X,sigma2);
        if n>=5000
            [VK,SK,~] = rsvd(KK,k*30);
            VS=VK*diag(diag(SK).^0.5);
            VV=VS'*VS;
        end
end
for u=1:length(lambda)
    disp(['lambda=' num2str(lambda(u))])
    switch method
        case 'linear'
            if n<5*d
                S=inv(XX+lambda(u)*eye(n))*XX; 
            else
                S=1/lambda(u)*X'*inv(eye(d)+1/lambda(u)*XX2)*X;
            end
        case 'kernel'
            if n<5000
            S=inv(KK+lambda(u)*eye(n))*KK; 
            else
            S=1/lambda(u)*VS*inv(eye(size(VS,2))+1/lambda(u)*VV)*VS';
            end
    end
S=abs(S-diag(diag(S)));
[~,ids]=sort(S,'descend');
for i=1:length(theta)
    idt=ids(1:theta(i),:)+repmat([0:n:n*(n-1)],theta(i),1);
    A=zeros(n,n);
    A(idt)=S(idt);
    A=A./repmat(sum(A),n,1);
    A=(A+A')/2;
    [E,D]=solve_eig(A,k+1);
    Vt{i}=E(:,1:k);
    gap(u,i)=(D(k+1)-mean(D(1:k)))/(1e-6+mean(D(1:k)));%
%     gap(u,i)=D(k+1)-D(k);%
end
V{u}=Vt;
end
end
%%
function [E,D]=solve_eig(A,k)
n=size(A,1);
% D=diag(sum(A).^(-0.5));
% L=eye(n)-D*A*D;
% [~,D,E]=svd(L);
% [D,idx]=sort(diag(D),'ascend');
% E=E(:,idx);
Temp=sparse(repmat(sum(A).^(-1),n,1)'.*A);
[E, D] = eigs(Temp, k+1, 'LR' );
[D,idx]=sort(diag(D),'descend');
D=1-D;
E=E(:,idx);
%     L=diag(sum(A))-A;
%     [E, D] = eigs(sparse(eye(n)-L), k+1, 'LR' );
%     [D,idx]=sort(diag(D),'descend');
%     D=1-D;
%     E=E(:,idx);
end