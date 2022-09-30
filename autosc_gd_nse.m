function [V,gap,acc,NN,DD]=autosc_gd_nse(X,k,theta,lambda,Label,ns,opt)
if isempty(theta)
    theta=[5:15];
end
if isempty(lambda)
    lambda=[0.01 0.1 1];
end
theta
lambda
n=size(X,2);
X=normalizeL2(X);
switch opt.sel_type
    case 'random'
        disp(['Select landmark data points randomly...'])
        ids=sort(randperm(n,ns),'ascend');
        Xs=X(:,ids);
    case 'k-means'
        disp(['Select landmark data points by k-means...'])
        [id,C,~,dist]=kmeans(X',ns,'Distance','cosine','Replicates',1);
        Xs=C';
end
V=[];
disp('Linear regression...')
[gap{1},V_L,DD{1}]=compute_gap(Xs,k,lambda,theta,'linear');
disp('Kernel regression...')
[gap{2},V_K,DD{2}]=compute_gap(Xs,k,lambda,theta,'kernel');
mg_1=max(gap{1}(:));
mg_2=max(gap{2}(:));
if mg_1>mg_2
    disp('Use linear regression!')
    [i,j]=find(gap{1}==mg_1);
    Zs=real(normalizeL2(V_L{i}{j}')');
    disp(['lambda=' num2str(lambda(i))])
    disp(['theta=' num2str(theta(j))])
else
    disp('Use kernel regression!')
    [i,j]=find(gap{2}==mg_2);
    Zs=real(normalizeL2(V_K{i}{j}')');
    disp(['lambda=' num2str(lambda(i))])
    disp(['theta=' num2str(theta(j))])
end
%
disp('Neural Sparse Embedding...')
[Z,NN]=PSE(X,Xs,Zs',k,opt);
Z=normalizeL2(Z')';
disp('Running k-means...')
L = kmeans(Z,k,'maxiter',1000,'replicates',50,'EmptyAction','singleton');
L=bestMap(Label(:),L(:));
acc=cluster_accuracy(Label,L);
end
%%
function [gap,V,DD]=compute_gap(X,k,lambda,theta,method)
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
    DD(i,:,u)=D;
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
%%
function [Z,NN]=PSE(X,Xs,Zs,k,options)
if isfield(options,'lambda');lambda=options.lambda;else lambda=0.01;end
if isfield(options,'maxiter');maxiter=options.maxiter;else maxiter=500;end
if isfield(options,'hidden_layers_structure');hidden_layers_structure=options.hidden_layers_structure;else hidden_layers_structure=[size(X,1)*k];end
if isfield(options,'activation_function');activation_function=options.activation_function;else activation_function='relu'; end
%if isfield(options,'maxiter_adam');maxiter_adam=options.maxiter_adam;else maxiter_adam=5000;end
if isfield(options,'adam_stepsize');adam_stepsize=options.adam_stepsize;else adam_stepsize=1e-2;end
if isfield(options,'weight_decay') weight_decay=options.weight_decay;else weight_decay=1e-4;end
if isfield(options,'epoch') epoch=options.epoch;else epoch=1000;end
if isfield(options,'batch_size') batch_size=options.batch_size;else batch_size=128;end
NN=[];
net_structure=[size(X,1) hidden_layers_structure size(Zs,1)]
for i=1:length(hidden_layers_structure)
    activation_functions{i}=activation_function;
end
activation_functions{i+1}='linear';
activation_functions
NN=nn_fc(Xs',Zs',net_structure,activation_functions,weight_decay,adam_stepsize,batch_size,epoch);
NN.X=X';NN.n=size(X,2);
disp('Regression ...')
NN=nn_ff(NN);
Z=NN.a{end};
end
%%
function NN=nn_ff(NN)
L=length(NN.nns);
X=[ones(NN.n,1) NN.X];
L=length(NN.nns);
NN.a{1}=X; 
for i=2:L
    switch NN.activation_func{i-1}
        case 'sigm'
            NN.a{i}=sigm(NN.a{i-1}*NN.W{i-1}');
        case 'tanh_opt'
            NN.a{i}=tanh_opt(NN.a{i-1}*NN.W{i-1}');
        case 'linear'
            NN.a{i}=NN.a{i-1}*NN.W{i-1}';
        case 'relu'
            NN.a{i}=max(NN.a{i-1}*NN.W{i-1}',0);
    end
    if i<L
        NN.a{i}=[ones(NN.n,1) NN.a{i}];
    end
end

%
end