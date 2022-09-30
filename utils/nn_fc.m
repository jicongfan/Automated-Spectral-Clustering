function NN=nn_fc(X,Y,nns,act_func,Wp,alpha,batch_size,epoch)
% a simple fully-connected neural network
% X:n by d
% Y:n by m
[n,m]=size(Y);
d=size(X,2);
NN=nn_setup(nns,act_func);
NN.nns=nns;
NN.X=X;
NN.m=m;
NN.n=n;
NN.d=d;
NN.Wp=Wp;% weight decay
NN.Y=Y;
disp('Training neural networks for regression ......')
% NN=nn_adam(NN,alpha,epoch);
NN=nn_adam_minibatch(NN,alpha,batch_size,epoch);
end
%%
function NN=nn_adam(NN,alpha,maxiter)         
opt.alpha=alpha;
opt.maxiter=maxiter; 
w=[];
for i=1:length(NN.W)
    w=[w;NN.W{i}(:)];
end
[J,w]=opt_Adam(@nn_fg,w,NN,opt);
t=1;
for i=1:length(NN.W)
    [a,b]=size(NN.W{i});
    NN.W{i}=reshape(w(t:t+a*b-1),a,b);
    t=t+a*b;
end
NN=nn_ff(NN);
end
%%
function NN=nn_adam_minibatch(NN,alpha,batch_size,epoch)         
w=[];
X=NN.X;
Y=NN.Y;
n=size(X,1);
for i=1:length(NN.W)
    w=[w;NN.W{i}(:)];
end
beta_1=0.9;% 0.9
beta_2=0.999;%0.999
e=1e-8;
m=0;
v=0;
t=0;
beta_1t=beta_1;
beta_2t=beta_2;
for i=1:epoch
    id=randperm(n,n);
    Xr=X(id,:);
    Yr=Y(id,:);
    batch=ceil(n/batch_size);
    for j=1:batch
        if j<batch
            Xb=Xr(batch_size*(j-1)+1:batch_size*j,:);
            Yb=Yr(batch_size*(j-1)+1:batch_size*j,:);
        else
            Xb=Xr(batch_size*(j-1)+1:end,:);
            Yb=Yr(batch_size*(j-1)+1:end,:);
        end
        NN.X=Xb;
        NN.Y=Yb;
        NN.n=size(Xb,1);
        t=t+1;
        [J(t),g] = feval(@nn_fg,w,NN);
        m=beta_1.*m+(1-beta_1).*g;
        v=beta_2.*v+(1-beta_2).*(g.^2);
        mh=m./(1-beta_1t);
        vh=v./(1-beta_2t);
        beta_1t=beta_1t*beta_1;
        beta_2t=beta_2t*beta_2;
        w_new=w-alpha*mh./(vh.^0.5+e);
        if max(abs(w_new-w))<1e-6
            break;
        end
        w=w_new;
    end
    if mod(i,20)==0||i==1||i==10
        disp(['epoch=' num2str(i)  '/' num2str(epoch) '  fun_val=' num2str(J(t)) '  alpha=' num2str(alpha)])
    end
    %alpha=alpha*0.999;
end
t=1;
for i=1:length(NN.W)
    [a,b]=size(NN.W{i});
    NN.W{i}=reshape(w(t:t+a*b-1),a,b);
    t=t+a*b;
end
NN=nn_ff(NN);
end
%% f, dW, dZ
function [f,g]=nn_fg(w,NN)
    t=1;
    for i=1:length(NN.W)
        [a,b]=size(NN.W{i});
        NN.W{i}=reshape(w(t:t+a*b-1),a,b);
        t=t+a*b;
    end
    NN = nn_ff(NN);
    NN = nn_bp(NN);
    g=[];
    sum_w=0;
    for i=1:length(NN.W)
        wt=NN.W{i}(:,2:end);
        sum_w=sum_w+sum(wt(:).^2);
        dW=NN.dW{i}+NN.Wp*[zeros(size(NN.W{i},1),1) NN.W{i}(:,2:end)];
        g=[g;dW(:)];
    end
    f=NN.loss+0.5*NN.Wp*sum_w;
end


%% 
function NN=nn_setup(s,activation_func)
if length(activation_func)~=(length(s)-1)
    error('The number of layers does not match the number of activation functions!')
end
NN.activation_func=activation_func;
NN.layer=length(s)-1;
for i=1:NN.layer
    NN.W{i}=(rand(s(i+1),s(i)+1)-0.5)*2*4*sqrt(6/(s(i+1)+s(i)));
end
end
%%
function NN=nn_ff(NN)
L=length(NN.nns);
X=[ones(NN.n,1) NN.X];% add bias 1
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

% pedictive error and value of loss function
NN.e=NN.Y-NN.a{L};
switch NN.activation_func{end}
    case {'sigm','relu', 'linear','tanh_opt'}
        NN.loss=1/2*sum(sum(NN.e.^2))/NN.n;
%     case 'softmax'
%         NN.loss=-sum(sum(Y.* log(NN.a{L})))/NN.n;
end
%
end
%%
function NN=nn_bp(NN)
L=length(NN.nns);
switch NN.activation_func{end}
    case 'sigm'
        d{L}=-NN.e.*(NN.a{L}.*(1-NN.a{L}));
    case 'tanh_opt'
        d{L}=-NN.e.*(1.7159*2/3 *(1-1/(1.7159)^2*NN.a{L}.^2));
    case {'softmax','linear'}
        d{L}=-NN.e;
    case 'relu'
        d{L}=-NN.e;
end
for i=L-1:-1:2
    switch NN.activation_func{i-1}
        case 'sigm'
            d_act=NN.a{i}.*(1-NN.a{i});
        case 'linear'
            d_act=1;
        case 'tanh_opt'
            d_act=1.7159*2/3 *(1-1/(1.7159)^2*NN.a{i}.^2);
        case 'relu'
            d_act=max(sign(NN.a{i}),0);
    end
    if i+1==L % in this case in d{n} there is not the bias term to be removed             
        d{i} = (d{i + 1} * NN.W{i}).* d_act; % Bishop (5.56)
    else % in this case in d{i} the bias term has to be removed
        d{i} = (d{i + 1}(:,2:end) * NN.W{i}) .* d_act;
    end
end
%
for i = 1:(L - 1)
    if i+1==L
        NN.dW{i} = (d{i + 1}' * NN.a{i}) / size(d{i + 1}, 1);
    else
        NN.dW{i} = (d{i + 1}(:,2:end)' * NN.a{i}) / size(d{i + 1}, 1);      
    end
end
end

