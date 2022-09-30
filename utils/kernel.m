function [K,XY]=kernel(X,Y,sigma2)
nx=size(X,2);
ny=size(Y,2);
XY=X'*Y;
xx=sum(X.*X,1);
yy=sum(Y.*Y,1);
D=repmat(xx',1,ny) + repmat(yy,nx,1) - 2*XY;
K=exp(-D/2/sigma2); 
end