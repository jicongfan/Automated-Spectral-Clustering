%%% This is the codes for runing AutoSC-GD on YaleB face datasets
clc
clear all
warning off
rng(1)
load('YaleB_32x32.mat');
X=fea';
X=normalizeL2(X);
Label=gnd;
k=length(unique(Label));
%
tau=[];
lambda=[];
tic
[V,gap,acc]=autosc_gd(X,k,tau,lambda,Label);
toc
acc
