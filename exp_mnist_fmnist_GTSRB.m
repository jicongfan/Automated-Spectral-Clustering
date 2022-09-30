%%% This is the codes for runing AutoSC-GD on mnist, fashion-MNIST,
%%% (raw images or features) and GTSRB
%%% Due to the size limitation, we cannot include all datasets here.
clc
clear all
warning off
rng(1)
load('mnist.mat');%30MB
% load('fmnist.mat');%68MB
% load('mnist_fea_500.mat');%260MB
% load('fmnist_fea_500.mat');%260MB
% load('GTSRB_500.mat');
X=normalizeL2(X);
k=length(unique(Label));
%%
tau=[];
lambda=[];
tic
% NSE parameters
opt.epoch=200;
opt.sel_type='k-means';
opt.batch_size=128;
opt.adam_stepsize=0.001;
opt.hidden_layers_structure=[k*20];
opt.weight_decay=1e-5;
opt.activation_function='relu';
[V,gap,acc,NN]=autosc_gd_nse(X,k,tau,lambda,Label,1000,opt);
toc
acc