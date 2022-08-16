clear;clc;
addpath(genpath('.\'));
demo_type = input('Demo (1. image, 2. CMU):');
m_rate = 0.8;
p_rate = 1/2;
noise_std = 0.0;%0.25;
%%
switch demo_type
    case 1
        % load image
       %  [fname,pname] = uigetfile('*.*');
      %   X = double(imread([pname,fname]))/255;
        load('results_fig5.mat')
        X=squeeze(h_LLR(:,:,1,:,1));
      %  load('face.mat')
         %X=X/255;
        tsize = size(X);
        % reordering data or not
        order_type = input('Reordering (1. Yes, 2. No):');
        switch order_type
            case 1
                k = round(p_rate*size(X));
                for i = 1 : 2
                    ridx = randperm(size(X,i));
                    sidx = ridx(randperm(k(i)));
                    P{i} = 1:size(X,i);
                    Q{i} = P{i};
                    P{i}(ridx(1:k(i))) = P{i}(sidx);
                    Q{i}(P{i}) = Q{i};
                end
                X = X(P{1},P{2},:);
            case 2
            otherwise
                disp('Wrong selection...default setting is No.');
        end
        % construction of missing data
        rand('seed',0);%rng(0);
        idx = randperm(numel(X));
        mark = zeros(tsize);
        mark(idx(1:floor(m_rate*numel(X)))) = 1;
        mark = boolean(mark);
        Xm = X + noise_std*randn(size(X));
        Xm(mark) = 0;
      % parameters: kappa/omega/tau/gnns/pnns/maxitr/mode_dim/mode_PoM/mode_noise/VSet/Rate/Affinity
        para = initial_para(10^0.2,10^-0.8,0.1,2,25,50,true,false,noise_std~=0,{[1],[2]},[1,1],{[],[]},tsize);
        [~,~,info,Xs] = DCOT_FAC_OINIT_OTV(Xm,mark,para,X,'image',[]);
    case 2
        % load CMU-PIE data
        load('LabelRelation.mat');
        Affinity{2} = abs(acosd(W_CMU.view([1:6,8:11,13],[1:6,8:11,13])));
        Affinity{2} = exp(-(Affinity{2}.^2/var(Affinity{2}(:))))-eye(size(Affinity{2},1));
        Affinity{3} = abs(acosd(W_CMU.illu));
        Affinity{3} = exp(-(Affinity{3}.^2/var(Affinity{3}(:))))-eye(size(Affinity{3},1));
        load('CMU0(30x11x21x1024).mat','FullTensor');
        X = double(FullTensor)/255;
        clear FullTensor;
        T=reshape(X,30,[]);T=pdist(T);T=squareform(T);
        Affinity{1} = exp(-T/mean(T(:)))-eye(size(T,1));
        % construction of missing data
        tsize = size(X);
      %  rng shuffle
        idx = randperm(prod(tsize(1:end-1)));
        mark = zeros(tsize(1:end-1));
        mark(idx(1:floor(m_rate*numel(idx)))) = 1;
        mark = logical(mark);
        mark = repmat(mark,[1,1,1,tsize(end)]);
        Xm = X + noise_std*randn(size(X));
        Xm(mark) = 0;
        % dimension reduction
        [~,S,V] = svd(reshape(X,prod(tsize(1:end-1)),[]),'econ');
        V = V(:,(cumsum(diag(S))/sum(diag(S)))<0.95);
        X = TensorProduct(X,V',4);
        Xm = TensorProduct(Xm,V',4);
        mark = mark(:,:,:,1:size(V,2));
         T=reshape(X,size(V,2),[]);T=pdist(T);T=squareform(T);
         Affinity{4} = exp(-T/mean(T(:)))-eye(size(T,1));
%         Affinity{4} = abs(acosd(unfolding(X, 4)'));
%         Affinity{4} = exp(-(Affinity{4}.^2/var(Affinity{4}(:))))-eye(size(Affinity{4},1));
         % parameters: kappa/omega/tau/gnns/pnns/maxitr/mode_dim/mode_PoM/mode_noise/VSet/Rate/Affinity
        para = initial_para(10^1.5,10^-1.9,0.1,4,8,25,true,false,noise_std~=0.1,{[1,2,3]},[1],Affinity,tsize);
        [~,~,info,Xs2] = DCOT_CMU_OINIT_OTV_3(Xm,mark,para,X,'CMU',V);
     %   [~,~,info,Xs2] = STDC(Xm,mark,para,X,'CMU',V);
    otherwise
        disp('Wrong input!');
end
