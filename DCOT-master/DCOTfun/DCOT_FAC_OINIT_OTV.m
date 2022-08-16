function [V,Core,info,X,itr] = DCOT_FAC_OINIT_OTV(X,mark,para_ST,Xg,src_type,B_SVD)
%% Simultaneous Tensor Decomposition and Completion
% Input:
%  -- X: an input N-th order tensor object
%  -- mark: boolean index of missing entries (1 indicates missing; 0 otherwise)
%  -- para_ST: parameters used for the proposed STDC algorithm
%  -- mode: determining whether the N-th submanifold (V_N, usually ignored in multilinear model analysis) is computed (set as 0) or not (set as 1)
%  -- Xg: the ground truth of X; if not available, just using X instead

%% Initialization
tic;
tsize = size(X);
% initialize manifold graphs
if ~isfield(para_ST,'VSet')
    para_ST.H = [];
else
    para_ST.H = construct_graphL(tsize,para_ST.VSet,para_ST.Rate,para_ST.gnns,para_ST.Affinity);
    for i = 1 : size(para_ST.H,1)
        if size(para_ST.H,2)==1 || numel(para_ST.H{i,2})==0
            for j = 1 : numel(tsize)
                para_ST.Ds{i,j} = eye(tsize(j));
            end
        else
            for j = 1 : numel(tsize)
                A = randn(tsize(j)); 
                B = imresize(A,[tsize(j) round(tsize(j)*para_ST.Rate(i))],'bilinear');
                para_ST.Ds{i,j} = A\B;
            end
        end
    end
end
% initilize factor matrices & permutation matrices (w.r.t the modes of tensor dimension & PoM)
vsize = tsize;
N = numel(tsize);
if para_ST.mode_dim, N = N-1; end;
if para_ST.mode_PoM
    randn('seed',1);
    for i = 1 : N
        V{i} = randn(tsize(i));
        V{i} = V{i}/norm(V{i});
        vsize(i) = size(V{i},1);
    end
    Xt = HaLRTC(X,mark,ones(N,1),10^-2,1.1,100,N,Xg);
    P = initial_PoM(Xt,para_ST.gnns^2,para_ST.pnns,N);
else
    %[V,Z] = mlsvd(X,[10,10,10]);
    %V=V(1:3);
    for i = 1 : N
        V{i} = eye(tsize(i));
        %V{i} = V{i}/norm(V{i});
        V{i}=V{i}';
        vsize(i) = size(V{i},1);
        P{i} = (1:tsize(i))';
    end
end

% initialize core tensor and augmented multiplier
Z =TensorChainProduct(X,V,1:N);
H =Z;
for i=1:size(Z,1)
    for j=1:size(Z,2)
      for kk=1: size(Z,3)
          H(i,j,kk,:)=  X(i,1,1,1);
      end
    end
end
%H=H/norm(H(:));
Y = zeros(tsize);
S = zeros(tsize);
% initialize algorithm parameters
norm_gt = norm(Xg(:));
norm_x = norm(X(:));
para_ST.alpha = ones(N,1);
para_ST.gamma = (para_ST.omega/para_ST.tau)/(norm_x^2);
xxt = reshape(X,tsize(1),[]);
xxt = norm(xxt*xxt');
ita = 1/(para_ST.tau*xxt);
ct = zeros(1,N);
for i = 1 : size(para_ST.H,1)
    ct = ct+double(para_ST.VSet{i});
end
for i = 1 : size(para_ST.H,1)
    list = 1 : N;
    list = list(para_ST.VSet{i});
    for j = 1 : size(para_ST.H,2)
        U{j} = para_ST.Ds{i,list(j)};
    end
    for j = 1 : size(para_ST.H,2)
        llt = reshape(TensorChainProduct(para_ST.H{i,j},U,[1:j-1 j+1:size(para_ST.H,2)]),tsize(list(j)),[]);
        llt = norm(llt*llt');
        para_ST.H{i,j} = para_ST.H{i,j}*para_ST.kappa*sqrt(ita*xxt/(2*llt*ct(list(j))));
    end
end
% message
disp(['Finish the initialization of all parameters within ',num2str(toc),' seconds...']);
disp('------------------------------------------------------------------------------');
disp('--                          Start DCOT algorithm..                          --');
disp('------------------------------------------------------------------------------');
%% Main algorithm
lambda = ita*(1.1^(para_ST.maxitr))/2;
tic;
switch src_type
    case 'image'
        figure('Position',get(0,'ScreenSize'));
        subplot(1,3,1);imshow3(abs(X), []);title('incomplete tensor');
    case 'CMU'
        figure('Position',get(0,'ScreenSize'));
        subplot(1,3,1);imshow(reshape(permute(reshape(TensorProduct(X(1,:,:,:),B_SVD,4),[11,21,32,32]),[3,2,4,1]),32*21,[]));title('incomplete tensor (1st subject)');
    otherwise
end
for itr = 1 : para_ST.maxitr
        
     % update V1,...,Vn
    [V,P,rank_vi,pstr] = optimize_otv_V(X-S,Y,Z,V,P,ita,tsize,vsize,para_ST);
    % update G    
    Xh=TensorChainProductT(H,V,1:numel(V));
    Xg=X-S-Xh;
    G = optimize_l1Z(Z,V,Xg,Y,ita,1e-5);
    %G=optimize_l2Z_CG(Z,V,Xg,Y,ita,para_ST.gamma);
    Xg=TensorChainProductT(G,V,1:numel(V));
    % update H
    Xh=X-S-Xg;
    Xhij=1/size(H,3)*squeeze(sum(Xh,3));
    Yij=1/size(H,3)*squeeze(sum(Y,3));
    for j=1:size(H,3)
        H(:,:,j)= optimize_visual_l2H(H,V(1:2),Xhij,Yij,ita,para_ST.gamma);
    end
    H=zeros(size(G));
    Z=G+H;
    % update X
    Xt = TensorChainProductT(Z,V,1:numel(V));
    % update S
    S = prox_l1(X-Xt+Y/ita, 1000/(max(tsize)*ita));
    %end
    X(mark) = Xt(mark)+S(mark)-Y(mark)/ita;
    if para_ST.mode_nse
        X(~mark) = ((ita*Xt(~mark)-Y(~mark))+lambda*Xg(~mark))/(ita+lambda);
    end
    residual = (norm(X(:)-Xt(:)-S(:)))/norm(Xt(:)+S(:));
    % update Y
    Y = Y+ita*(X-Xt-S);
    % assessment
    info.rse(itr) = norm(X(mark)-Xg(mark)-S(mark))/norm_gt;
    info.rank_vi(:,itr) = rank_vi;
    info.residual(:,itr) = residual;
    % display
    disp_t = ['DCOT completed at ',int2str(itr),'-th iteration step within ',num2str(toc),' seconds...'];%('%,pstr,' data are permuted)'];
    switch src_type
        case 'image'
            subplot(1,3,2);plot(info.rse);axis([0,para_ST.maxitr,0,inf]);title('# iterations vs. RSEs');
        %   subplot(1,3,1);imshow(S(:,:,1));title('completed tensor');
            subplot(1,3,3);    imshow3(abs(X), []);title('completed tensor');
            axes('position',[0,0,1,1],'visible','off');
            text(0.05,0.98,['### ',disp_t]);
            pause(0.1);
        case 'CMU'
            hold on
            subplot(1,3,2);plot(info.rse);axis([0,para_ST.maxitr,0,inf]);title('# iterations vs. RSEs');
            %subplot(1,3,1);imshow3(reshape(permute(reshape(TensorProduct(Z(1,:,:,:),B_SVD,4),[11,21,32,32]),[3,2,4,1]),32*21,[]));title('completed tensor (1st subject)');
            %subplot(1,3,2);imshow(reshape(permute(reshape(TensorProduct(X(1,:,:,:),B_SVD,4),[11,21,32,32]),[3,2,4,1]),32*21,[]));title('core tensor (1st subject)');
            subplot(1,3,3);imshow(reshape(permute(reshape(TensorProduct(X(1,:,:,:),B_SVD,4),[11,21,32,32]),[3,2,4,1]),32*21,[]));title('completed tensor (1st subject)');
            axes('position',[0,0,1,1],'visible','off');
            text(0.05,0.98,['### ',disp_t]);
            pause(0.1);
        otherwise
    end
    disp(disp_t);
    disp(disp_t);
    % stopping criterion
    %if residual<=0.01,% break; end;
       ita = ita*1.1;
end
Core = Z;
for i = 1 : N
    V{i} = V{i}';
end
info.P = P;
