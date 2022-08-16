function [V,Core,info,X,itr] = DCOT_CMU_OINIT_OTV_nolap(X,mark,para_ST,Xg,src_type,B_SVD)
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

% initilize factor matrices & permutation matrices (w.r.t the modes of tensor dimension & PoM)
vsize = tsize;
N = numel(tsize);
if para_ST.mode_dim, N = N-1; end;
%  [V,Z] = mlsvd(X,[11,11,11,100]);
%V=V(1:3);
for i = 1 : 3
    %V{i} = rand(5,tsize(i));
    V{i} = eye(tsize(i));
    %V{i} = V{i}/norm(V{i});
    % V{i}=V{i}';
    vsize(i) = size(V{i},1);
    P{i} = (1:tsize(i))';
end

% initialize core tensor and augmented multiplier
Z = TensorChainProduct(X,V,1:N);
H = Z;
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
        CC=reshape(permute(reshape(TensorProduct(Xg(1,:,:,:),B_SVD,4),[11,21,32,32]),[3,2,4,1]),32*21,[]);
        subplot(1,3,1);imshow(CC),title(' tensor (1st subject)');
        BB=reshape(permute(reshape(TensorProduct(X(1,:,:,:),B_SVD,4),[11,21,32,32]),[3,2,4,1]),32*21,[]);
        subplot(1,3,2);imshow(BB),title(' incomplete tensor (1st subject)');
    otherwise
end
for itr = 1 : para_ST.maxitr
    
    % update V1,...,Vn
    [V,P,rank_vi,pstr] = optimize_otv_nolap(X-S,Y,Z,V,P,ita,tsize,vsize,para_ST);
    % update G
    X_H=TensorChainProductT(H,V,1:numel(V));
    X_G=X-S-X_H;
    G = optimize_l1Z(Z,V,X_G,Y,ita,1e-3);
    X_G=TensorChainProductT(G,V,1:numel(V));
    % update H
    X_H=X-S-X_G;
    for i=1:size(H,1)
        Xhi=squeeze(X_H(i,:,:,:));
        Yi=squeeze(Y(i,:,:,:));
        for j=1:size(H,3)
            Xhij=1/11*squeeze(sum(Xhi,2));
            Yij=1/11*squeeze(sum(Yi,2));
            H(i,:,j,:)= optimize_l2H(H,V(2),Xhij,Yij,ita, para_ST.gamma);
        end
    end
    Z=G+H;
    % update X
    Xt = TensorChainProductT(Z,V,1:numel(V));
    % update S
    S = prox_l1(X-Xt+Y/ita, 10000/(max(tsize)*ita));
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
    %Scappe saddle
    %     if  itr>1&& info.residual(itr)>     info.residual(itr-1)
    %         Y= Y+ 1e-8*rand(size(Y));
    %     end
    % display
    disp_t = ['DCOT completed at ',int2str(itr),'-th iteration step', num2str(toc), ' seconds...'];%('%,pstr,' data are permuted)'];
    switch src_type
        case 'image'
            subplot(1,3,2);plot(info.rse);axis([0,para_ST.maxitr,0,inf]);title('# iterations vs. RSEs');
            %   subplot(1,3,1);imshow(S(:,:,1));title('completed tensor');
            subplot(1,3,3);    imshow(abs(X), []);title('completed tensor');
            axes('position',[0,0,1,1],'visible','off');
            text(0.05,0.98,['### ',disp_t]);
            pause(0.1);
        case 'CMU'
            hold on
            %subplot(1,3,1);plot(info.rse);axis([0,para_ST.maxitr,0,inf]);title('# iterations vs. RSEs');
            %subplot(1,3,1);imshow3(reshape(permute(reshape(TensorProduct(Z(1,:,:,:),B_SVD,4),[11,21,32,32]),[3,2,4,1]),32*21,[]));title('completed tensor (1st subject)');
            %subplot(1,3,2);imshow(reshape(permute(reshape(TensorProduct(X(1,:,:,:),B_SVD,4),[11,21,32,32]),[3,2,4,1]),32*21,[]));title('core tensor (1st subject)');
            AA=reshape(permute(reshape(TensorProduct(X(1,:,:,:),B_SVD,4),[11,21,32,32]),[3,2,4,1]),32*21,[]);
            subplot(1,3,3);imshow(AA);title('completed tensor (1st subject)');
            axes('position',[0,0,1,1],'visible','off');
            text(0.05,0.98,['### ',disp_t]);
            pause(0.1);
        otherwise
    end
    disp(disp_t);
    disp(disp_t);
    % stopping criterion
    if residual<=0.00001, break; end;
    ita = ita*1.2;
    
end
Core = Z;
for i = 1 : N
    V{i} = V{i}';
end
info.P = P;
