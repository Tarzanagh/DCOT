function [V,P,rank_vi,pstr] = optimize_otv_nolap(X,Y,Z,V,P,ita,tsize,vsize,para_ST,itr)
%% Initialization
num_V = numel(V);
rank_vi = zeros(1,num_V);
pstr = '(';

%% Update by linearization
for i = 1 : num_V
     list = 1:num_V;
     list(i) = [];
    Bi = reshape(shiftdim(TensorChainProductT(Z,V,list),i-1),vsize(i),[]);
    Ci = -Bi*(reshape(shiftdim(Y,i-1),tsize(i),[])+ita*reshape(shiftdim(X,i-1),tsize(i),[]))';
    Bi = (Bi*Bi')*ita/2;
    Bt = Bi+Bi';
    sig = norm(Bt);
    %sig = norm(Bt);
    grad = Bt*V{i}+Ci;    
    %grad = Bt*V{i}+Ci;    
    if i<num_V
        grad = grad;%+1/32*(V{i}*V{i}'-V{i+1}*V{i+1}')*V{i};
            %+1/128*V{I}*(V{i}'*V{I}-Am{i+1}'*Am{i+1});
    else
        grad = grad;%+1/32*(V{i-1}*V{i-1}'-V{i}*V{i}')*V{i};
    end
    [Vi,r] = optimize_LRM(V{i},grad,sig,para_ST.alpha(i));
% 
    V{i} = Vi;
    if para_ST.mode_PoM
        [pidx,info,cpu_time] = PermutationOnManifolds(V{i},At,tsize(i),para_ST.pnns,'euclidean',false);
        P{i} = pidx(P{i});
    end
    rank_vi(i) = r;
    pstr = [pstr,int2str(sum((1:tsize(i))'~=P{i})),','];
end
pstr(end) = ')';

function [optimum,r] = optimize_LRM(Mk,grad,sig,alpha)
%% M = arg min { alpha*|| M ||_* + <grad(Mk),M-Mk> + (sig/2)*|| M-Mk ||_F }
optimum=Mk-grad/(sig);
r=size(Mk,1);
%  try
%     [U,S,V] = svd(Mk-grad/(sig),'econ');
%     S = diag(S);
%     [S,r] = shrinkage(S,alpha/(sig));
%     if r==0
%         optimum = Mk;
%     else
%         optimum = U*diag(S)*V';
%     end
% catch
%     disp('SVD fails to coverge...');
%     optimum = Mk;
%     r = 0;
% end


