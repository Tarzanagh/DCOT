function [Z] = optimize_l1Z(Z,V,X,Y,ita,gamma)
%% Initialization
normvv=0;
N = numel(V);
for i = 1 : N
    VVt{i} = V{i}*V{i}';
    normvv=normvv+norm(VVt{i});
end
r = TensorChainProductT(Z,V,1:N)-(X+Y/ita);
r = -TensorChainProduct(r,V,1:N);
Z=prox_l1(Z+1/normvv *r,gamma);

