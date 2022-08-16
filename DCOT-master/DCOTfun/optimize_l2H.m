function Z = optimize_l2H(H,V,X,Y,ita,gamma)

wt = 1;
N = numel(V);

for i = ndims(X) : -1 : N+1
    wt = ones(size(X,i),1)*wt';
    wt = wt(:);
end
for i = N : -1 : 1
    [U,S] = eig(V{i}*V{i}');
    Q{i} = S;
    P{i} = U;
    wt = diag(S)*wt';
    wt = wt(:);
end
wt = reshape(ita*wt+2*gamma,size(H,2), size(H,4));  % diagonal vector of (ita*kron(S_1,..S_n) + 2*gamma*I)^(-1)
Z = TensorChainProduct(ita*X+Y,V,1:N);
Z = TensorChainProductT(Z,P,1:N);
Z = Z./wt;
Z = TensorChainProduct(Z,P,1:N);

