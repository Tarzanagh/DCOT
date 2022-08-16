function H = myconstruct_graphL(tsize,VSet,Rate,nbs,Affinity)
L1=kron([Affinity(1),Affinity(2)]);
% L2=kron([Affinity(3),Affinity(4)]);
% idx = randperm(size(L1,1));
% [S1,D]=rnys(diag(sum(L1,2))-L1,5, 200,idx);
% idx = randperm(size(L2,1));
% [S2,D]=rnys(diag(sum(L2,2))-L2,15, 1000,idx);
% S=kron(S1,S2);
% H = [];
H{1,1} = reshape(diag(sum(L1,2))-L1,[tsize(1:2),size(L1,2)]);
 %disp(['Construction of the ',int2str(m),'-th sub-graph completed...']);
end
