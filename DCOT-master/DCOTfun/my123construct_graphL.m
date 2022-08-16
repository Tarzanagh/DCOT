function H = my123construct_graphL(tsize,VSet,Rate,nbs,Affinity)
L1=kron([Affinity]);
idx = randperm(size(L1,1));
[S1,D]=rnys(diag(sum(L1,2))-L1,25,1000,idx);
%[S2,D]=rnys(diag(sum(L2,2))-L2,10, 500,idx);
S=S1;
H = [];
H{1,1} = reshape(S',[tsize(1:3),size(S,2)]);
 %disp(['Construction of the ',int2str(m),'-th sub-graph completed...']);
end