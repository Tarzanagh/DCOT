function [U] = Gram_Schmidt(U,orth_const)

N =size(U,2);
R=size(U{1},1);

Q = U{1};
t = size(Q);
J = t(2);
dim_max = t(1) -1 ;
for n = 1:N
    Q = U{1};
    t = size(Q);
    if t(1)-1 <dim_max
        dim_max = t(1) - 1;
    end
end

for n = 1:N
    Q = U{n};
    for i=1:J
        Q(:,i) = Q(:,i)/norm(Q(:,i));
        if i <= dim_max
            for j=i+1:J
                Q(:,j) = Q(:,j) - orth_const * (Q(:,j)'*Q(:,i))*Q(:,i);
            end
        end
    end
    U{n} = Q;
end
end