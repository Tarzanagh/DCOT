function perf(T,logplot)
%PERF    Performace profiles
%
% PERF(T,logplot)-- produces a performace profile as described in
%   Benchmarking optimization software with performance profiles,
%   E.D. Dolan and J.J. More', 
%   Mathematical Programming, 91 (2002), 201--213.
% Each column of the matrix T defines the performance data for a solver.
% Failures on a given problem are represented by a NaN.
% The optional argument logplot is used to produce a 
% log (base 2) performance plot.
%
% This function is based on the perl script of Liz Dolan.
%
% Jorge J. More', June 2004

if (nargin < 2)
    logplot = 0;
end

colors  = ['r' 'b' 'k' 'g' 'y' 'r' 'c'];
lines   = ['-' '--' '-' '-.' '-' '--' '-' '-.''-'];
markers = ['s' '^' 'x' '*' 'v' '^' 'o'];%markers = {'+','o','*','.','x','s','d','^','v','>','<','p','h'} 

[np,ns] = size(T);

% Minimal performance per solver

minperf = min(T,[],2);

% Compute ratios and divide by smallest element in each row.

r = zeros(np,ns);
for p = 1: np
  r(p,:) = T(p,:)/minperf(p);
end

if (logplot)~= 0
    r= log2(r);
end

max_ratio = max(max(r));

% Replace all NaN's with twice the max_ratio and sort.

r(find(isnan(r))) = 2*max_ratio;
r = sort(r);

% Plot stair graphs with markers.

for s = 1: ns
 [xs,ys] = stairs(r(:,s),[1:np]/np);
 %option = [lines(s) colors(s) markers(s)];
 plot(xs,ys);
 hold on;
end
 set (gca,'FontName','Symbol');
xlabel ('\tau')
ylabel('\rho(\tau)')
legend( {'DCOT-N(H)', 'DCOT(H)', 'HOSVD', 'PCA', 'LPP', 'MFA', 'LDA'});
%legend( {'LDA','MFA','LPP','PCA', 'HOSVD','DCOT(H)','DCOT-N(H)'});
title('Fig. 1. Performance profile for IT','FontName','Helvetica')

% Axis properties are set so that failures are not shown,
% but with the max_ratio data points shown. This highlights
% the "flatline" effect.

axis([ 0 1.1*max_ratio 0 1 ]);

% Legends and title should be added.

