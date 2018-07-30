% Demonstrate geometric based jacobian solution
%   jac_qn is computed using RVC toolbox jacob0
%   jac_g  is computed using numerically using the geometric based solution
%
% Reference: 
%   - http://ttuadvancedrobotics.wikidot.com/the-jacobian
%   - RVC toolbox
% 
% 2018-07-29
% Zihan Chen 


clc; clear;

mdl_puma560;
jac_qn = p560.jacob0(qn);


% geometry based solution
[Te, Tall]= p560.fkine(qn);
pe = Te(1:3, 4);

% loop through all robot links
% For revolute joint
%   Ji = [(z(i-1) x (pe - p(i-1)); z(i-1)]
% For prismatic joint
%   Ji = [z(i-1); zeros(3,1)]
jac_g = zeros(6, p560.n);
Tp = p560.base;
for i = 1:p560.n
    % Tp = transform(i-1)
    % pp = position previous, i-1
    % zp = z axis previous, i-1
    pp = Tp(1:3,4);
    zp = Tp(1:3,3);
    jac_g(:,i) = [cross(zp, pe-pp); zp];
    Tp = Tall(:,:,i);
end

% compare jac_qn and jac_g
jac_diff = jac_g - jac_qn;
jac_diff_sum = sum(abs(jac_diff(:)));
disp(['sum(jac_diff) = ' num2str(jac_diff_sum)])



