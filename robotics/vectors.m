% Robot vectors 
%   cross product and skew symmetric matrix
%
% Reference: 
%   - RVC toolbox
% 
% 2018-07-30
% Zihan Chen


% skew symmetric matrix from a vector 
v1 = [1 2 3].';
sv1 = skew(v1);

% ------------------
% cross product 
% ------------------
% v3 = v1 x v2
v2 = [2 2 2].';
v3 = cross(v1, v2);

% Property 1: v2 x v1 = - v1 x v2
v3_v2xv1 = cross(v2, v1);
assert(all((v3 + v3_v2xv1) == [0; 0; 0]));

% Property2: v3 = skew(v1) * v2
% we can use skew symmetric matrix to compute cross product
v3_skew = skew(v1) * v2;

% Combine Property 1 & 2
% v3 = v1 x v2 = skew(-v2) * v1
v3_p3 = skew(-v2) * v1;
assert(all(v3 == v3_p3));





