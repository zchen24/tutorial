function [qt, qdt, qddt] = cubictraj(q0, qf, qd0, qdf)

% say Q0, QF, QD0, QDF are scalers

% q = a0 + a1*t1 + a2*t2 + a3*t3
%
% q0 = a0 
% qf = a0 + a1*t1 + a2*t2 + a3*t3
% qd0 = a1
% qf0 = a1 + 2*a2*t + 3*a3*t2

a0 = q0;
a1 = qd0;
a2 = 0;
a3 = 0;


% qf = q0 + qd0*t + a2*t2 + a3*t3
% qf0 = qd0 + 2*a2*t + 

t = 0:0.1:1;
t = t(:);
tt = [t.^3 t.^2 t ones(size(t))];
c = [a3 a2 a1 a0]';
qt = tt * c;

if nargout >= 2
	% qdt = 3*a3*t^3 + 2*a2*t^2 + a1
	cd = [3*a3 2*a2 a1 zeros(size(t))];
	qdt = tt * cd;
end

if nargout == 3
	% qddt = 9*a3*t^2 + 4*a2*t
	cdd = [zeros(size(t)) 9*a3 4*a2 zeros(size(t))];
	qddt = tt * cdd;
end

end