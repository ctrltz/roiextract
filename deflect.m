function [w] = deflect(P, I, L, C, lambda)
%DEFLECT
    if isempty(C)
        C = eye(size(L, 1));
    end

    S = L * L' + lambda * C;
    w = I * inv(P' * inv(S) * P) * P' * inv(S);
end

