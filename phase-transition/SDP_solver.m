
function X = SDP_solver(A)
    
    %% Mosek solver in CVX solves SDP proposed by Hajek et al. (2016)
    % --- INPUT ---
    % A:   Adjacency matrix (a sparse 0-1 matrix). 

    % --- OUTPUT ---
    % X:   optimal cluster matrix


%     fprintf(' ***************************** SDP solved by Mosek *****************************\n');

    n = size(A,1);    
    cvx_solver mosek
    cvx_begin  sdp quiet    
    cvx_precision low
            variable X(n,n) semidefinite
            maximize sum(sum(A.*X))
            diag(X) == 1;    
            ones(n,1)'*X*ones(n,1) == 0
    cvx_end
    
    
end