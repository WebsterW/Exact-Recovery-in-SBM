

function [x, iter, val_collector] = GPM(A, x, opts)

        %%  GPM solves regularized MLE
        % --- INPUT ---
        % A:    adjacency matrix (a sparse 0-1 matrix). 
        % x:    starting point
        % xt:   ground truth
        % opts.rho:   regularizer parameter
        % opts.tol:    desired tolerance for suboptimality
        % opt.T:    maximal number of iterations
        % opt.report_interval (optinal):    number of iterations before we report the progress (default: 100)   
        % opt.quiet (optinal):    be quiet or not.
        
        % --- OUTPUT ---
        % x:    returned clusters by GPM
        % iter:  terminal number of iteration
        % val_collector: function value of each iteration
        % dist_iter: gap between iterate and ground truth at each iteration
                
       %% Parameter setting
        rho = opts.rho;  
        maxiter = opts.T; 
        tol = opts.tol; 
        if isfield(opts,'report_interval')
            report_interval = opts.report_interval;
        else
            report_interval = 1;  
        end
        if isfield(opts,'quiet')
            quiet = opts.quiet;
        else
            quiet = false;
        end
        n = size(A, 1);
        Ax = A*x; %%% matrix-vector product
        fval = -x'*Ax +  rho * (ones(n,1)'*x)^2;  %%% compute function value
        val_collector(1) = fval;  
        
        for iter = 1:maxiter
                 
                xold = x;
               %% Check fixed-point condition
                x1 = x + Ax - rho*ones(n,1)'*x*ones(n,1);  x1(x1>=0) = 1; x1(x1<0) = -1; 
                dist = norm(x - x1);
                
               %% update of GPM + PGD                
                if iter <= floor(0.5*log(n)/log(log(n)))
                       x = Ax - rho*ones(n,1)'*x*ones(n,1); x = x/norm(x)*sqrt(n);          %%%% power iteration
                else     
                       x = Ax - rho*ones(n,1)'*x*ones(n,1);
                       x( x >= 0) = 1;                                     %%%%%%%% GPM iteration     
                       x( x < 0) = -1;
                end
                
                Ax = A*x; 
                fval = -x'*Ax + rho * (ones(n,1)'*x)^2;                 
                               
                if mod(iter, report_interval) == 0 && ~quiet            
                    fprintf('iternum: %2d, suboptimality: %8.4e, fval: %.3f \n', iter, dist, fval) 
                end
                                
                val_collector(iter+1) = fval; 
                
               %%  stopping criterion
                if norm(x-xold) <= tol 
                        break;
                end                
                
        end
        

end