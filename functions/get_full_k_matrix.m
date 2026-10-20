function K=get_full_k_matrix(input)
    %this function returns a sample from a realization of a gaussian
    %process for learning the hyperparameters 

    sigma_f = 1.5;     % signal variance
    ell = 1.0;         % length scale
    ell2=1.9;
    rng(42);

    covfunc = {@covSEard, [log(ell);log(ell2); log(sigma_f)]};

    K = feval(covfunc{:}, input);

end