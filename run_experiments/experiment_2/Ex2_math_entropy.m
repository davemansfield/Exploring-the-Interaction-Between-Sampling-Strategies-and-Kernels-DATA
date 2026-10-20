%% Infotaxis sampling, all kernels, Environment 1
%
% Environment 1 (deterministic sin(x2)+x1 target).
% Runs the infotaxis acquisition function (domain-wide, posterior-
% corrected expected information gain) across all kernel configurations.
% Results saved as separate named variables per kernel,
% checkpointed to disk after every kernel.
%
% -------------------------------------------------------------------------

clear; close all;

N_kernels = 8;      % kernel indices n = 1..8 in cov_reset (Table 1)
N_hp_it   = 20;
N_run     = 50;
N_steps   = 100;
res       = 50;

xtestpoints = linspace(-5,5,res);
[Xtest1, Xtest2] = meshgrid(xtestpoints, xtestpoints);
xs = [Xtest1(:) Xtest2(:)];

f = @(x) sin(x(:,2)) + x(:,1);
target = f(xs);   % deterministic — same for every run/kernel

output_file = 'env1_entropy_results.mat';

for n = 1:N_kernels

    save_RMSE      = zeros(N_steps, N_run);
    save_timeopt   = zeros(N_steps, N_run);
    save_timestep  = zeros(N_steps, N_run);
    save_NMLL      = zeros(N_steps, N_run);
    save_ls1       = zeros(N_steps, N_run);
    save_ls2       = zeros(N_steps, N_run);

    for run = 1:N_run

        rng('shuffle')     % shared starting points/noise across kernels per replicate index
        xidx0 = randi([1, res^2], 2, 1);

        [cov, hyp, mean, lik] = cov_reset(n);

        xr = xs(xidx0,:);
        y  = target(xidx0(1));
        y2 = target(xidx0(2));
        y  = [y; y2];

        jitter = 1e-6;
        hyp.lik = log(exp(hyp.lik) + jitter);

        for i = 1:N_steps
            tstart_opt = tic;
            hyp = minimize(hyp, @gp, -N_hp_it, @infGaussLik, mean, cov, lik, xr, y);
            [mod, var] = gp(hyp, @infGaussLik, mean, cov, lik, xr, y, xs);
            tend_opt = toc(tstart_opt);

            NMLL = gp(hyp, @infGaussLik, mean, cov, lik, xr, y);

            tstart_step = tic;
            rangeR = 1;
            [idx, ~] = rangesearch(xs, xr(end,:), rangeR);
            idx = cell2mat(idx);
            xc = xs(idx,:);
            xcvar = var(idx);

            % --- domain-wide expected information gain (posterior-corrected) ---
            sn2 = exp(2*hyp.lik);
            Ntr = size(xr,1);

            Kxr    = feval(cov{:}, hyp.cov, xr);            % [Ntr x Ntr]
            Ky     = Kxr + sn2*eye(Ntr);
            Lc     = chol(Ky, 'lower');

            Kxs_xr = feval(cov{:}, hyp.cov, xs, xr);        % [2500 x Ntr]
            Kxc_xr = feval(cov{:}, hyp.cov, xc, xr);        % [Ncand x Ntr]

            V1 = Lc \ Kxs_xr';                              % [Ntr x 2500]
            V2 = Lc \ Kxc_xr';                              % [Ntr x Ncand]

            Kxc_prior = feval(cov{:}, hyp.cov, xs, xc);     % [2500 x Ncand]
            Kxc_post  = Kxc_prior - V1' * V2;

            denom   = xcvar' + sn2;
            dvar    = (Kxc_post.^2) ./ denom;
            var_new = max(var - dvar, 1e-12);
            IG      = 0.5 * sum(log(var) - log(var_new), 1);

            [~, max_idx] = max(IG);
            % ---------------------------------------------------------------------

            xnew = xc(max_idx,:);

            xr(end+1,:) = xnew;
            ynew = f(xnew) + 0.1*randn(1);
            y(end+1) = ynew;
            tend_step = toc(tstart_step);

            save_RMSE(i,run)     = rmse(target, mod);
            save_timeopt(i,run)  = tend_opt;
            save_timestep(i,run) = tend_step;
            save_NMLL(i,run)     = NMLL;
            save_ls1(i,run)      = hyp.cov(1);
            if numel(hyp.cov) > 1
                save_ls2(i,run)  = hyp.cov(2);
            end
        end
    end

    eval(sprintf('save_RMSE_cov%d_env1_entropy = save_RMSE;', n));
    eval(sprintf('save_timeopt_cov%d_env1_entropy = save_timeopt;', n));
    eval(sprintf('save_timestep_cov%d_env1_entropy = save_timestep;', n));
    eval(sprintf('save_NMLL_cov%d_env1_entropy = save_NMLL;', n));
    eval(sprintf('save_ls1_cov%d_env1_entropy = save_ls1;', n));
    eval(sprintf('save_ls2_cov%d_env1_entropy = save_ls2;', n));

    varnames = { sprintf('save_RMSE_cov%d_env1_entropy', n), ...
                 sprintf('save_timeopt_cov%d_env1_entropy', n), ...
                 sprintf('save_timestep_cov%d_env1_entropy', n), ...
                 sprintf('save_NMLL_cov%d_env1_entropy', n), ...
                 sprintf('save_ls1_cov%d_env1_entropy', n), ...
                 sprintf('save_ls2_cov%d_env1_entropy', n) };

    if n == 1
        save(output_file, varnames{:});
    else
        save(output_file, varnames{:}, '-append');
    end

    fprintf('Finished kernel n=%d — saved cov%d variables to %s\n', n, n, output_file);
end