%% Infotaxis sampling, all kernels, Environment 3
%
% Environment 3 (real temperature data, Intel Berkeley Research Lab dataset).
% Runs the infotaxis acquisition function (domain-wide, posterior-
% corrected expected information gain) across all kernel configurations.
% Results saved as separate named variables per kernel,
% checkpointed to disk after every kernel.
%
% Each run automatically retries with a new seed if it hits a numerical
% failure (e.g. chol: matrix must be positive definite), which can occur
% during L-BFGS optimisation on real, noisier data.
%
% REQUIRES: 'data' table already loaded in the workspace (with columns
% epoch, x, y, temp).
%
% -------------------------------------------------------------------------

N_kernels = 8;      % kernel indices n = 1..8 in cov_reset (Table 1)
N_hp_it   = 20;
N_run     = 50;
N_steps   = 100;
res       = 50;
MAX_RETRIES = 20;   % per-run retry cap before giving up on that run entirely

xtestpoints = linspace(-5,5,res);
[Xtest1, Xtest2] = meshgrid(xtestpoints, xtestpoints);
xs = [Xtest1(:) Xtest2(:)];

% --- Environment 3 target: real temperature field ---
epochTest = 300;
selectedData = data(data.epoch == epochTest, :);
x12 = linspace(6,30,50);
x22 = linspace(6,30,50);
[X2,Y2] = meshgrid(x12,x22);
conc_field = griddata(selectedData.x, selectedData.y, selectedData.temp, X2, Y2, 'natural');
target = reshape(conc_field, [2500,1]);
% ------------------------------------------------------------------------------

output_file = 'env3_entropy_results.mat';

for n = 1:N_kernels

    save_RMSE      = zeros(N_steps, N_run);
    save_timeopt   = zeros(N_steps, N_run);
    save_timestep  = zeros(N_steps, N_run);
    save_NMLL      = zeros(N_steps, N_run);
    save_ls1       = zeros(N_steps, N_run);
    save_ls2       = zeros(N_steps, N_run);

    for run = 1:N_run

        run_ok = false;
        attempt = 0;

        while ~run_ok

            attempt = attempt + 1;
            if attempt > MAX_RETRIES
                error('Kernel %d, run %d failed %d times in a row — stopping.', n, run, MAX_RETRIES);
            end

            try
                rng(run * 1000 + attempt)   % different draw each retry, still reproducible
                xidx0 = randi([1, res^2], 2, 1);

                [cov, hyp, mean, lik] = cov_reset(n);

                xr = xs(xidx0,:);
                y  = target(xidx0(1));
                y2 = target(xidx0(2));
                y  = [y; y2];

                jitter = 1e-6;
                hyp.lik = log(exp(hyp.lik) + jitter);

                RMSE_run      = zeros(N_steps,1);
                timeopt_run   = zeros(N_steps,1);
                timestep_run  = zeros(N_steps,1);
                NMLL_run      = zeros(N_steps,1);
                ls1_run       = zeros(N_steps,1);
                ls2_run       = zeros(N_steps,1);

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

                    Kxr    = feval(cov{:}, hyp.cov, xr);
                    Ky     = Kxr + sn2*eye(Ntr);
                    Lc     = chol(Ky, 'lower');

                    Kxs_xr = feval(cov{:}, hyp.cov, xs, xr);
                    Kxc_xr = feval(cov{:}, hyp.cov, xc, xr);

                    V1 = Lc \ Kxs_xr';
                    V2 = Lc \ Kxc_xr';

                    Kxc_prior = feval(cov{:}, hyp.cov, xs, xc);
                    Kxc_post  = Kxc_prior - V1' * V2;

                    denom   = xcvar' + sn2;
                    dvar    = (Kxc_post.^2) ./ denom;
                    var_new = max(var - dvar, 1e-12);
                    IG      = 0.5 * sum(log(var) - log(var_new), 1);

                    [~, max_idx] = max(IG);
                    % ---------------------------------------------------------------------

                    xnew = xc(max_idx,:);
                    [~, xidx_new] = ismember(xnew, xs, 'rows');

                    xr(end+1,:) = xnew;
                    ynew = target(xidx_new) + 0.1*randn(1);
                    y(end+1) = ynew;
                    tend_step = toc(tstart_step);

                    RMSE_run(i)     = rmse(target, mod);
                    timeopt_run(i)  = tend_opt;
                    timestep_run(i) = tend_step;
                    NMLL_run(i)     = NMLL;
                    ls1_run(i)      = hyp.cov(1);
                    if numel(hyp.cov) > 1
                        ls2_run(i)  = hyp.cov(2);
                    end
                end

                % completed all N_steps with no error — commit this run's results
                save_RMSE(:,run)     = RMSE_run;
                save_timeopt(:,run)  = timeopt_run;
                save_timestep(:,run) = timestep_run;
                save_NMLL(:,run)     = NMLL_run;
                save_ls1(:,run)      = ls1_run;
                save_ls2(:,run)      = ls2_run;

                run_ok = true;

            catch ME
                fprintf('Kernel %d, run %d, attempt %d failed (%s) — retrying with new seed\n', ...
                    n, run, attempt, ME.message);
                % loop repeats with attempt+1, new rng draw
            end
        end
    end

    eval(sprintf('save_RMSE_cov%d_env3_entropy = save_RMSE;', n));
    eval(sprintf('save_timeopt_cov%d_env3_entropy = save_timeopt;', n));
    eval(sprintf('save_timestep_cov%d_env3_entropy = save_timestep;', n));
    eval(sprintf('save_NMLL_cov%d_env3_entropy = save_NMLL;', n));
    eval(sprintf('save_ls1_cov%d_env3_entropy = save_ls1;', n));
    eval(sprintf('save_ls2_cov%d_env3_entropy = save_ls2;', n));

    varnames = { sprintf('save_RMSE_cov%d_env3_entropy', n), ...
                 sprintf('save_timeopt_cov%d_env3_entropy', n), ...
                 sprintf('save_timestep_cov%d_env3_entropy', n), ...
                 sprintf('save_NMLL_cov%d_env3_entropy', n), ...
                 sprintf('save_ls1_cov%d_env3_entropy', n), ...
                 sprintf('save_ls2_cov%d_env3_entropy', n) };

    if n == 1
        save(output_file, varnames{:});
    else
        save(output_file, varnames{:}, '-append');
    end

    fprintf('Finished kernel n=%d — saved cov%d variables to %s\n', n, n, output_file);
end