
% ---------------- SETTINGS ----------------
n = 1;
N_hp_it = 20;
N_run = 50;
N_steps = 100;
LB = 0;
UB = 5;
res = 50;

switch_interval = 10;
switch_tol = 1e-3;
kernel_set = [1 2 3 4 5 6];

% ---------------- STORAGE ----------------
save_RMSE_cov8_env2_greedy_switch = zeros(N_steps,N_run);
save_timeopt_cov8_env2_greedy_switch = zeros(N_steps,N_run);
save_NMLL_cov8_env2_greedy_switch = zeros(N_steps,N_run);
save_ls1_cov8_env2_greedy_switch = zeros(N_steps,N_run);
save_ls2_cov8_env2_greedy_switch = zeros(N_steps,N_run);

kernel_history_greedy = zeros(N_steps,N_run);

% ---------------- GRID ----------------
xtestpoints = linspace(-5,5,res);
[Xtest1, Xtest2] = meshgrid(xtestpoints,xtestpoints);
xs = [Xtest1(:), Xtest2(:)];

% ---------------- TRUE FUNCTION ----------------
K = get_full_k_matrix(xs);
target = mvnrnd(zeros(res^2,1),K)';

rng('shuffle')

% ============================================================
for run = 1:N_run

    % ---------------- INITIAL MODEL ----------------
    current_kernel_id = 1;
    [covfunc, hyp, meanfunc, likfunc] = cov_reset(current_kernel_id);

    % ---------------- INITIAL SAMPLES ----------------
    xidx = randi([1,res^2],2,1);

    xr = xs(xidx,:);
    y  = target(xidx);

    hyp.lik = log(exp(hyp.lik) + 1e-6);

    % ========================================================
    for i = 1:N_steps

        % ---------------- OPTIMISE CURRENT MODEL ----------------
        tstart_opt = tic;

        hyp = minimize(hyp,@gp,-N_hp_it,@infGaussLik, ...
                       meanfunc,covfunc,likfunc,xr,y);

        [y_pred, y_var] = gp(hyp,@infGaussLik, ...
                             meanfunc,covfunc,likfunc,xr,y,xs);

        tend_opt = toc(tstart_opt);

        NMLL = gp(hyp,@infGaussLik, ...
                  meanfunc,covfunc,likfunc,xr,y);

        % ====================================================
        % KERNEL SWITCHING BLOCK
        % ====================================================
        best_score = NMLL;

        best_covfunc = covfunc;
        best_hyp = hyp;
        best_meanfunc = meanfunc;
        best_likfunc = likfunc;
        best_id = current_kernel_id;

        if mod(i,switch_interval) == 0

            for k_id = kernel_set

                if k_id == current_kernel_id
                    continue;
                end

                [cov_k, hyp_k, mean_k, lik_k] = cov_reset(k_id);

                hyp_k.lik = hyp.lik;

                try
                    hyp_k = minimize(hyp_k,@gp,-10,@infGaussLik, ...
                                     mean_k,cov_k,lik_k,xr,y);

                    score_k = gp(hyp_k,@infGaussLik, ...
                                 mean_k,cov_k,lik_k,xr,y);

                    if score_k < best_score - switch_tol
                        best_score = score_k;

                        best_covfunc = cov_k;
                        best_hyp = hyp_k;
                        best_meanfunc = mean_k;
                        best_likfunc = lik_k;
                        best_id = k_id;
                    end

                catch
                    % ignore unstable kernels
                end
            end
        end

        % ---------------- APPLY BEST KERNEL ----------------
        covfunc = best_covfunc;
        hyp = best_hyp;
        meanfunc = best_meanfunc;
        likfunc = best_likfunc;
        current_kernel_id = best_id;

        kernel_history_greedy(i,run) = current_kernel_id;

        % ====================================================
        % GREEDY UNCERTAINTY ACQUISITION
        % ====================================================

        range = 1;

        [NNidx, ~] = rangesearch(xs, xr(end,:), range);
        NNidx = cell2mat(NNidx);

        NNvar = y_var(NNidx);
        SDreal = sqrt(y_var);

        [~, row] = max(SDreal);
        xnew = xs(row,:);

        % ---------------- UPDATE DATA ----------------
        xr(end+1,:) = xnew;
        y(end+1,1)  = target(row) + 0.1*randn;

        % ---------------- METRICS ----------------
        save_RMSE_cov8_env2_greedy_switch(i,run) = rmse(target,y_pred);
        save_timeopt_cov8_env2_greedy_switch(i,run) = tend_opt;
        save_NMLL_cov8_env2_greedy_switch(i,run) = NMLL;
        save_ls1_cov8_env2_greedy_switch(i,run) = hyp.cov(1);
        save_ls2_cov8_env2_greedy_switch(i,run) = hyp.cov(2);

    end
end