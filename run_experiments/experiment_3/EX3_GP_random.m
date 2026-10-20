
% ---------------- SETTINGS ----------------
N_hp_it = 20;
N_run = 50;
N_steps = 100;

res = 50;
n = 2;

switch_interval = 10;
switch_tol = 1e-3;

kernel_set = [1 2 3 4 5 6];

% ---------------- STORAGE ----------------
save_RMSE_switch = zeros(N_steps,N_run);
save_NMLL_switch  = zeros(N_steps,N_run);
kernel_history    = zeros(N_steps,N_run);

% ---------------- GRID ----------------
xtestpoints = linspace(-5,5,res);
[Xtest1, Xtest2] = meshgrid(xtestpoints,xtestpoints);
xs = [Xtest1(:), Xtest2(:)];

% ---------------- TRUE FUNCTION (ENVIRONMENT 2) ----------------
K = get_full_k_matrix(xs);
target = mvnrnd(zeros(res^2,1),K)';

rng('shuffle')

% ============================================================
for run = 1:N_run

    % ---------------- INITIAL KERNEL ----------------
    current_kernel_id = n;
    [covfunc, hyp, meanfunc, likfunc] = cov_reset(current_kernel_id);

    % ---------------- INITIAL SAMPLES ----------------
    xidx = randi([1,res^2],2,1);

    xr = xs(xidx,:);
    y  = target(xidx);

    hyp.lik = log(exp(hyp.lik) + 1e-6);

    % ========================================================
    for i = 1:N_steps

        % ====================================================
        % FIT CURRENT SWITCHING MODEL
        % ====================================================
        hyp = minimize(hyp,@gp,-N_hp_it,@infGaussLik, ...
                       meanfunc,covfunc,likfunc,xr,y);

        [pred, var] = gp(hyp,@infGaussLik, ...
                         meanfunc,covfunc,likfunc,xr,y,xs);

        NMLL = gp(hyp,@infGaussLik, ...
                  meanfunc,covfunc,likfunc,xr,y);

        % ====================================================
        % KERNEL SWITCHING STEP
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
                    % skip unstable kernels
                end
            end
        end

        % ---------------- APPLY BEST MODEL ----------------
        covfunc = best_covfunc;
        hyp = best_hyp;
        meanfunc = best_meanfunc;
        likfunc = best_likfunc;
        current_kernel_id = best_id;

        kernel_history(i,run) = current_kernel_id;

        % ====================================================
        % RANDOM WALK ACQUISITION
        % ====================================================
        xidx_cur = xidx(end);

        k = 3;
        n_cols = res;
        n_rows = res;

        direction = randi([1,4]);
        row = ceil(xidx_cur / n_cols);
        col = mod(xidx_cur-1, n_cols) + 1;

        if direction == 1
            new_col = min(col + k, n_cols);
            xidx_new = (row - 1) * n_cols + new_col;
        elseif direction == 2
            new_col = max(col - k, 1);
            xidx_new = (row - 1) * n_cols + new_col;
        elseif direction == 3
            new_row = min(row + k, n_rows);
            xidx_new = (new_row - 1) * n_cols + col;
        else
            new_row = max(row - k, 1);
            xidx_new = (new_row - 1) * n_cols + col;
        end

        % ---------------- UPDATE DATA ----------------
        xr(end+1,:) = xs(xidx_new,:);
        y(end+1,1)  = target(xidx_new) + 0.1*randn;

        xidx = [xidx; xidx_new];

        % ---------------- METRICS ----------------
        save_RMSE_switch(i,run) = rmse(target,pred);
        save_NMLL_switch(i,run) = NMLL;

    end
end

% ============================================================
% PLOTS
% ============================================================
figure;
plot(save_RMSE_switch);
title('Kernel Switching GP (Random Walk Environment)');

figure;
imagesc(kernel_history);
colorbar;
title('Kernel Switching Timeline');
xlabel('Run'); ylabel('Step');