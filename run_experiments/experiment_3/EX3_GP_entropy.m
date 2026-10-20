% ---------------- SETTINGS ----------------
n = 1;
N_hp_it = 20;
N_run = 50;
N_steps = 100;

res = 50;
switch_interval = 10;
switch_tol = 1e-3;

kernel_set = [1 2 3 4 5 6];

% ---------------- STORAGE ----------------
save_RMSE = zeros(N_steps,N_run);
save_NMLL  = zeros(N_steps,N_run);

kernel_history = zeros(N_steps,N_run);

% ---------------- GRID ----------------
xgrid = linspace(-5,5,res);
[X1,X2] = meshgrid(xgrid,xgrid);
xs = [X1(:), X2(:)];

% ---------------- TRUE FUNCTION ----------------
K = get_full_k_matrix(xs);
target = mvnrnd(zeros(res^2,1),K)';

rng('shuffle')

% ============================================================
for run = 1:N_run

    % ---------------- INITIAL MODEL ----------------
    current_kernel_id = n;
    [covfunc, hyp, meanfunc, likfunc] = cov_reset(current_kernel_id);

    % ---------------- INITIAL DATA ----------------
    xidx = randi([1,res^2],2,1);

    xr = xs(xidx,:);
    y  = target(xidx);

    hyp.lik = log(exp(hyp.lik) + 1e-6);

    % ========================================================
    for i = 1:N_steps

        % ---------------- GP FIT ----------------
        hyp = minimize(hyp,@gp,-N_hp_it,@infGaussLik, ...
                       meanfunc,covfunc,likfunc,xr,y);

        [mu, s2] = gp(hyp,@infGaussLik, ...
                      meanfunc,covfunc,likfunc,xr,y,xs);

        s2 = s2(:);

        NMLL = gp(hyp,@infGaussLik, ...
                  meanfunc,covfunc,likfunc,xr,y);

        % ====================================================
        % KERNEL SWITCHING
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

        % ---------------- APPLY MODEL ----------------
        covfunc = best_covfunc;
        hyp = best_hyp;
        meanfunc = best_meanfunc;
        likfunc = best_likfunc;
        current_kernel_id = best_id;

        kernel_history(i,run) = current_kernel_id;

        % ====================================================
        % INFOTAXIS ACQUISITION (domain-wide expected information gain)
        % ====================================================
        range = 1;

        neigh = rangesearch(xs, xr(end,:), range);

        idx = neigh{1};

        if isempty(idx)
            idx = 1:size(xs,1);
        end

        xc = xs(idx,:);
        xcvar = s2(idx);

        % avoid NaNs breaking entropy
        xcvar(xcvar < 1e-12) = 1e-12;

        % --- domain-wide expected information gain (posterior-corrected) ---
        sn2 = exp(2*hyp.lik);
        Ntr = size(xr,1);

        Kxr    = feval(covfunc{:}, hyp.cov, xr);            % [Ntr x Ntr]
        Ky     = Kxr + sn2*eye(Ntr);
        Lc     = chol(Ky, 'lower');

        Kxs_xr = feval(covfunc{:}, hyp.cov, xs, xr);        % [res^2 x Ntr]
        Kxc_xr = feval(covfunc{:}, hyp.cov, xc, xr);        % [Ncand x Ntr]

        V1 = Lc \ Kxs_xr';                                   % [Ntr x res^2]
        V2 = Lc \ Kxc_xr';                                   % [Ntr x Ncand]

        Kxc_prior = feval(covfunc{:}, hyp.cov, xs, xc);      % [res^2 x Ncand]
        Kxc_post  = Kxc_prior - V1' * V2;                    % posterior cross-covariance

        denom   = xcvar' + sn2;
        dvar    = (Kxc_post.^2) ./ denom;
        var_new = max(s2 - dvar, 1e-12);
        IG      = 0.5 * sum(log(s2) - log(var_new), 1);      % [1 x Ncandidates]

        [~, best_local] = max(IG);
        % ---------------------------------------------------------------------

        xnew = xc(best_local,:);

        % grid index of the selected point
        d = sum((xs - xnew).^2,2);
        [~, xidx_new] = min(d);

        % ---------------- UPDATE DATA ----------------
        xr = [xr; xnew];
        y  = [y; target(xidx_new) + 0.1*randn];

        % ---------------- METRICS ----------------
        save_RMSE(i,run) = rmse(target,mu);
        save_NMLL(i,run)  = NMLL;

    end
end

% ============================================================
% PLOTS
% ============================================================
figure;
plot(save_RMSE);
title('RMSE over time');

figure;
imagesc(kernel_history);
colorbar;
title('Kernel switching timeline');
xlabel('Run'); ylabel('Step');