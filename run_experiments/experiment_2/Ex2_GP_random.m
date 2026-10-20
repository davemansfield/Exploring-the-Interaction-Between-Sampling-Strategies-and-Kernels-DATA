%% Random sampling, all kernels, Environment 2

N_kernels=8;
N_hp_it=20;
N_run=50;
N_steps=102;
LB=0;
UB=5;
res=50;


xtestpoints=linspace(-5,5,res);
[Xtest1, Xtest2]=meshgrid(xtestpoints,xtestpoints);
xs = [Xtest1(:) Xtest2(:)]; %test grid

K=get_full_k_matrix(xs);
target=mvnrnd(zeros(50^2,1),K)';

rng('shuffle')

output_file = 'env2_random_results.mat';

for n = 1:N_kernels

    save_RMSE     = zeros(N_steps,N_run);
    save_timeopt  = zeros(N_steps,N_run);
    save_timestep = zeros(N_steps,N_run);
    save_NMLL     = zeros(N_steps,N_run);
    save_ls1      = zeros(N_steps,N_run);
    save_ls2      = zeros(N_steps,N_run);

    for run=1:N_run
        [cov,hyp,mean,lik]=cov_reset(n);
        %reset samples
        xidx=randi([1,2500],2,1);

        xr=xs(xidx,:); %random samples between -5 and 5
        y=target(xidx(1));
        y2=target(xidx(2));
        y=[y ;y2];
        %y=y+0.1*randn(size(y));

        jitter = 1e-6;  % Small positive value for jitter
        hyp.lik = log(exp(hyp.lik) + jitter);


        for i=1:N_steps
            xidx=xidx(end);
            %time prediction
            tstart_opt=tic;
            hyp = minimize(hyp,@gp,-N_hp_it,@infGaussLik,mean,cov,lik,xr,y);
            [pred,var] = gp(hyp,@infGaussLik,mean,cov,lik,xr,y,xs);
            tend_opt=toc(tstart_opt);

            NMLL=gp(hyp,@infGaussLik,mean,cov,lik,xr,y);

            tstart_step=tic;
            k = 3;  % Number of squares to move

            n_cols = 50;  % Number of columns in the grid
            n_rows = 50;  % Number of rows in the grid

            direction = randi([1,4]);
            row = ceil(xidx / n_cols);  % Current row
            col = mod(xidx-1, n_cols) + 1;  % Current column

            if direction == 1  % Move right
                new_col = min(col + k, n_cols);  % Ensure we don't go beyond the last column
                xidx = (row - 1) * n_cols + new_col;  % Update xidx based on the new column
            elseif direction == 2  % Move left
                new_col = max(col - k, 1);  % Ensure we don't go before the first column
                xidx = (row - 1) * n_cols + new_col;  % Update xidx based on the new column
            elseif direction == 3  % Move down
                new_row = min(row + k, n_rows);  % Ensure we don't go beyond the last row
                xidx = (new_row - 1) * n_cols + col;  % Update xidx based on the new row
            elseif direction == 4  % Move up
                new_row = max(row - k, 1);  % Ensure we don't go before the first row
                xidx = (new_row - 1) * n_cols + col;  % Update xidx based on the new row
            end
            xnew=xs(xidx,:);

            xr(end+1,:)=xnew(:,:);
            ynew=target(xidx);
            y(end+1,1)=ynew+0.1*randn(1);
            tend_step=toc(tstart_step);

            %collect data
            save_RMSE(i, run) = rmse(target,pred);
            save_timeopt(i, run)=tend_opt;
            save_timestep(i, run)=tend_step;
            save_NMLL(i,run)=NMLL;
            save_ls1(i,run)=hyp.cov(1);
            save_ls2(i,run)=hyp.cov(2);


            % tlo=tiledlayout(2,3);
            % num=length(xr);
            % 
            % h(1)=nexttile(tlo);
            % tarplot=reshape(target,[50,50]);
            % imagesc([-7,7],[-7,7],tarplot');
            % rectx=[-5,5,5,-5,-5];
            % recty=[5,5,-5,-5,5];
            % 
            % set(gca,'YDir','normal')
            % %     hold on;
            % %     plot(rectx,recty,'b-','LineWidth',2);
            % 
            % h(2)=nexttile(tlo);
            % scatter(xr(:,2),xr(:,1),[],y);
            % title('samples taken')
            % 
            % 
            % h(3)=nexttile(tlo);
            % modplot=reshape(pred,[50,50]);
            % imagesc([-7,7],[-7,7],modplot');
            % title('model prediction')
            % set(gca,'YDir','normal')
            % 
            % h(4)=nexttile(tlo);
            % varplot=reshape(var,[50,50]);
            % imagesc([-7,7],[-7,7],varplot');
            % title('variance')
            % set(gca,'YDir','normal')
            % drawnow
            % 
            % pause(0.25);
        end
    end

    eval(sprintf('save_RMSE_cov%d_env2_random = save_RMSE;', n));
    eval(sprintf('save_timeopt_cov%d_env2_random = save_timeopt;', n));
    eval(sprintf('save_timestep_cov%d_env2_random = save_timestep;', n));
    eval(sprintf('save_NMLL_cov%d_env2_random = save_NMLL;', n));
    eval(sprintf('save_ls1_cov%d_env2_random = save_ls1;', n));
    eval(sprintf('save_ls2_cov%d_env2_random = save_ls2;', n));

    varnames = { sprintf('save_RMSE_cov%d_env2_random', n), ...
                 sprintf('save_timeopt_cov%d_env2_random', n), ...
                 sprintf('save_timestep_cov%d_env2_random', n), ...
                 sprintf('save_NMLL_cov%d_env2_random', n), ...
                 sprintf('save_ls1_cov%d_env2_random', n), ...
                 sprintf('save_ls2_cov%d_env2_random', n) };

    if n == 1
        save(output_file, varnames{:});
    else
        save(output_file, varnames{:}, '-append');
    end

    fprintf('Finished kernel n=%d — saved cov%d variables to %s\n', n, n, output_file);
end
