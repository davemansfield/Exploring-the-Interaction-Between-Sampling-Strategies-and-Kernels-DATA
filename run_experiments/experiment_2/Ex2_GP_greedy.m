%% Greedy sampling, all kernels, Environment 2

N_kernels=8;
N_hp_it=20;
N_run=50;
N_steps=100;
LB=0;
UB=5;
res=50;


xtestpoints=linspace(-5,5,res);
[Xtest1, Xtest2]=meshgrid(xtestpoints,xtestpoints);
xs = [Xtest1(:) Xtest2(:)]; %test grid


K=get_full_k_matrix(xs);
target=mvnrnd(zeros(50^2,1),K)';
rng('shuffle')

output_file = 'env2_greedy_results.mat';

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
            %time prediction
            tstart_opt=tic;
            hyp = minimize(hyp,@gp,-N_hp_it,@infGaussLik,mean,cov,lik,xr,y);
            [mod,var] = gp(hyp,@infGaussLik,mean,cov,lik,xr,y,xs);
            tend_opt=toc(tstart_opt);

            NMLL=gp(hyp,@infGaussLik,mean,cov,lik,xr,y);

            tstart_step=tic;
            range=1;
            [NNidx,D]=rangesearch(xs,xr(end,:),range); %indices of xg in that range
            NNidx=cell2mat(NNidx);
            %make corresponding array of variance values 
            NNvar=var(NNidx);
            SD=sqrt(NNvar);%standard deviation
            SDreal=sqrt(var);

            %find maximum value of SD inside range around quad
            maxVal=max(SD,[],'all'); %find index of maximum SD

            %find the index of the maximum value on the full grid
            [row,col]=find(SDreal==maxVal);
            xnew=xs(row(1),:); %find the corresponding next sampling location on xg

            xr(end+1,:)=xnew(1,:);
            ynew=target(row(1));
            y(end+1)=ynew+0.1*randn(1);
            tend_step=toc(tstart_step);

            %collect data
            save_RMSE(i, run) = rmse(target,mod);
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
            %     hold on;
            %     plot(rectx,recty,'b-','LineWidth',2);
            % 
            % h(2)=nexttile(tlo);
            % scatter(xr(:,2),xr(:,1),[],y);
            % title('samples taken')
            % 
            % 
            % h(3)=nexttile(tlo);
            % modplot=reshape(mod,[50,50]);
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

    eval(sprintf('save_RMSE_cov%d_env2_greedy = save_RMSE;', n));
    eval(sprintf('save_timeopt_cov%d_env2_greedy = save_timeopt;', n));
    eval(sprintf('save_timestep_cov%d_env2_greedy = save_timestep;', n));
    eval(sprintf('save_NMLL_cov%d_env2_greedy = save_NMLL;', n));
    eval(sprintf('save_ls1_cov%d_env2_greedy = save_ls1;', n));
    eval(sprintf('save_ls2_cov%d_env2_greedy = save_ls2;', n));

    varnames = { sprintf('save_RMSE_cov%d_env2_greedy', n), ...
                 sprintf('save_timeopt_cov%d_env2_greedy', n), ...
                 sprintf('save_timestep_cov%d_env2_greedy', n), ...
                 sprintf('save_NMLL_cov%d_env2_greedy', n), ...
                 sprintf('save_ls1_cov%d_env2_greedy', n), ...
                 sprintf('save_ls2_cov%d_env2_greedy', n) };

    if n == 1
        save(output_file, varnames{:});
    else
        save(output_file, varnames{:}, '-append');
    end

    fprintf('Finished kernel n=%d — saved cov%d variables to %s\n', n, n, output_file);
end
