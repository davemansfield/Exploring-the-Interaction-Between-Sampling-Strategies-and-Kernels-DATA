%% Experiment 1, iterative random sampling, all kernels, Environment 3

N_kernels=8;
N_hp_it=20;
N_run=50;
N_steps=100;
LB=0;
UB=5;
res=50;


xtestpoints=linspace(6,30,res);
[Xtest1, Xtest2]=meshgrid(xtestpoints,xtestpoints);
xs = [Xtest1(:) Xtest2(:)]; %test grid


epochTest=300;

selectedData = data(data.epoch == epochTest, :);

x12=linspace(6,30,50);
x22=linspace(6,30,50);

[X2,Y2]=meshgrid(x12,x22);


conc_field=griddata(selectedData.x,selectedData.y,selectedData.temp,X2,Y2,'natural');
target=reshape(conc_field,[2500,1]);
target(isnan(target))=17.4;

output_file = 'ex1_env3_it_results.mat';

for n = 1:N_kernels

    RMSE = zeros(N_steps,N_run);

    for run=1:N_run
        [cov,hyp,mean,lik]=cov_reset(n);
        %reset samples
        xidx=randi([1,2500],2,1);

        xr=xs(xidx,:); %random samples between -5 and 5


        y=target(xidx(1));
        y2=target(xidx(2));
        y=[y ;y2];
        y=y+0.1*randn(size(y));
        rng('shuffle')

        jitter = 1e-6;  % Small positive value for jitter
        hyp.lik = log(exp(hyp.lik) + jitter);


        for i=1:N_steps
            %time prediction
            tstart=tic;
            hyp = minimize(hyp,@gp,-N_hp_it,@infGaussLik,mean,cov,lik,xr,y);
            [mod,var] = gp(hyp,@infGaussLik,mean,cov,lik,xr,y,xs);
            tend=toc(tstart);

            xidxnew=randi([1,2500],1);
            xnew=xs(xidxnew,:);

            xr(end+1,:)=xnew(:,:);
            ynew=target(xidxnew);
            y(end+1,1)=ynew+0.1*randn(1);


            %collect data
            RMSE(i, run) = rmse(target,mod);


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
            modplot=reshape(mod,[50,50]);
            imagesc(modplot');
            title('model prediction')
            set(gca,'YDir','normal')
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
        xr=0;
        y=0;
    end

    eval(sprintf('RMSE_cov%d_env3_ran_it = RMSE;', n));
    eval(sprintf('av_RMSE_cov%d_env3_ran_it = sum(RMSE(N_steps,:))/N_run;', n));

    varnames = { sprintf('RMSE_cov%d_env3_ran_it', n), sprintf('av_RMSE_cov%d_env3_ran_it', n) };

    if n == 1
        save(output_file, varnames{:});
    else
        save(output_file, varnames{:}, '-append');
    end

    fprintf('Finished kernel n=%d — saved cov%d variables to %s\n', n, n, output_file);
end
