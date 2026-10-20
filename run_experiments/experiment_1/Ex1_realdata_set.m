%% Experiment 1, static dataset, all kernels, Environment 3

N_kernels=8;
N_hp_it=20;
N_run=50;
N_steps=100;
LB=-5;
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
%target(isnan(target))=19.4;

output_file = 'ex1_env3_set_results.mat';

for n = 1:N_kernels

    RMSE = zeros(1,N_run);

    for run=1:N_run
        [cov,hyp,mean,lik]=cov_reset(n);
        %reset samples
        xidx=randi([1,2500],N_steps,1);


        xr=xs(xidx,:); %random samples between -5 and 5
        y=target(xidx);
        y=y+0.1*randn(size(y)); %add simulated noise

        hyp = minimize(hyp,@gp,-N_hp_it,@infGaussLik,mean,cov,lik,xr,y);
        [mod,var] = gp(hyp,@infGaussLik,mean,cov,lik,xr,y,xs);

        % %collect data
        RMSE(run) = rmse(target,mod);
        % 
        % tlo=tiledlayout(2,3);
        % num=length(xr);
        % 
        % h(1)=nexttile(tlo);
        % tarplot=reshape(target,[50,50]);
        % imagesc([-5,5],[-5,5],tarplot');
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
        modplot=reshape(mod,[50,50]);
        imagesc([-5,5],[-5,5],modplot');
        title('model prediction')
        set(gca,'YDir','normal')
        % 
        % h(4)=nexttile(tlo);
        % varplot=reshape(var,[50,50]);
        % imagesc([-5,5],[-5,5],varplot');
        % title('variance')
        % set(gca,'YDir','normal')
        % drawnow
        % 
        % pause(0.25);
    end

    eval(sprintf('RMSE_cov%d_env3_set = RMSE;', n));
    eval(sprintf('av_RMSE_cov%d_env3 = sum(RMSE)/N_run;', n));

    varnames = { sprintf('RMSE_cov%d_env3_set', n), sprintf('av_RMSE_cov%d_env3', n) };

    if n == 1
        save(output_file, varnames{:});
    else
        save(output_file, varnames{:}, '-append');
    end

    fprintf('Finished kernel n=%d — saved cov%d variables to %s\n', n, n, output_file);
end
