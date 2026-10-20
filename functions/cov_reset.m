%set up cov functions
function [cov,hyp,mean,lik] = cov_reset(n)
    switch n
        case 1
            cov = {@covSEiso}; %squared exponential isotropic
            el1 = 0.7; %length scale
            sf = 0.1; % 
            hyp.cov = log([el1;sf]);
        
            %mean 
            mean = {@meanZero}; 
            hyp.mean = [];
            
            %likelihood
            lik = {@likGauss};
            sn = 0.1;
            hyp.lik = log(sn); 
        case 2
            %squared exponential ARD
            cov={@covSEard};
            el1 = 0.7;
            el2 = 0.7;
            sf = 0.1;
            hyp.cov = log([el1;el2;sf]);
    
            %mean 
            mean = {@meanZero}; 
            hyp.mean = [];
            
            %likelihood
            lik = {@likGauss};
            sn = 0.1;
            hyp.lik = log(sn); 
        case 3
            %cov matern 3/2
            cov={@covMaterniso,3};
            el1 = 0.7;
            sf = 0.1;
            hyp.cov = log([el1;sf]);
    
            %mean 
            mean = {@meanZero}; 
            hyp.mean = [];
            
            %likelihood
            lik = {@likGauss};
            sn = 0.1;
            hyp.lik = log(sn); 
    
        case 4
             %cov matern 3/2
            cov={@covMaternard,3};
            el1 = 0.7;
            el2 = 0.7;
            sf = 0.1;
            hyp.cov = log([el1;el2;sf]);
    
            %mean 
            mean = {@meanZero}; 
            hyp.mean = [];
            
            %likelihood
            lik = {@likGauss};
            sn = 0.1;
            hyp.lik = log(sn); 
    
        case 5
            %cov matern 5/2
            cov={@covMaterniso,5};
            el1 = 0.7;
            sf = 0.1;
            hyp.cov = log([el1;sf]);
    
            %mean 
            mean = {@meanZero}; 
            hyp.mean = [];
            
            %likelihood
            lik = {@likGauss};
            sn = 0.1;
            hyp.lik = log(sn);
        case 6
            %cov matern 5/2
            cov={@covMaternard,5};
            el1 = 0.7;
            el2 = 0.7;
            sf = 0.1;
            hyp.cov = log([el1;el2;sf]);
            %mean 
            mean = {@meanZero}; 
            hyp.mean = [];
            
            %likelihood
            lik = {@likGauss};
            sn = 0.1;
            hyp.lik = log(sn);
    
        case 7
            cov={@covPPiso,2};
            el1 = 0.7;
            sf = 0.1;
            hyp.cov = log([el1;sf]);
    
            %mean 
            mean = {@meanZero}; 
            hyp.mean = [];
            
            %likelihood
            lik = {@likGauss};
            sn = 0.1;
            hyp.lik = log(sn);
    
        case 8
            cov={@covPPard,2};
            el1 = 0.7;
            el2 = 0.7;
            sf = 0.1;
            hyp.cov = log([el1;el2;sf]);
            %mean 
            mean = {@meanZero}; 
            hyp.mean = [];
            
            %likelihood
            lik = {@likGauss};
            sn = 0.1;
            hyp.lik = log(sn);
    end
end
