function [acq] = GPUCBacqFunc(mod,var,i)
    %acq function for the UCB algorithm
    %term * var is the UCB, i represents time and the current iteration in
    %the main loop

    beta=100;
    delta=0.01;
    acq= mod + beta*sqrt(log((i^2*pi^2)/(3*delta))/2)*var;