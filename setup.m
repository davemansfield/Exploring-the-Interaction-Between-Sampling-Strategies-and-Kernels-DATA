function setup(gpml_dir)
% SETUP  Add the GPML toolbox and this repository to the MATLAB path.
%
%   setup('/path/to/gpml-matlab')   add the GPML core folders from the given location
%   setup                           GPML is already on the path
%
% The GPML MATLAB Toolbox (v4.2) is available separately from
% https://gaussianprocess.org/gpml/code/matlab/doc/
%
% Only the GPML core folders are added (the same folders as GPML's own
% startup.m), so a customised startup.m cannot place other copies of the
% repository functions on the path. The repository is added last so its
% functions take precedence over any same-named files already on the path.

root = fileparts(mfilename('fullpath'));

if nargin > 0
    if ~isfile(fullfile(gpml_dir, 'gp.m'))
        error('gp.m not found in %s. Point setup at the root of the GPML toolbox.', gpml_dir);
    end
    gpml_dirs = {'', 'cov', 'doc', 'inf', 'lik', 'mean', 'prior', 'util', ...
                 fullfile('util', 'minfunc'), fullfile('util', 'minfunc', 'compiled'), ...
                 fullfile('util', 'sparseinv')};
    for d = gpml_dirs
        addpath(fullfile(gpml_dir, d{1}));
    end
end

if exist('gp', 'file') ~= 2 || exist('covSEard', 'file') ~= 2
    error(['GPML toolbox not found on the MATLAB path. Download it from ' ...
           'https://gaussianprocess.org/gpml/code/matlab/doc/ and call ' ...
           'setup(''/path/to/gpml-matlab'').']);
end

addpath(genpath(fullfile(root, 'run_experiments')));
addpath(fullfile(root, 'functions'));

fprintf('Paths set: GPML, run_experiments/ and functions/.\n');
end
