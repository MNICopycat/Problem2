%% Matlab stub for task 2 in assignment 4 in Image analysis

    load heart_data % load data
    
%% mean and std dev

    BV = background_values;
    CV = chamber_values;

    mean_background = mean(BV)
    stdev_background = std(BV)
    mean_chamber = mean(CV)
    stdev_chamber = std(CV)

%% Load data above first
    M = size(im,1); % height of image, change this!
    N = size(im,2); % width of image, change this!

    n = M*N; % Number of image pixels

    % create neighbour structure

    Neighbours = edges4connected(M,N); % use 4-neighbours (or 8-neighbours with edges8connected)

    i=Neighbours(:,1); 
    j=Neighbours(:,2);
    A = sparse(i,j,1,n,n); % create sparse matrix of connections between pixels 


    % We can make A into a graph, and show it (test this for example for M = 5, N = 6 to
    % see. For the full image it's not easy to see structure)
    Ag = graph(A);
    plot(Ag);

% Choose weights:


% Decide how important a short curve length is:
lambda = -log(length(CV)/(length(CV)+length(BV)));
 

A = A*lambda% set regularization term so  that A_ij = lambda

%negative log of the probabilities result in a constant term and the log of
%an exponential term with a variable, the log and exponential cancel out
%leaving x-mean^2 / 2*stdev^2. We ignore the constant term as it does not
%affect the optimization problem.

PC = ((im - mean_chamber).^2) / (2*(stdev_chamber.^2))
PB = ((im - mean_background).^2) / (2*(stdev_background.^2))

Ts = reshape(PC,[],1) % set weights to source, according to assignment!
%Ts = sparse((im(:)-mu1).^2); %for standard weighting without statistical weight
Tt = reshape(PB,[],1) % set weights to sink, according to assignment!
%Tt = sparse((im(:)-mu2).^2); %for standard weighting without statistical weight

% create matrix of the full graph, adding source and sink as nodes n+1 and
% n+2 respectively

F = sparse(zeros(n+2,n+2));
F(1:n,1:n) = A; % set regularization weights
F(n+1,1:n) = Ts'; % set data terms 
F(1:n,n+1) = Ts; % set data terms 
F(n+2,1:n) = Tt'; % set data terms 
F(1:n,n+2) = Tt; % set data terms 

% make sure that you understand what the matrix F represents!

Fg = graph(F); % turn F into a graph Fg

plot(Fg)


help maxflow % see how Matlab's maxflow function works

[MF,GF,CS,CT] = maxflow(Fg,n+1,n+2); % run maxflow on graph with source node (n+1) and sink node (n+2)

disp(MF) % shows the optmization value (maybe not so interesting)

% CS contains the pixels connected to the source node (including the source
% node n+1 as final entry (CT contains the sink nodes).

% We can construct out segmentation mask using these indices
seg = zeros(M,N);
seg(CS(1:end-1)) = 1; % set source pixels to 1
imagesc(im)
colormap(gray)
figure
imagesc(seg)
colormap(gray)