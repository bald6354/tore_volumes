function frames = readUAVimgTimes(fn)

%% Import data from text file
% Script for importing data from the following text file:
%
%    filename: /home/wescomp/Dropbox/WesDocs/UD/Research/eventCameraFeatures/uavData/events.txt
%
% Auto-generated by MATLAB on 04-Jan-2021 11:57:37

%% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 3);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = " ";

% Specify column names and types
opts.VariableNames = ["id", "timestamp", "fn"];
opts.VariableTypes = ["double", "double", "char"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";

% Import the data
events = readtable(fn, opts);


%% Clear temporary variables
clear opts

frames.timeStamp = events.timestamp.*1e6;
frames.filepath = events.fn;
