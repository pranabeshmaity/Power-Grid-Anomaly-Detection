clc
clear
close all
format long
tic

% --- Main ----------------------------------------------------------------
CaseFolder    = 'CaseFolderLocation';
InstallFolder = 'C:\DSATools_20-SL\Tsat\bin';
Binfile       = 'Case1_TSAT.bin';
Switching_file = 'Case1_Disturbance.swi';
Monitor_file = 'Case1_Monitoring.mon';
FO_location   = 'Case1_FO_LocationType.dat';

%------Read branch data from monitor file ---------------------------------
[frombus,tobus,lineID] = TSAT_readbranch_frommonitorfile(CaseFolder,Monitor_file);

% -----Read FO location file-----------------------------------------------
[FO_parameter] = TSAT_read_ForceOscillation(CaseFolder,FO_location);

%-------------Change the event source bus ---------------------------------
newFolder = CaseFolder;
oldFolder = cd(newFolder);

% Open the file for reading
fid = fopen('Headers_279.csv', 'r');

Governor_data    = readtable('HYGOV_Header.csv');
Governor_numbers = Governor_data.starting_numbers;
Gen_ID           = Governor_data.third_strings;
[N,~]            = size(Governor_data);

% Read the data using textscan
data_cell = textscan(fid, '%s', 'Delimiter', ',', 'EmptyValue', NaN);
fclose(fid);
numeric_data = (data_cell{1});

freqProfile = xlsread('Case1_FO_FrequencyProfile.csv');
MagProfile  = xlsread('Case1_FO_MagnitudeProfile.csv');

FO_freq   = 0.1:0.1:1.5;
FO_Mag    = 0.01;
FO_source = "HYGOV";
Type      = "FO";

cd(oldFolder)
freqProfile = zeros(1, 2);

for i = 1:length(FO_freq)
    if ~isnan(FO_freq(i))
        freqProfile(1, 2) = FO_freq(i);
    end
    
    filePath = 'temp_FO_FrequencyProfile.csv';
    writematrix(freqProfile, filePath);
    
    for j = 1:1:N
        parts = strsplit(numeric_data{j}, '|');
        
        first_elements{j} = num2str(Governor_numbers(j,1));       
        last_elements{j}  = char(Gen_ID(j,1));
        
        FO_parameter{3}  = first_elements{j};        
        FO_parameter{4}  = char("'" + Gen_ID(j,1) + "'");
        FO_parameter{19} = ['''temp_FO_FrequencyProfile.csv''']; 
        FO_parameter{20} = ['''Case1_FO_MagnitudeProfile.csv'''];

        if ~(strcmp(last_elements{j}, 'S') || strcmp(last_elements{j}, 'W'))
            
            TSAT_Change_ForceOscillation(CaseFolder,FO_location,FO_parameter);
            TSAT_Run(CaseFolder,InstallFolder);
            
            [data_store,time] = TSAT_ResponseReader(CaseFolder,Binfile);
            
            header_bus = (cellstr(numeric_data)).';
            Gen_V = data_store(:, :, 1);
            Gen_P = data_store(:, :, 2);
            Gen_Q = data_store(:, :, 3);
            Gen_A = data_store(:, :, 4);
            
            Header = ["Time" header_bus+"_V", header_bus+"_MW", header_bus+"_MVAR", header_bus+"_deg"];
            FO_data = [time,Gen_V,Gen_P,Gen_Q,Gen_A];       

            filename = sprintf('.\\Governorc_cases\\%s_%s_%.2fpu_%.2fHz_%s_%s.csv', ...
                               Type, FO_source, FO_Mag, FO_freq(i), FO_parameter{3}, string(last_elements{j}));
            filename = strtrim(filename);

            fid = fopen(filename, 'w');
            if fid == -1
                error('Could not open file for writing: %s', filename);
            end

            fprintf(fid, '%s,', Header{1:end-1});
            fprintf(fid, '%s\n', Header{end});
            fclose(fid);

            dlmwrite(filename, FO_data, '-append', 'precision', '%.15f', 'delimiter', ',');
            j = j + 1;
        end
    end
end

toc

% -------------------------------------------------------------------------
% --- Functions -----------------------------------------------------------
% -------------------------------------------------------------------------

function [frombus,tobus,lineID] = TSAT_readbranch_frommonitorfile(CaseFolder,Monitor_file)
newFolder = CaseFolder;
oldFolder = cd(newFolder);

fid = fopen(Monitor_file,'r');
g = textscan(fid,'%s','delimiter','\n');
data = cell(1,length(g{1}));
fclose(fid);

fid = fopen(Monitor_file,'r');
i = 1;
while ~feof(fid)
    l = fgetl(fid);
    data{i} = l;
    if(strfind(l,'{Branch}'))
        idx1 = i;
    end
    if(strfind(l,'{End Branch}'))
        idx2 = i;
    end
    i = i+1;
end
fclose(fid);

Branch_parameter = data(idx1+1:idx2-1);
nonemptyBranch_parameter = Branch_parameter(~cellfun('isempty', Branch_parameter));

for i = 1:length(nonemptyBranch_parameter)
    output_str = strsplit(nonemptyBranch_parameter{i}, '/');
    output_str = strtrim(output_str{1});
    output_str = strrep(output_str, {','}, '');
    output_str = strrep(output_str, {''''}, '');
    output_str = strsplit(output_str{1});
    frombus{i} = output_str{1};
    tobus{i}   = output_str{2};
    lineID{i}  = output_str{3};
    i = i+1;
end
cd(oldFolder)
end

function TSAT_Run(CaseFolder,InstallFolder)
newFolder = CaseFolder;
oldFolder = cd(newFolder);
command = [InstallFolder,'\tsat_batch Case1_TSAT.TSA'];
system(command);
cd(oldFolder)
end

function [data_store, time] = TSAT_ResponseReader(CaseFolder,Binfile)
newFolder = CaseFolder;
oldFolder = cd(newFolder);
r = actxserver('ResultScript.BinReader');
r.file = Binfile;
identifier = {'gen_v', 'gen_p', 'gen_q','bus_va'};
r.ctg = 1;
r.scen = 1;
t = r.timeValues();
t = cell2mat(t);
time_size = size(t);

load('busID_279.mat') 
load('busNames_279.mat') 
busID_279 = busID;
busNames_279 = busNames;
identifier1 = identifier([1, 2, 3, 4]);
num_bus = 279;
data_store_279 = zeros(time_size(1), num_bus, length(identifier1));

for j = 1:length(identifier1)
    r.quan = identifier1{j};
    for i = 1:num_bus
        r.bus1 = busNames_279{i};
        r.id   = busID_279{i};
        y = r.curveValues();
        data_store_279(:, i, j) = cell2mat(y);
    end
end
cd(oldFolder)
data_store = data_store_279;
time = t;
end

function [Eventbus_parameter]=TSAT_read_ForceOscillation(CaseFolder,FO_location)
newFolder = CaseFolder;
oldFolder = cd(newFolder);
fid = fopen(FO_location,'r');
g = textscan(fid,'%s','delimiter','\n');
g = g{1};

Eventbus_parameter = {}; 
for i = 1:numel(g)
    if isempty(strtrim(g{i}))
        continue;
    end
    if ~startsWith(g{i}, '/')
        settings = split(g{i}, ',');
        Eventbus_parameter = [Eventbus_parameter; settings]'; 
    end
end

fclose(fid); 
cd(oldFolder)
end

function TSAT_Change_ForceOscillation(CaseFolder,FO_location,FO_parameter)
newFolder = CaseFolder;
oldFolder = cd(newFolder);

fid = fopen(FO_location, 'r');
g = textscan(fid, '%s', 'delimiter', '\n');
fclose(fid);

g = g{1}; 
fid = fopen('FO_temp.dat', 'w');

for i = 1:numel(g)
    if isempty(strtrim(g{i}))
        continue;
    end
    if ~startsWith(g{i}, '/')
        fprintf(fid, '%s/\n', strjoin(cellstr(FO_parameter), ','));
    else
        fprintf(fid, '%s\r\n', g{i});
    end
end

fclose(fid);
cd(oldFolder)
end
