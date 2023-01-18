
% Scripts to add chan info

% general_path to be adapted to the computer you are working on
general_path = '';
path_to_cap = '';
% for example:
% path_to_cap = 'Users/username/Documents/MATLAB/eeglab2022.0/plugins/dipfit/standard_BEM/elec/standard_1005.elc'

% folders where the data are
folder = strcat(general_path, 'synchronized_files_reref/');

% files to udpate for each task
StreckeList = dir(strcat(folder, '*Strecke*.set'));
RouteList = dir(strcat(folder, '*Route*.set'));
WartsList = dir(strcat(folder, '*wärts*.set'));
LdList = dir(strcat(folder, '*LD*.set'));
GoogleList = dir(strcat(folder, '*Google*.set'));
KlinikList = dir(strcat(folder, '*Klinik*.set'));
EoList = dir(strcat(folder, '*EO*.set'));
EcList = dir(strcat(folder, '*_EC*.set'));
List = cat(1, StreckeList, RouteList, WartsList, LdList, GoogleList, KlinikList, EoList, EcList);
% List1 = cat(1, StreckeList1, GoogleList1, KlinikList1, EoList1, EcList1);
% List2 = cat(1, StreckeList2, GsoogleList2, KlinikList2, EoList2, EcList2);

% Loop on the files for this tasl
for file = 1:length(List)

    clear ALLEEG EEG CURRENTSET ALLCOM;
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

    disp(List(file).name);

    EEG = pop_loadset('filename',List(file).name,'filepath',folder);
    [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
    EEG = eeg_checkset( EEG );
    EEG=pop_chanedit(EEG, 'lookup',path_to_cap);
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'savemode','resave');
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

end
