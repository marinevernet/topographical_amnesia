
% Scripts to synchronize file from the two EEG amplifiers (front and back)

% re-referencing ?
rereferencing = 1;
% NB : do the synchronization on the original data, then re-reref
% we tried the opposite first

% general_path to be adapted to the computer you are working on
general_path = ''; 

% folders where the data are
folder1 = strcat(general_path, 'PC1 kompletter Datensatz/');
folder2 = strcat(general_path, 'PC2 kompletter Datensatz/');

if rereferencing
    folder_save = strcat(general_path, 'synchronized_files_reref/');
else
    folder_save = strcat(general_path, 'synchronized_files/');
end
if ~exist(folder_save, 'dir')
	mkdir(folder_save)
end

% files to synchronize for each task
StreckeList1 = dir(strcat(folder1, '*Strecke*.edf'));
StreckeList2 = dir(strcat(folder2, '*Strecke*.edf'));
StreckeList2(13) = []; % remove 20161214152031_Strecke13.edf and keep only 20161214152055_Strecke13_real.edf
RouteList1 = dir(strcat(folder1, '*Route*.edf'));
RouteList2 = dir(strcat(folder2, '*Route*.edf'));
WartsList1 = dir(strcat(folder1, '*w�rts*.edf'));
WartsList2 = dir(strcat(folder2, '*w�rts*.edf'));
LdList1 = dir(strcat(folder1, '*LD*.edf'));
LdList2 = dir(strcat(folder2, '*LD*.edf'));
GoogleList1 = dir(strcat(folder1, '*Google*.edf'));
GoogleList2 = dir(strcat(folder2, '*Google*.edf'));
KlinikList1 = dir(strcat(folder1, '*Klinik*.edf'));
KlinikList2 = dir(strcat(folder2, '*Klinik*.edf'));
KlinikList1(9) = []; % remove Klinik9 which cannot be open (on the PC1)
KlinikList2(9) = []; % remove Klinik9 which cannot be open (on the PC1)
KlinikList1(3) = []; % remove Klinik3 which cannot be open (on the PC2)
KlinikList2(3) = []; % remove Klinik3 which cannot be open (on the PC2)EoList1 = dir(strcat(folder1, '*EO*.edf'));
EoList1 = dir(strcat(folder1, '*EO*.edf'));
EoList2 = dir(strcat(folder2, '*EO*.edf'));
EcList1 = dir(strcat(folder1, '*_EC*.edf'));
EcList2 = dir(strcat(folder2, '*_EC*.edf'));
List1 = cat(1, StreckeList1, RouteList1, WartsList1, LdList1, GoogleList1, KlinikList1, EoList1, EcList1);
List2 = cat(1, StreckeList2, RouteList2, WartsList2, LdList2, GoogleList2, KlinikList2, EoList2, EcList2);
% List1 = cat(1, StreckeList1, GoogleList1, KlinikList1, EoList1, EcList1);
% List2 = cat(1, StreckeList2, GsoogleList2, KlinikList2, EoList2, EcList2);

% Initialize some variables
sampling_rate = zeros(length(List1), 2);
file_duration = zeros(length(List1), 5);
shift = zeros(length(List1), 1);
checkEO = 0;

% Loop on the files for this tasl
for file = 1:length(List1)

    clear ALLEEG EEG CURRENTSET ALLCOM;
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

    % load the data
    common_name = List1(file).name(find(List1(file).name == '_')+1:end-4);
    if strcmp(common_name, 'EO') && checkEO == 0
        checkEO = 1;
    elseif strcmp(common_name, 'EO') && checkEO == 1
        common_name = 'EO1';
    end

    disp(common_name);
    disp(List1(file).name);
    disp(List2(file).name);
    EEG = pop_biosig(strcat(folder1, List1(file).name));
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname','','gui','off');
    EEG = pop_biosig(strcat(folder2, List2(file).name));
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','','gui','off');

    % check the sampling rates and the recordings duration
    sampling_rate(file, 1) = ALLEEG(1).srate;
    sampling_rate(file, 2) = ALLEEG(2).srate;
    file_duration(file, 1) = size(ALLEEG(1).data, 2)/ALLEEG(1).srate;
    file_duration(file, 2) = size(ALLEEG(2).data, 2)/ALLEEG(2).srate;
    file_duration(file, 3) = file_duration(file, 2) - file_duration(file, 1);

    % We will remove the first and last 5000 samples of the first file to do the synch
    LTD = 5000;
    if strcmp(common_name, 'LD') || strcmp(common_name, 'RouteSD')
        LTD = 500;
    end

    % synchronization procedure on a segment where the first and last 5000
    % samples of the first file has been discarded (and choose segments of
    % similar length for the second file)
    % we will compare Cz (6th channel of file 1) with Pz (3rd channel of
    % file 2)
    % NB: for with rereferencing, we tried comparing F3 (2nd channel of file 1)
    % with P3 (4th channel of file 2) after the re-referencing, but we
    % decided to do first the synchronization and then the re-ref
    channel1 = 6;
    channel2 = 3;
    length_to_synch = length(ALLEEG(1).data(1, LTD:end-LTD-1));
    r = [];
    p = [];
    for i = 1:length(ALLEEG(2).data(1, :))-length_to_synch
        [corr1, corr2] = corrcoef(ALLEEG(1).data(channel1, LTD:end-LTD-1), ALLEEG(2).data(channel2, i:i+length_to_synch-1));
        r(i) = corr1(1, 2);
        p(i) = corr2(1, 2);
    end
    [M, I] = max(abs(r));
    shift(file) = (I-LTD)/ALLEEG(1).srate;

    % plot the r and p values
    figure; subplot(2, 1, 1); hold all;
    plot(1:length(ALLEEG(2).data(1, :))-length_to_synch, r, 'r');
    plot(1:length(ALLEEG(2).data(1, :))-length_to_synch, p, 'k');
    plot([I, I], [min(r) max(r)], 'g--');
    xlabel('sample shift (we expect a value around 5000)')
    legend('r', 'p-value');
    title(common_name);

    % plot the synchronization
    subplot(2, 1, 2);
    plot(ALLEEG(1).data(6, LTD:end-LTD-1), ALLEEG(2).data(3, I:I+length_to_synch-1));
    xlabel('file 1'); ylabel('file 2');
    title('best correlation between the two files');

    % cut the files accordingly and record the new duration
    if I > LTD
        ALLEEG(2) = eeg_eegrej( ALLEEG(2), [1 I-LTD] );
    elseif I < LTD
        ALLEEG(1) = eeg_eegrej( ALLEEG(1), [1 LTD-I] );
    end
    if size(ALLEEG(1).data, 2) > size(ALLEEG(2).data, 2)
        ALLEEG(1) = eeg_eegrej( ALLEEG(1), [size(ALLEEG(2).data, 2)+1 size(ALLEEG(1).data, 2)] );
    elseif size(ALLEEG(1).data, 2) < size(ALLEEG(2).data, 2)
        ALLEEG(2) = eeg_eegrej( ALLEEG(2), [size(ALLEEG(1).data, 2)+1 size(ALLEEG(2).data, 2)] );
    end
    file_duration(file, 4) = size(ALLEEG(1).data, 2)/ALLEEG(1).srate;
    file_duration(file, 5) = size(ALLEEG(2).data, 2)/ALLEEG(2).srate;
    file_duration(file, 6) = file_duration(file, 5) - file_duration(file, 4);

    % if rereferencing, do that here
    if rereferencing
        ALLEEG(1) = pop_reref( ALLEEG(1), 6); % reref to Cz
        ALLEEG(2) = pop_reref( ALLEEG(2), 3); % reref to Pz
    end

    % concatenate second file to first file
    ALLEEG(1).event = cat(2, ALLEEG(1).event, ALLEEG(2).event);
    ALLEEG(1).chanlocs = cat(1, ALLEEG(1).chanlocs, ALLEEG(2).chanlocs);
    ALLEEG(1).nbchan = ALLEEG(1).nbchan + ALLEEG(2).nbchan;
    ALLEEG(1).data = cat(1, ALLEEG(1).data, ALLEEG(2).data);

    % save the synchronization file
    ALLEEG(1) = pop_saveset( ALLEEG(1), 'filename',strcat(common_name, '.set'),'filepath', folder_save);

end

disp('Sampling rate should always be the same');
sampling_rate
disp('Duration of orginal files and difference (columns 1-3)');
disp('Duration of final files and differences (columns 4-6)');
file_duration
disp('Shift between the two files (in ms)');
shift*1000
