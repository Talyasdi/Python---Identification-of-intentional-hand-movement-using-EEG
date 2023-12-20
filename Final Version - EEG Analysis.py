from scipy.stats import f_oneway
from mne import create_info, EpochsArray
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import stats


# Before running this code please reffer to the experiment documentation

"""
# Pre Run Tasks

Upload all of your EEG data to the project folder.
List of needed files per run:
1.  EEG file (.eeg)
2.  VHDR file (.vhdr)
3.  VMRK file (.vmrk)
4.  Montage file (.bvef)

"""

# Pre-Run Objects To Change
vhdr_file_path = 'Ori_1st_trail.vhdr'
montage_file_path = 'BC-MR-32.bvef'

# Insert reference electrode name, and closest to eyes two electrodes
referance_elctrode = 'ECG'
eye1_elctrode = 'Fp1'
eye2_elctrode = 'Fp2'

# In case of using a faulty montage, a run is still possible but channel rename is needed to fit a 32-elctrode montage
renaming_channels = {'FC1' : 'Oz','FC2' : 'FC1','CP1' : 'FC2','CP2' : 'CP1','FC5' : 'CP2','FC6': 'FC5','CP5' : 'FC6', 'CP6': 'CP5','TP9' : 'CP6','TP10' : 'TP9','Eog' : 'TP10','Ekg1' : 'POz', 'Ekg2' : 'ECG'}

# Define frequancy pass to be filtred in the analysis
low_pass = 8
high_pass = 12 #possibly 30

# Define number of elctrodes used
channels_number = 32

# Define how many seconds before and after hand movment each epoch will contain
tmin_epochs = -4
tmax_epochs = 2 #possibly 0.5

# Specific channels to be used in the epochs
channel_group = ['Fz', 'Cz', 'C3', 'C4']

# Define the names of events
start_block_event_name = "New Segment"
hand_movement_event_name = "Comment/start"


def data_loading():
    # Loading raw vhdr data
    raw = mne.io.read_raw_brainvision(vhdr_file_path, preload=True)

    # Montage and Reference Setting
    # In case of using a faulty montage, a run is still possible but channel rename is needed to fit a 32-elctrode montage
    raw.rename_channels(renaming_channels)
    channels_names = raw.ch_names

    montage = mne.channels.read_custom_montage(montage_file_path)     # Load montage
    raw.set_montage(montage, match_case=True)
    raw.set_eeg_reference(ref_channels=[referance_elctrode]    # Set reference
    return raw


def filtering_and_pre_processing(raw):
    # In this part filtering and ICA is being used. **In the ICA part a human judjment is reqired**
    raw.filter(low_pass,high_pass) # Applay band pass filtering to data
    raw.plot(n_channels=channels_number) # Raw data plot after bandpass filter

    # Create an ICA object
    ica = mne.preprocessing.ICA(n_components=channels_number, random_state=0)
    ica.fit(raw)
    """
    ### HUMAN JUDJMENT IS NEEDED HERE ###
    This block plots ICAs and relevant metadata.
    Basing that, humman judjment is needed to determine what components and channels may be needed to remove after
    """
    # Plot the ICA components
    ica.plot_components();

    # The x-axis represents the time in seconds.
    # It shows the temporal progression of the independent component's time series.
    ica.plot_sources(inst=raw)

    """
    iterate through each component's mixing matrix and find the index of the channel with
    the maximum absolute value of the component. This index corresponds to the associated
    electrode channel.
    """
    ch_names = ica.info['ch_names']
    for idx, component in enumerate(ica.mixing_matrix_.T):
        electrode_name = ch_names[np.argmax(np.abs(component))]
        print(f"Component {idx + 1} is associated with electrode: {electrode_name}")

    # Using ICA to find 'bad' components based on two elctrodes we defined (closest to the eyes)
    bad_idx1, scores1 = ica.find_bads_eog(raw,eye1_elctrode, threshold=2)
    bad_idx2, scores2 = ica.find_bads_eog(raw,eye2_elctrode, threshold=2)

    # CHANGE THIS LIST, and add the 'bad' components indexes based on the former block (in your impression)
    bad_indexes_to_add = [3, 6, 7, 8, 9, 13, 15]
    concatenated_list = list(set(bad_idx1 + bad_idx2 + bad_indexes_to_add))

    ica.exclude = concatenated_list     # Eye exclude
    filtered_raw = ica.apply(raw.copy(), exclude=ica.exclude)     # Getting clean data after applying ICA reduction
    filtered_raw.plot(n_channels=channels_number)     # Raw data plot after ICA
    return filtered_raw


def epocs_analysis(filtered_raw):
    # Extracting events from recording
    events = mne.events_from_annotations(filtered_raw)
    event_ids = events[1]
    events = events[0]
    eventes_no = len(events)

    # Visualizing events in block number 1 (contains 23 hand movments)
    mne.viz.plot_events(events[0:24])

    # Creating epochs objects
    epochs = mne.Epochs(filtered_raw, events, event_id=event_ids, tmin=tmin_epochs, tmax= tmax_epochs, preload=True)

    # Visualizing the epochs
    epochs.plot()
    epochs[start_block_event_name].plot_image(picks=[5])
    epochs[hand_movement_event_name].plot_image(picks=[5])
    epochs.info()

    """
    In this block we define what epochs each block contains.
    You need to change it acording to the blocks of hand movments. Running "print(events)" will print a list of all event indexes,
    so you can clearly see all "Comment/start" events (index- 10001) and "New Segment" events (index- 99999).
    For epoch statistical comparison the number of events in each epoch needs to be the same.
    In this specific analysis of 'ori_first_trial' the number of events in each epoch was different, thus some of the blocks where cut short.
    """
    block1 = epochs[1:22].reorder_channels(channel_group)
    block2 = epochs[24:45].reorder_channels(channel_group) # Original num of epochs for this block: [24:50]
    block3 = epochs[51:72].reorder_channels(channel_group) # Original num of epochs for this block: [51:75]
    block4 = epochs[76:97].reorder_channels(channel_group) # Original num of epochs for this block: [76:105]
    block5 = epochs[106:127].reorder_channels(channel_group) # Original num of epochs for this block: [106:131]
    #block6 = epochs[132:139].reorder_channels(channel_group) # Short block- ignore
    block7 = epochs[140:161].reorder_channels(channel_group) # Original num of epochs for this block: [140:165]

    # Visualizing blocks 1 and 2
    block1.plot_image()
    block2.plot_image()
    blocks_list = [block1, block2, block3, block4, block5, None, block7]
    return blocks_list

def statistical_analysis(blocks_list):
    """
    For this specific analysis, we use Coorelation Coefficient and ANOVA test.
    These analysis compare 2 groups, so we combined 3 blocks at a time to compare
    the first half and second half of the EEG recording. For a more accurate analysis,
    we reccomend using ANOVA to compare all individual blocks of the experimen
    This section of code requires changes that cannot be inserted in the "Pre-Run Objects To Change" section,
    because of the nature of groups compared.
    """

    # Combining first-half (of EEG recording) blocks and second-half blocks
    blocks1to3 = [blocks_list[0], blocks_list[1], blocks_list[2]]
    block1to3concatenated = mne.concatenate_epochs(blocks1to3)
    blocks4to7 = [blocks_list[3], blocks_list[4], blocks_list[6]]
    block4to7concatenated = mne.concatenate_epochs(blocks4to7)

    # Coorelation coefficiant
    # Get the data for two epochs
    blocks1to3 = blocks_list[0].get_data()[0]
    blocks4to7 = blocks_list[1].get_data()[0]

    # Compute the correlation between the two epochs
    corr_coef = np.corrcoef(blocks1to3, blocks4to7)[0, 1]
    print(corr_coef)

    # ANOVA
    # Getting data from the epochs (necessary for running any computing function)
    data_block1 = blocks_list[0].get_data()
    data_block2 = blocks_list[1].get_data()
    data_block3 = blocks_list[2].get_data()
    data_block4 = blocks_list[3].get_data()
    data_block5 = blocks_list[4].get_data()
    data_block7 = blocks_list[6].get_data()

    #ANOVA calculation
    f_value1, p_value1 = f_oneway(data_block1, data_block2) #data_block3, data_block4, data_block5, data_block7)
    f_value_single1 = f_value1[0]

    print("F-value1:", f_value1)
    print("p-value1:", p_value1)
    print("f_value_single1:", f_value_single1)

    # Assuming you have 7 blocks: block1, block2, ..., block7
    blocks = [blocks_list[0], blocks_list[1], blocks_list[2], blocks_list[3], blocks_list[4], blocks_list[6]]

    # Collect data from all blocks
    all_data = [block.get_data().reshape(block.get_data().shape[0], -1) for block in blocks]

    # Perform one-way ANOVA
    f_statistic, p_value = stats.f_oneway(*all_data)

    # Calculate the correlation-like value based on F-statistic
    correlation_value = 1 - (f_statistic / (f_statistic + np.sum(all_data[0].shape[1] - 1)))

    print("F-Statistic:", f_statistic)
    print("P-Value:", p_value)
    print("Correlation Value:", correlation_value)

    # Assuming `block1` and `block2` are already defined
    data_block1 = blocks_list[0].get_data()
    data_block2 = blocks_list[1].get_data()

    # Reshape the data for t-test (time points x channels)
    data_block1 = data_block1.reshape(data_block1.shape[0], -1)
    data_block2 = data_block2.reshape(data_block2.shape[0], -1)

    # Perform independent t-test for each time point and channel
    t_values, p_values = stats.ttest_ind(data_block1, data_block2)

    # Correct for multiple comparisons if needed
    # You can use methods like Bonferroni or False Discovery Rate (FDR)
    counter = 0
    p_counter = 0
    # Perform analysis for each channel separately
    for channel_idx, (t_val, p_val) in enumerate(zip(t_values.T, p_values.T)):
      counter += 1
      if p_val < 0.05:
        print(p_val)
        p_counter += 1

    print(counter, p_counter)

def running_all():
    raw = data_loading()
    filtered_raw = filtering_and_pre_processing(raw)
    blocks_list = epocs_analysis(filtered_raw)
    statistical_analysis(blocks_list)

if __name__ == "__main__":
    running_all()