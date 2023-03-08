import matplotlib.colorbar
import matplotlib.colors
import matplotlib.pyplot as plt

import numpy as np

import os
import os.path as op

from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_test


#  ************* USEFUL FONCTIONS ******************

def create_epoch_with_montage(raw, info, montage):
    epochs = mne.EpochsArray(np.array(raw), info)
    epochs.set_montage(montage)
    return epochs


def extract_power_in_freq_band(epochs):
    spectrum = epochs.compute_psd(fmax = 50)
    psds, freqs = spectrum.get_data(return_freqs=True)
    freqbound1 = np.where(freqs>0)[0][0]
    freqbound2 = np.where(freqs>4)[0][0]
    freqbound3 = np.where(freqs>8)[0][0]
    freqbound4 = np.where(freqs>10)[0][0]
    freqbound5 = np.where(freqs>13)[0][0]
    freqbound6 = np.where(freqs>30)[0][0]
    freqbound7 = np.where(freqs>45)[0][0]
    psds_band = [psds[:, :, freqbound1:freqbound2].mean(axis=2), \
                 psds[:, :, freqbound2:freqbound3].mean(axis=2), \
                 psds[:, :, freqbound3:freqbound4].mean(axis=2), \
                 psds[:, :, freqbound4:freqbound5].mean(axis=2), \
                 psds[:, :, freqbound5:freqbound6].mean(axis=2), \
                 psds[:, :, freqbound6:freqbound7].mean(axis=2)]
    return spectrum, psds, freqs, psds_band


def averaging_extracting_converting(spectrum, psds):
    psdslog = 10 * np.log10(psds) # convert to dB
    avEp_psds = 10 * np.log10(spectrum.average().get_data()) # average across epochs, extract values, convert to dB
    avEp_psds_mean = avEp_psds.mean(axis=0)
    avEp_psds_std = avEp_psds.std(axis=0)
    return psdslog, avEp_psds, avEp_psds_mean, avEp_psds_std


def power_spectrum_stat_comparison_spatial_clusters(psds1, psds2, adjacency, pairing):
    spatial_clusters = list()
    for freq_band in np.arange(6):
        print('freq band : ', freq_band)
        if pairing == 1:
            t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(
                psds2[freq_band] - psds1[freq_band], out_type='mask', n_permutations=1024, n_jobs=6,
                verbose=False, adjacency=adjacency)
        else:
            t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(
                [psds1[freq_band], psds2[freq_band]], out_type='mask', n_permutations=1024, n_jobs=6,
                verbose=False, adjacency=adjacency)
        sign_clusters = np.where(cluster_pv <= .05)[0]
        print('sign_clusters : ', sign_clusters)
        mask_ti = np.array(clusters)[sign_clusters].sum(axis=0).astype('bool')
        print('mask_ti : ', mask_ti)
        spatial_clusters.append(mask_ti)
    spatial_clusters = np.array(spatial_clusters)
    return spatial_clusters


def power_spectrum_stat_comparison_freq_clusters(psds1, psds2, pairing, channel_choice):
    freq_clusters = list()
    if channel_choice == 'each_channel':
        for chan in np.arange(14):
            clusters2 = list()
            if pairing == 1:
                t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(
                    psds2[:,chan,:] - psds1[:,chan,:], out_type='mask', n_permutations=1024, n_jobs=6,
                    verbose=False, adjacency=None)
            else:
                t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(
                    [psds1[:,chan,:], psds2[:,chan,:]], out_type='mask', n_permutations=1024, n_jobs=6,
                    verbose=False, adjacency=None)
            sign_clusters = np.where(cluster_pv <= .05)[0]
            if sign_clusters.size > 0:
                if sign_clusters.size == 1:
                    clusters2 = [np.arange(clusters[sign_clusters[0]][0].start, clusters[sign_clusters[0]][0].stop)]
                else:
                    for nb_siclu, siclu in enumerate(sign_clusters):
                        if nb_siclu == 0:
                            clusters2 = np.arange(clusters[siclu][0].start, clusters[siclu][0].stop)
                        elif nb_siclu == 1:
                            clusters2 = [clusters2, np.arange(clusters[siclu][0].start, clusters[siclu][0].stop)] # c'est ici que ça change
                        else:
                            clusters2.append(np.arange(clusters[siclu][0].start, clusters[siclu][0].stop))
                freq_clusters.append(clusters2)
            else:
                freq_clusters.append([])

    elif channel_choice == 'all_together':
        clusters2 = list()
        if pairing == 1:
            t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(
                psds2.mean(axis=1) - psds1.mean(axis=1), out_type='mask', n_permutations=1024, n_jobs=6,
                verbose=False, adjacency=None)
        else:
            t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(
                [psds1.mean(axis=1), psds2.mean(axis=1)], out_type='mask', n_permutations=1024, n_jobs=6,
                verbose=False, adjacency=None)
        sign_clusters = np.where(cluster_pv <= .05)[0]
        if sign_clusters.size > 0:
            if sign_clusters.size == 1:
                clusters2 = [np.arange(clusters[sign_clusters[0]][0].start, clusters[sign_clusters[0]][0].stop)]
            else:
                for nb_siclu, siclu in enumerate(sign_clusters):
                    if nb_siclu == 0:
                        clusters2 = np.arange(clusters[siclu][0].start, clusters[siclu][0].stop)
                    elif nb_siclu == 1:
                        clusters2 = [clusters2, np.arange(clusters[siclu][0].start, clusters[siclu][0].stop)] # c'est ici que ça change
                    else:
                        clusters2.append(np.arange(clusters[siclu][0].start, clusters[siclu][0].stop))
            freq_clusters = clusters2

    else:
        if channel_choice == 'front_back':
            front_back_chan = [np.arange(0, 7), np.arange(7, 14)]
        elif channel_choice == 'left_right':
            front_back_chan = [[1, 3, 5, 7, 10, 12], [2, 4, 6, 8, 11, 13]]
        elif channel_choice == 'front_back_left_right':
            front_back_chan = [[1, 3, 5], [2, 4, 6], [7, 10, 12], [8, 11, 13]]
        for chan_group in front_back_chan:
            clusters2 = list()
            if pairing == 1:
                t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(
                    psds2[:,chan_group,:].mean(axis=1) - psds1[:,chan_group,:].mean(axis=1), out_type='mask', n_permutations=1024, n_jobs=6,
                    verbose=False, adjacency=None)
            else:
                t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(
                    [psds1[:,chan_group,:].mean(axis=1), psds2[:,chan_group,:].mean(axis=1)], out_type='mask', n_permutations=1024, n_jobs=6,
                    verbose=False, adjacency=None)
            sign_clusters = np.where(cluster_pv <= .05)[0]
            if sign_clusters.size > 0:
                if sign_clusters.size == 1:
                    clusters2 = [np.arange(clusters[sign_clusters[0]][0].start, clusters[sign_clusters[0]][0].stop)]
                else:
                    for nb_siclu, siclu in enumerate(sign_clusters):
                        if nb_siclu == 0:
                            clusters2 = np.arange(clusters[siclu][0].start, clusters[siclu][0].stop)
                        elif nb_siclu == 1:
                            clusters2 = [clusters2, np.arange(clusters[siclu][0].start, clusters[siclu][0].stop)] # c'est ici que ça change
                        else:
                            clusters2.append(np.arange(clusters[siclu][0].start, clusters[siclu][0].stop))
                freq_clusters.append(clusters2)
            else:
                freq_clusters.append([])

    return freq_clusters


def power_spectrum_right_left_stat_comparison_freq_clusters(psds1Alog, psds2Alog, psds1Blog, psds2Blog):

    # front_back_chan = [[1, 3, 5], [2, 4, 6], [7, 10, 12], [8, 11, 13]]
    # front_back_name = ['Front Left electrodes', 'Front Right electrodes', 'Back Left electrodes', 'Back Right electrodes']
    for psds_nb, psds in enumerate([psds1Alog, psds2Alog, psds1Blog, psds2Blog]):
        freq_clusters = list()
        for front_back in ['front', 'back']:
            if front_back == 'front':
                elec_left = [1, 3, 5]
                elec_right = [2, 4, 6]
            elif front_back == 'back':
                elec_left = [7, 10, 12]
                elec_right = [8, 11, 13]
            clusters2 = list()
            t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(
                psds[:,elec_left,:].mean(axis=1) - psds[:,elec_right,:].mean(axis=1), out_type='mask', n_permutations=1024, n_jobs=6,
                verbose=False, adjacency=None)
            sign_clusters = np.where(cluster_pv <= .05)[0]
            if sign_clusters.size > 0:
                if sign_clusters.size == 1:
                    clusters2 = [np.arange(clusters[sign_clusters[0]][0].start, clusters[sign_clusters[0]][0].stop)]
                else:
                    for nb_siclu, siclu in enumerate(sign_clusters):
                        if nb_siclu == 0:
                            clusters2 = np.arange(clusters[siclu][0].start, clusters[siclu][0].stop)
                        elif nb_siclu == 1:
                            clusters2 = [clusters2, np.arange(clusters[siclu][0].start, clusters[siclu][0].stop)] # c'est ici que ça change
                        else:
                            clusters2.append(np.arange(clusters[siclu][0].start, clusters[siclu][0].stop))

                freq_clusters.append(clusters2)
            else:
                freq_clusters.append([])
        if psds_nb == 0: freq_clusters_1A = freq_clusters
        elif psds_nb == 1: freq_clusters_2A = freq_clusters
        elif psds_nb == 2: freq_clusters_1B = freq_clusters
        elif psds_nb == 3: freq_clusters_2B = freq_clusters
    return freq_clusters_1A, freq_clusters_2A, freq_clusters_1B, freq_clusters_2B


def plot_topomap_spectrum_difference_reduced_single_fig(psds_band1A, psds_band2A, psds_band1B, psds_band2B,
                                                        spatial_clustersA, supertitle):

    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    subfigs = fig.subfigures(1, 6, wspace=0.2)

    norm = matplotlib.colors.CenteredNorm(vcenter=0)
    dic_mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=10)

    for freq_band in np.arange(6):
        if freq_band == 0:
            title = 'Delta \n (0-4 Hz)'
            pmax1 = 3e-2 # 8e-3
            pmax2 = 3e-2 # 2e-2 #-3e-3
            ticks1 = [0, 2e-2] #[0, 1e-2, 2e-2, 3e-2]
            ticks2 = [0, 2e-2]
        elif freq_band == 1:
            title = 'Theta \n (4-8 Hz)'
            pmax1 = 2.2e-3 # 1e-3
            pmax2 = 1.2e-3 # 6e-4
            ticks1 = [0, 2e-3]
            ticks2 = [0, 1e-3]
        elif freq_band == 2:
            title = 'Alpha 1 \n (8-10 Hz)'
            pmax1 = 6e-4 # 1e-3 # 8e-4
            pmax2 = 5e-4
            ticks1 = [0, 5e-4] # [0, 2e-4, 4e-4, 6e-4]
            ticks2 = [0, 4e-4]
        elif freq_band == 3:
            title = 'Alpha 2 \n (10-13 Hz)'
            pmax1 = 1.2e-3 #2e-3 # 1e-3 # 8e-4
            pmax2 = 3e-4 # 5e-4
            ticks1 = [0, 1e-3] # [0, 1e-3, 2e-3]
            ticks2 = [0, 2e-4]
        elif freq_band == 4:
            title = 'Beta \n (13-30 Hz)'
            pmax1 = 6e-4 # 3e-4
            pmax2 = 3e-4
            ticks1 = [0, 5e-4]
            ticks2 = [0, 2e-4]
        elif freq_band == 5:
            title = 'Gamma \n (30-45 Hz)'
            pmax1 = 6e-4 # 4e-4
            pmax2 = 3e-4
            ticks1 = [0, 5e-4]
            ticks2 = [0, 2e-4]

        axes = subfigs[freq_band].subplots(nrows=4, ncols=1) #(1, 2, sharey=True)
        subfigs[freq_band].suptitle(title, fontsize=20, weight='bold')

        pc1, cm1 = mne.viz.topomap.plot_topomap(psds_band1A[freq_band].mean(axis=0), pos=rawA.info, show=False,  outlines='head', sphere='eeglab', axes=axes[0], vlim=(0, pmax1), cmap='binary') #, cnorm=norm, cmap="RdBu_r")
        pc2, cm2 = mne.viz.topomap.plot_topomap(psds_band2A[freq_band].mean(axis=0), pos=rawA.info, show=False,  outlines='head', sphere='eeglab', axes=axes[1], vlim=(0, pmax1), cmap='binary') #, cnorm=norm, cmap="RdBu_r")
        pc3, cm3 = mne.viz.topomap.plot_topomap(psds_band2A[freq_band].mean(axis=0)-psds_band1A[freq_band].mean(axis=0), pos=rawA.info, mask=spatial_clustersA[freq_band].T, mask_params=dic_mask_params, show=False,  outlines='head',  sphere='eeglab', axes=axes[2], vlim=(0, pmax2), cmap='binary') #, cnorm=norm, cmap="RdBu_r")

        subfigs[freq_band].delaxes(axes[3])

        ax_x_start1 = 0.3 #0.35
        ax_x_start3 = 0.7
        ax_x_width = 0.08 # 0.05
        ax_y_start1 = 0.1
        ax_y_height = 0.08 #0.08

        cbar_ax = subfigs[freq_band].add_axes([ax_x_start1, ax_y_start1, ax_x_width, ax_y_height])
        clb = subfigs[freq_band].colorbar(pc1, cax=cbar_ax,  label='power', ticks=ticks1)
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))
        ticklabs = clb.ax.get_yticklabels()
        clb.set_label('power', size=15)

        cbar_ax = subfigs[freq_band].add_axes([ax_x_start3, ax_y_start1, ax_x_width, ax_y_height])
        clb = subfigs[freq_band].colorbar(pc3, cax=cbar_ax,  label='power \ndiff', ticks=ticks2)
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))
        ticklabs = clb.ax.get_yticklabels()
        clb.set_label('power \ndiff', size=15)

        if freq_band == 0:
            axes[0].set_ylabel('Start', fontsize=20, weight='bold')
            axes[1].set_ylabel('End', fontsize=20, weight='bold')
            axes[2].set_ylabel('End-Start', fontsize=20, weight='bold')
        else:
            axes[0].set_ylabel('Start', fontsize = 20, color='white')
            axes[1].set_ylabel('End', fontsize = 20, color='white')
            axes[2].set_ylabel('End-Start', fontsize = 20, color='white')

    figure = plt.gcf()
-    plt.savefig('figure3_topomap_reduced_'+supertitle+'.png', bbox_inches='tight')


def plot_topomap_spectrum_difference_single_fig(psds_band1A, psds_band2A, psds_band1B, psds_band2B,
                                                spatial_clusters1, spatial_clusters2, spatial_clusters3, spatial_clusters4, supertitle):

    fig = plt.figure(constrained_layout=True, figsize=(25, 15))
    subfigs = fig.subfigures(2, 3, wspace=0.07)

    for freq_band in np.arange(6):
        if freq_band == 0:
            title = 'Delta (0-4 Hz)'
            pmax1 = 3e-2 # 8e-3
            pmax2 = 2e-2 #-3e-3
            number1 = 0
            number2 = 0
            ticks1 = [0, 1e-2, 2e-2, 3e-2]
            ticks2 = [-2e-2, 0, 2e-2]
        elif freq_band == 1:
            title = 'Theta (4-8 Hz)'
            pmax1 = 2e-3 # 1e-3
            pmax2 = 1e-3 # 6e-4
            number1 = 0
            number2 = 1
            ticks1 = [0, 1e-3, 2e-3]
            ticks2 = [-1e-3, 0, 1e-3]
        elif freq_band == 2:
            title = 'Alpha 1 (8-10 Hz)'
            pmax1 = 6e-4 # 1e-3 # 8e-4
            pmax2 = 5e-4
            number1 = 0
            number2 = 2
            ticks1 = [0, 2e-4, 4e-4, 6e-4]
            ticks2 = [-5e-4, 0, 5e-4]
        elif freq_band == 3:
            title = 'Alpha 2 (10-13 Hz)'
            pmax1 = 2e-3 # 1e-3 # 8e-4
            pmax2 = 5e-4
            number1 = 1
            number2 = 0
            ticks1 = [0, 1e-3, 2e-3]
            ticks2 = [-5e-4, 0, 5e-4]
        elif freq_band == 4:
            title = 'Beta (13-30 Hz)'
            pmax1 = 6e-4 # 3e-4
            pmax2 = 3e-4
            number1 = 1
            number2 = 1
            ticks1 = [0, 2e-4, 4e-4, 6e-4]
            ticks2 = [-2e-4, 0, 2e-4]
        elif freq_band == 5:
            title = 'Gamma (30-45 Hz)'
            pmax1 = 6e-4 # 4e-4
            pmax2 = 3e-4
            number1 = 1
            number2 = 2
            ticks1 = [0, 2e-4, 4e-4, 6e-4]
            ticks2 = [-2e-4, 0, 2e-4]

        norm = matplotlib.colors.CenteredNorm(vcenter=0)

        axes = subfigs[number1, number2].subplots(nrows=3, ncols=3) #(1, 2, sharey=True)
        subfigs[number1, number2].suptitle(title, fontsize=20)

        dic_mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=10)
        pc1, cm1 = mne.viz.topomap.plot_topomap(psds_band1A[freq_band].mean(axis=0), pos=rawA.info, show=False,  outlines='head', sphere='eeglab', axes=axes[0][0], vlim=(0, pmax1)) #, cnorm=norm, cmap="RdBu_r")
        pc2, cm2 = mne.viz.topomap.plot_topomap(psds_band2A[freq_band].mean(axis=0), pos=rawA.info, show=False,  outlines='head', sphere='eeglab', axes=axes[0][1], vlim=(0, pmax1)) #, cnorm=norm, cmap="RdBu_r")
        pc3, cm3 = mne.viz.topomap.plot_topomap(psds_band2A[freq_band].mean(axis=0)-psds_band1A[freq_band].mean(axis=0), pos=rawA.info, mask=spatial_clusters1[freq_band].T, mask_params=dic_mask_params, show=False,  outlines='head',  sphere='eeglab', axes=axes[0][2], vlim=(-pmax2, pmax2)) #, cnorm=norm, cmap="RdBu_r")
        pc4, cm4 = mne.viz.topomap.plot_topomap(psds_band1B[freq_band].mean(axis=0), pos=rawA.info, show=False,  outlines='head',  sphere='eeglab', axes=axes[1][0], vlim=(0, pmax1)) #, cnorm=norm, cmap="RdBu_r")
        pc5, cm5 = mne.viz.topomap.plot_topomap(psds_band2B[freq_band].mean(axis=0), pos=rawA.info, show=False,  outlines='head',  sphere='eeglab', axes=axes[1][1], vlim=(0, pmax1)) #, cnorm=norm, cmap="RdBu_r")
        pc6, cm6 = mne.viz.topomap.plot_topomap(psds_band2B[freq_band].mean(axis=0)-psds_band1B[freq_band].mean(axis=0), pos=rawA.info, mask=spatial_clusters2[freq_band].T, mask_params=dic_mask_params, show=False,  outlines='head',  sphere='eeglab', axes=axes[1][2], vlim=(-pmax2, pmax2)) #, cnorm=norm, cmap="RdBu_r")
        pc7, cm7 = mne.viz.topomap.plot_topomap(psds_band1A[freq_band].mean(axis=0)-psds_band1B[freq_band].mean(axis=0), pos=rawA.info, mask=spatial_clusters3[freq_band].T, mask_params=dic_mask_params, show=False,  outlines='head',  sphere='eeglab', axes=axes[2][0], vlim=(-pmax2, pmax2)) #, cnorm=norm, cmap="RdBu_r")
        pc8, cm8 = mne.viz.topomap.plot_topomap(psds_band2A[freq_band].mean(axis=0)-psds_band2B[freq_band].mean(axis=0), pos=rawA.info, mask=spatial_clusters4[freq_band].T, mask_params=dic_mask_params, show=False,  outlines='head',  sphere='eeglab', axes=axes[2][1], vlim=(-pmax2, pmax2)) #, cnorm=norm, cmap="RdBu_r")

        ax_x_start1 = 0.75
        ax_x_start2 = 0.90
        ax_x_width = 0.03
        ax_y_start = 0.05
        ax_y_height = 0.15
        cbar_ax = subfigs[number1, number2].add_axes([ax_x_start1, ax_y_start, ax_x_width, ax_y_height])
        clb = subfigs[number1, number2].colorbar(pc1, cax=cbar_ax,  label='power', ticks=ticks1)
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))
        cbar_ax = subfigs[number1, number2].add_axes([ax_x_start2, ax_y_start, ax_x_width, ax_y_height])
        clb = subfigs[number1, number2].colorbar(pc7, cax=cbar_ax, label='power \n difference', ticks=ticks2)
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))

        axes[0][0].set_title('Start', fontsize = 15)
        axes[0][1].set_title('End', fontsize = 15)
        axes[0][2].set_title('End-Start', fontsize = 15)
        axes[0][0].set_ylabel('with \n TD', fontsize = 15)
        axes[1][0].set_ylabel('without \n TD', fontsize = 15)
        axes[2][0].set_ylabel('with - without \n TD', fontsize = 15)
        subfigs[number1, number2].delaxes(axes[2][2])

    fig.tight_layout
    figure = plt.gcf()
    plt.savefig('figure3_topomap_'+supertitle+'.png', bbox_inches='tight')


def plot_topomap_spectrum_difference(psds_band1A, psds_band2A, psds_band1B, psds_band2B,
                                     spatial_clusters1, spatial_clusters2, spatial_clusters3, spatial_clusters4):

    for freq_band in np.arange(6):
        if freq_band == 0:
            title = 'Delta (0-4 Hz)'
            pmax1 = 3e-2 # 8e-3
            pmax2 = 2e-2 #-3e-3
        elif freq_band == 1:
            title = 'Theta (4-8 Hz)'
            pmax1 = 2e-3 # 1e-3
            pmax2 = 1e-3 # 6e-4
        elif freq_band == 2:
            title = 'Alpha 1 (8-10 Hz)'
            pmax1 = 6e-4 # 1e-3 # 8e-4
            pmax2 = 5e-4
        elif freq_band == 3:
            title = 'Alpha 2 (10-13 Hz)'
            pmax1 = 2e-3 # 1e-3 # 8e-4
            pmax2 = 5e-4
        elif freq_band == 4:
            title = 'Beta (13-30 Hz)'
            pmax1 = 6e-4 # 3e-4
            pmax2 = 3e-4
        elif freq_band == 5:
            title = 'Gamma (30-45 Hz)'
            pmax1 = 6e-4 # 4e-4
            pmax2 = 3e-4
        norm = matplotlib.colors.CenteredNorm(vcenter=0)
        fig, axes = plt.subplots(nrows=3, ncols=3)
        pc1, cm1 = mne.viz.topomap.plot_topomap(psds_band1A[freq_band].mean(axis=0), pos=rawA.info, show=False,  axes=axes[0][0]) #, vlim=(0, pmax1)) #, cnorm=norm, cmap="RdBu_r")
        pc2, cm2 = mne.viz.topomap.plot_topomap(psds_band2A[freq_band].mean(axis=0), pos=rawA.info, show=False,  axes=axes[0][1]) #, vlim=(0, pmax1)) #, cnorm=norm, cmap="RdBu_r")
        pc3, cm3 = mne.viz.topomap.plot_topomap(psds_band2A[freq_band].mean(axis=0)-psds_band1A[freq_band].mean(axis=0), pos=rawA.info, mask=spatial_clusters1[freq_band].T, show=False,  axes=axes[0][2]) #, vlim=(-pmax2, pmax2)) #, cnorm=norm, cmap="RdBu_r")
        pc4, cm4 = mne.viz.topomap.plot_topomap(psds_band1B[freq_band].mean(axis=0), pos=rawA.info, show=False,  axes=axes[1][0]) #, vlim=(0, pmax1)) #, cnorm=norm, cmap="RdBu_r")
        pc5, cm5 = mne.viz.topomap.plot_topomap(psds_band2B[freq_band].mean(axis=0), pos=rawA.info, show=False,  axes=axes[1][1]) #, vlim=(0, pmax1)) #, cnorm=norm, cmap="RdBu_r")
        pc6, cm6 = mne.viz.topomap.plot_topomap(psds_band2B[freq_band].mean(axis=0)-psds_band1B[freq_band].mean(axis=0), pos=rawA.info, mask=spatial_clusters2[freq_band].T, show=False,  axes=axes[1][2]) #, vlim=(-pmax2, pmax2)) #, cnorm=norm, cmap="RdBu_r")
        pc7, cm7 = mne.viz.topomap.plot_topomap(psds_band1A[freq_band].mean(axis=0)-psds_band1B[freq_band].mean(axis=0), pos=rawA.info, mask=spatial_clusters3[freq_band].T, show=False,  axes=axes[2][0]) #, vlim=(-pmax2, pmax2)) #, cnorm=norm, cmap="RdBu_r")
        pc8, cm8 = mne.viz.topomap.plot_topomap(psds_band2A[freq_band].mean(axis=0)-psds_band2B[freq_band].mean(axis=0), pos=rawA.info, mask=spatial_clusters4[freq_band].T, show=False,  axes=axes[2][1]) #, vlim=(-pmax2, pmax2)) #, cnorm=norm, cmap="RdBu_r")

        ax_x_start1 = 0.68
        ax_x_start2 = 0.82
        ax_x_width = 0.03
        ax_y_start = 0.15
        ax_y_height = 0.15
        cbar_ax = fig.add_axes([ax_x_start1, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(pc1, cax=cbar_ax,  label='power') #, ticks=[0, pmax1])
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))
        cbar_ax = fig.add_axes([ax_x_start2, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(pc7, cax=cbar_ax, label='power \n difference') #, ticks=[-pmax2, 0, pmax2])
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))

        cbar_ax = fig.add_axes([0.3, 0.9, ax_x_width, ax_y_height])
        clb = fig.colorbar(pc1, cax=cbar_ax,  label='power') #, ticks=[0, pmax])
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))
        cbar_ax = fig.add_axes([0.6, 0.9, ax_x_width, ax_y_height])
        clb = fig.colorbar(pc2, cax=cbar_ax, label='power difference') #, ticks=[pmin, 0, pmax])
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))
        cbar_ax = fig.add_axes([0.9, 0.9, ax_x_width, ax_y_height])
        clb = fig.colorbar(pc3, cax=cbar_ax,  label='power') #, ticks=[0, pmax])
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))
        cbar_ax = fig.add_axes([0.3, 0.6, ax_x_width, ax_y_height])
        clb = fig.colorbar(pc4, cax=cbar_ax,  label='power') #, ticks=[0, pmax])
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))
        cbar_ax = fig.add_axes([0.6, 0.6, ax_x_width, ax_y_height])
        clb = fig.colorbar(pc5, cax=cbar_ax,  label='power') #, ticks=[0, pmax])
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))
        cbar_ax = fig.add_axes([0.9, 0.6, ax_x_width, ax_y_height])
        clb = fig.colorbar(pc6, cax=cbar_ax,  label='power') #, ticks=[0, pmax])
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))
        cbar_ax = fig.add_axes([0.3, 0.3, ax_x_width, ax_y_height])
        clb = fig.colorbar(pc7, cax=cbar_ax,  label='power') #, ticks=[0, pmax])
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))
        cbar_ax = fig.add_axes([0.6, 0.3, ax_x_width, ax_y_height])
        clb = fig.colorbar(pc8, cax=cbar_ax,  label='power') #, ticks=[0, pmax])
        clb.ax.yaxis.set_label_position('left')
        clb.formatter.set_powerlimits((0, 0))

        axes[0][0].set_title('Start', fontsize = 10)
        axes[0][1].set_title('End', fontsize = 10)
        axes[0][2].set_title('End-Start', fontsize = 10)
        axes[0][0].set_ylabel('with \n TD', fontsize = 10)
        axes[1][0].set_ylabel('without \n TD', fontsize = 10)
        axes[2][0].set_ylabel('with - without \n TD', fontsize = 10)

        fig.tight_layout
        fig.delaxes(axes[2][2])
        figure = plt.gcf()
        os.makedirs('power_topomap', exist_ok=True)
        plt.savefig('power_topomap/'+title+'.png', bbox_inches='tight')



def plot_figure_power_spectrum_single_plot(freqs1A, freqs2A, freqs1B, freqs2B,
                                           values_mean1A, values_mean2A, values_mean1B, values_mean2B,
                                           name):
    fig = plt.subplots(1,1)
    plt.plot(freqs1A, values_mean1A, label='W/ Start', color='b')
    plt.plot(freqs2A, values_mean2A, label='W/ End', color='c')
    plt.plot(freqs1B, values_mean1B, label='W/O Start', color='r')
    plt.plot(freqs2B, values_mean2B, label='W/O End', color='m')
    plt.legend()
    # plt.show()
    figure1 = plt.gcf()
    figure1.set_size_inches(24, 12)
    os.makedirs('power_spectrum_single_plot', exist_ok=True)
    plt.savefig('power_spectrum_single_plot/'+name+'.png', bbox_inches='tight')


def plot_figure_power_spectrum_single_fig(freqs1A, freqs2A, freqs1B, freqs2B,
                                          psds1Alog, psds2Alog, psds1Blog, psds2Blog,
                                          freq_clustersA, freq_clustersB, freq_clusters1, freq_clusters2,
                                          front_back_chan, front_back_name, channel_choice, supertitle):

    if channel_choice == 'front_back':
        fig = plt.figure(constrained_layout=True, figsize=(20, 20))
        subfigs = fig.subfigures(2, 1, wspace=0.07)
    if channel_choice == 'left_right':
        fig = plt.figure(constrained_layout=True, figsize=(40, 20))
        subfigs = fig.subfigures(1, 2, wspace=0.07)
    if channel_choice == 'front_back_left_right':
        fig = plt.figure(constrained_layout=True, figsize=(40, 40))
        subfigs = fig.subfigures(2, 2, wspace=0.07)

    for chan_group_nb, chan_group in enumerate(front_back_chan):

        values_mean1A = psds1Alog[:, chan_group, :].mean(axis=1).mean(axis=0)
        values_mean2A = psds2Alog[:, chan_group, :].mean(axis=1).mean(axis=0)
        values_mean1B = psds1Blog[:, chan_group, :].mean(axis=1).mean(axis=0)
        values_mean2B = psds2Blog[:, chan_group, :].mean(axis=1).mean(axis=0)
        values_std1A = psds1Alog[:, chan_group, :].mean(axis=1).std(axis=0)
        values_std2A = psds2Alog[:, chan_group, :].mean(axis=1).std(axis=0)
        values_std1B = psds1Blog[:, chan_group, :].mean(axis=1).std(axis=0)
        values_std2B = psds2Blog[:, chan_group, :].mean(axis=1).std(axis=0)
        values_freq_clustersA = freq_clustersA[chan_group_nb]
        values_freq_clustersB = freq_clustersB[chan_group_nb]
        values_freq_clusters1 = freq_clusters1[chan_group_nb]
        values_freq_clusters2 = freq_clusters2[chan_group_nb]
        name = front_back_name[chan_group_nb]

        if channel_choice == 'front_back_left_right':
            if chan_group_nb < 2:
                axes = subfigs[0][chan_group_nb].subplots(nrows=2, ncols=2) #(1, 2, sharey=True)
                subfigs[0][chan_group_nb].suptitle(name, fontsize=30)
            else:
                axes = subfigs[1][chan_group_nb-2].subplots(nrows=2, ncols=2) #(1, 2, sharey=True)
                subfigs[1][chan_group_nb-2].suptitle(name, fontsize=30)

        else:
            axes = subfigs[chan_group_nb].subplots(nrows=2, ncols=2) #(1, 2, sharey=True)
            subfigs[chan_group_nb].suptitle(name, fontsize=30)

        axes[0][0].plot(freqs1A, values_mean1A, color='b', label='with TD, start')
        axes[0][0].plot(freqs2A, values_mean2A, color='c', label='with TD, end')
        axes[0][1].plot(freqs1B, values_mean1B, color='r', label='without TD, start')
        axes[0][1].plot(freqs2B, values_mean2B, color='m', label='without TD, end')
        axes[1][0].plot(freqs1A, values_mean1A, color='b', label='with TD, start')
        axes[1][0].plot(freqs1B, values_mean1B, color='r', label='without TD, start')
        axes[1][1].plot(freqs2A, values_mean2A, color='c', label='with TD, end')
        axes[1][1].plot(freqs2B, values_mean2B, color='m', label='without TD, end')
        axes[0][0].fill_between(freqs1A, values_mean1A - values_std1A, values_mean1A + values_std1A,
                        color='b', alpha=.1, edgecolor='none')
        axes[0][0].fill_between(freqs2A, values_mean2A - values_std2A, values_mean2A + values_std2A,
                        color='c', alpha=.1, edgecolor='none')
        axes[0][1].fill_between(freqs1B, values_mean1B - values_std1B, values_mean1B + values_std1B,
                        color='r', alpha=.1, edgecolor='none')
        axes[0][1].fill_between(freqs2B, values_mean2B - values_std2B, values_mean2B + values_std2B,
                        color='m', alpha=.1, edgecolor='none')
        axes[1][0].fill_between(freqs1A, values_mean1A - values_std1A, values_mean1A + values_std1A,
                        color='b', alpha=.1, edgecolor='none')
        axes[1][0].fill_between(freqs1B, values_mean1B - values_std1B, values_mean1B + values_std1B,
                        color='r', alpha=.1, edgecolor='none')
        axes[1][1].fill_between(freqs2A, values_mean2A - values_std2A, values_mean2A + values_std2A,
                        color='c', alpha=.1, edgecolor='none')
        axes[1][1].fill_between(freqs2B, values_mean2B - values_std2B, values_mean2B + values_std2B,
                        color='m', alpha=.1, edgecolor='none')
        for clus in values_freq_clustersA:
            axes[0][0].fill_between(freqs1A[clus], values_mean1A[clus] - values_std1A[clus], values_mean1A[clus] + values_std1A[clus],
                            color='b', alpha=.5, edgecolor='none')
            axes[0][0].fill_between(freqs2A[clus], values_mean2A[clus] - values_std2A[clus], values_mean2A[clus] + values_std2A[clus],
                            color='c', alpha=.5, edgecolor='none')
        for clus in values_freq_clustersB:
            axes[0][1].fill_between(freqs1B[clus], values_mean1B[clus] - values_std1B[clus], values_mean1B[clus] + values_std1B[clus],
                            color='r', alpha=.5, edgecolor='none')
            axes[0][1].fill_between(freqs2B[clus], values_mean2B[clus] - values_std2B[clus], values_mean2B[clus] + values_std2B[clus],
                            color='m', alpha=.5, edgecolor='none')
        for clus in values_freq_clusters1:
            axes[1][0].fill_between(freqs1A[clus], values_mean1A[clus] - values_std1A[clus], values_mean1A[clus] + values_std1A[clus],
                            color='b', alpha=.5, edgecolor='none')
            axes[1][0].fill_between(freqs1B[clus], values_mean1B[clus] - values_std1B[clus], values_mean1B[clus] + values_std1B[clus],
                            color='r', alpha=.5, edgecolor='none')
        for clus in values_freq_clusters2:
            axes[1][1].fill_between(freqs2A[clus], values_mean2A[clus] - values_std2A[clus], values_mean2A[clus] + values_std2A[clus],
                            color='c', alpha=.5, edgecolor='none')
            axes[1][1].fill_between(freqs2B[clus], values_mean2B[clus] - values_std2B[clus], values_mean2B[clus] + values_std2B[clus],
                            color='m', alpha=.5, edgecolor='none')
        axes[0][0].yaxis.set_ticks(np.arange(-50, -9, 20))
        axes[0][1].yaxis.set_ticks(np.arange(-50, -9, 20))
        axes[1][0].yaxis.set_ticks(np.arange(-50, -9, 20))
        axes[1][1].yaxis.set_ticks(np.arange(-50, -9, 20))
        axes[0][0].set_ylim([-50, -10])
        axes[0][1].set_ylim([-50, -10])
        axes[1][0].set_ylim([-50, -10])
        axes[1][1].set_ylim([-50, -10])
        axes[0][0].tick_params(axis='both', which='major', labelsize=15)
        axes[0][1].tick_params(axis='both', which='major', labelsize=15)
        axes[1][0].tick_params(axis='both', which='major', labelsize=15)
        axes[1][1].tick_params(axis='both', which='major', labelsize=15)
        axes[0][0].legend(fontsize=25)
        axes[0][1].legend(fontsize=25)
        axes[1][0].legend(fontsize=25)
        axes[1][1].legend(fontsize=25)
        axes[1][0].set_xlabel('Frequency (Hz)', fontsize = 25)
        axes[1][1].set_xlabel('Frequency (Hz)', fontsize = 25)
        axes[0][0].set_ylabel('PSD (dB)', fontsize = 25)
        axes[1][0].set_ylabel('PSD (dB)', fontsize = 25)

    figure2 = plt.gcf()
    plt.savefig('figure2_spectrum_'+ channel_choice + '_'+ supertitle+'.png', bbox_inches='tight')


def plot_figure_power_spectrum_right_left_single_fig(freqs1A, freqs2A, freqs1B, freqs2B,
                                                     psds1Alog, psds2Alog, psds1Blog, psds2Blog,
                                                     freq_clusters_1A, freq_clusters_2A, freq_clusters_1B, freq_clusters_2B,
                                                     front_back_chan, front_back_name, supertitle):

# front_back_chan = [[1, 3, 5], [2, 4, 6], [7, 10, 12], [8, 11, 13]]
# front_back_name = ['Front Left electrodes', 'Front Right electrodes', 'Back Left electrodes', 'Back Right electrodes']

    fig = plt.figure(constrained_layout=True, figsize=(20, 20))
    subfigs = fig.subfigures(2, 1, wspace=0.07)

    for chan_group_nb, chan_group_name in enumerate(['Frontal electrodes', 'Backward electrodes']):

        if chan_group_name == 'Frontal electrodes':
            left_electrodes = [1, 3, 5]
            right_electrodes = [2, 4, 6]
        elif chan_group_name == 'Backward electrodes':
            left_electrodes = [7, 10, 12]
            right_electrodes = [8, 11, 13]

        values_mean1A_left = psds1Alog[:, left_electrodes, :].mean(axis=1).mean(axis=0)
        values_mean2A_left = psds2Alog[:, left_electrodes, :].mean(axis=1).mean(axis=0)
        values_mean1B_left = psds1Blog[:, left_electrodes, :].mean(axis=1).mean(axis=0)
        values_mean2B_left = psds2Blog[:, left_electrodes, :].mean(axis=1).mean(axis=0)
        values_mean1A_right = psds1Alog[:, right_electrodes, :].mean(axis=1).mean(axis=0)
        values_mean2A_right = psds2Alog[:, right_electrodes, :].mean(axis=1).mean(axis=0)
        values_mean1B_right = psds1Blog[:, right_electrodes, :].mean(axis=1).mean(axis=0)
        values_mean2B_right = psds2Blog[:, right_electrodes, :].mean(axis=1).mean(axis=0)
        values_std1A_left = psds1Alog[:, left_electrodes, :].mean(axis=1).std(axis=0)
        values_std2A_left = psds2Alog[:, left_electrodes, :].mean(axis=1).std(axis=0)
        values_std1B_left = psds1Blog[:, left_electrodes, :].mean(axis=1).std(axis=0)
        values_std2B_left = psds2Blog[:, left_electrodes, :].mean(axis=1).std(axis=0)
        values_std1A_right = psds1Alog[:, right_electrodes, :].mean(axis=1).std(axis=0)
        values_std2A_right = psds2Alog[:, right_electrodes, :].mean(axis=1).std(axis=0)
        values_std1B_right = psds1Blog[:, right_electrodes, :].mean(axis=1).std(axis=0)
        values_std2B_right = psds2Blog[:, right_electrodes, :].mean(axis=1).std(axis=0)
        values_freq_clusters1A = freq_clusters_1A[chan_group_nb]
        values_freq_clusters2A = freq_clusters_2A[chan_group_nb]
        values_freq_clusters1B = freq_clusters_1B[chan_group_nb]
        values_freq_clusters2B = freq_clusters_2B[chan_group_nb]

        axes = subfigs[chan_group_nb].subplots(nrows=2, ncols=2) #(1, 2, sharey=True)
        subfigs[chan_group_nb].suptitle(chan_group_name, fontsize=30)

        axes[0][0].plot(freqs2A, values_mean1A_right, color='darkblue', label='right')
        axes[0][0].plot(freqs1A, values_mean1A_left, color='b', label='left')
        axes[0][1].plot(freqs2B, values_mean2A_right, color='teal', label='right')
        axes[0][1].plot(freqs1B, values_mean2A_left, color='c', label='left')
        axes[1][0].plot(freqs1B, values_mean1B_right, color='brown', label='right')
        axes[1][0].plot(freqs1A, values_mean1B_left, color='r', label='left')
        axes[1][1].plot(freqs2B, values_mean2B_right, color='darkviolet', label='right')
        axes[1][1].plot(freqs2A, values_mean2B_left, color='m', label='left')
        axes[0][0].fill_between(freqs1A, values_mean1A_left - values_std1A_left, values_mean1A_left + values_std1A_left,
                        color='b', alpha=.1, edgecolor='none')
        axes[0][0].fill_between(freqs1A, values_mean1A_right - values_std1A_right, values_mean1A_right + values_std1A_right,
                        color='darkblue', alpha=.1, edgecolor='none')
        axes[0][1].fill_between(freqs2A, values_mean2A_left - values_std2A_left, values_mean2A_left + values_std2A_left,
                        color='c', alpha=.1, edgecolor='none')
        axes[0][1].fill_between(freqs2A, values_mean2A_right - values_std2A_right, values_mean2A_right + values_std2A_right,
                        color='teal', alpha=.1, edgecolor='none')
        axes[1][0].fill_between(freqs1B, values_mean1B_left - values_std1B_left, values_mean1B_left + values_std1B_left,
                        color='r', alpha=.1, edgecolor='none')
        axes[1][0].fill_between(freqs1B, values_mean1B_right - values_std1B_right, values_mean1B_right + values_std1B_right,
                        color='brown', alpha=.1, edgecolor='none')
        axes[1][1].fill_between(freqs2B, values_mean2B_left - values_std2B_left, values_mean2B_left + values_std2B_left,
                        color='m', alpha=.1, edgecolor='none')
        axes[1][1].fill_between(freqs2B, values_mean2B_right - values_std2B_right, values_mean2B_right + values_std2B_right,
                        color='darkviolet', alpha=.1, edgecolor='none')
        for clus in values_freq_clusters1A:
            axes[0][0].fill_between(freqs1A[clus], values_mean1A_left[clus] - values_std1A_left[clus], values_mean1A_left[clus] + values_std1A_left[clus],
                            color='b', alpha=.5, edgecolor='none')
            axes[0][0].fill_between(freqs1A[clus], values_mean1A_right[clus] - values_std1A_right[clus], values_mean1A_right[clus] + values_std1A_right[clus],
                            color='darkblue', alpha=.5, edgecolor='none')
        for clus in values_freq_clusters2A:
            axes[0][1].fill_between(freqs2A[clus], values_mean2A_left[clus] - values_std2A_left[clus], values_mean2A_left[clus] + values_std2A_left[clus],
                            color='c', alpha=.5, edgecolor='none')
            axes[0][1].fill_between(freqs2A[clus], values_mean2A_right[clus] - values_std2A_right[clus], values_mean2A_right[clus] + values_std2A_right[clus],
                            color='teal', alpha=.5, edgecolor='none')
        for clus in values_freq_clusters1B:
            axes[1][0].fill_between(freqs1B[clus], values_mean1B_left[clus] - values_std1B_left[clus], values_mean1B_left[clus] + values_std1B_left[clus],
                            color='r', alpha=.5, edgecolor='none')
            axes[1][0].fill_between(freqs1B[clus], values_mean1B_right[clus] - values_std1B_right[clus], values_mean1B_right[clus] + values_std1B_right[clus],
                            color='brown', alpha=.5, edgecolor='none')
        for clus in values_freq_clusters2B:
            axes[1][1].fill_between(freqs2B[clus], values_mean2B_left[clus] - values_std2B_left[clus], values_mean2B_left[clus] + values_std2B_left[clus],
                            color='m', alpha=.5, edgecolor='none')
            axes[1][1].fill_between(freqs2B[clus], values_mean2B_right[clus] - values_std2B_right[clus], values_mean2B_right[clus] + values_std2B_right[clus],
                            color='darkviolet', alpha=.5, edgecolor='none')
        axes[0][0].yaxis.set_ticks(np.arange(-50, -9, 20))
        axes[0][1].yaxis.set_ticks(np.arange(-50, -9, 20))
        axes[1][0].yaxis.set_ticks(np.arange(-50, -9, 20))
        axes[1][1].yaxis.set_ticks(np.arange(-50, -9, 20))
        axes[0][0].set_ylim([-50, -10])
        axes[0][1].set_ylim([-50, -10])
        axes[1][0].set_ylim([-50, -10])
        axes[1][1].set_ylim([-50, -10])
        axes[0][0].tick_params(axis='both', which='major', labelsize=15)
        axes[0][1].tick_params(axis='both', which='major', labelsize=15)
        axes[1][0].tick_params(axis='both', which='major', labelsize=15)
        axes[1][1].tick_params(axis='both', which='major', labelsize=15)
        axes[0][0].legend(fontsize=25)
        axes[0][1].legend(fontsize=25)
        axes[1][0].legend(fontsize=25)
        axes[1][1].legend(fontsize=25)
        axes[0][0].set_title('With TD, start', fontsize = 25)
        axes[0][1].set_title('With TD, end', fontsize = 25)
        axes[1][0].set_title('Without TD, start', fontsize = 25)
        axes[1][1].set_title('Without TD, end', fontsize = 25)
        axes[1][0].set_xlabel('Frequency (Hz)', fontsize = 25)
        axes[1][1].set_xlabel('Frequency (Hz)', fontsize = 25)
        axes[0][0].set_ylabel('PSD (dB)', fontsize = 25)
        axes[1][0].set_ylabel('PSD (dB)', fontsize = 25)

    figure2 = plt.gcf()
    plt.savefig('figure2bis_spectrum_Left_vs_Rigt_'+ supertitle+'.png', bbox_inches='tight')


def plot_figure_power_spectrum(freqs1A, freqs2A, freqs1B, freqs2B,
                               values_mean1A, values_mean2A, values_mean1B, values_mean2B,
                               values_std1A, values_std2A, values_std1B, values_std2B,
                               freq_clustersA, freq_clustersB, freq_clusters1, freq_clusters2,
                               name):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(freqs2A, values_mean2A, color='c', label='end')
    ax1.plot(freqs1A, values_mean1A, color='b', label='start')
    ax2.plot(freqs2B, values_mean2B, color='m', label='end')
    ax2.plot(freqs1B, values_mean1B, color='r', label='start')
    ax3.plot(freqs1B, values_mean1B, color='r', label='without TD')
    ax3.plot(freqs1A, values_mean1A, color='b', label='with TD')
    ax4.plot(freqs2A, values_mean2A, color='c', label='with TD')
    ax4.plot(freqs2B, values_mean2B, color='m', label='without TD')
    ax1.fill_between(freqs1A, values_mean1A - values_std1A, values_mean1A + values_std1A,
                    color='b', alpha=.1, edgecolor='none')
    ax1.fill_between(freqs2A, values_mean2A - values_std2A, values_mean2A + values_std2A,
                    color='c', alpha=.1, edgecolor='none')
    ax2.fill_between(freqs1B, values_mean1B - values_std1B, values_mean1B + values_std1B,
                    color='r', alpha=.1, edgecolor='none')
    ax2.fill_between(freqs2B, values_mean2B - values_std2B, values_mean2B + values_std2B,
                    color='m', alpha=.1, edgecolor='none')
    ax3.fill_between(freqs1A, values_mean1A - values_std1A, values_mean1A + values_std1A,
                    color='b', alpha=.1, edgecolor='none')
    ax3.fill_between(freqs1B, values_mean1B - values_std1B, values_mean1B + values_std1B,
                    color='r', alpha=.1, edgecolor='none')
    ax4.fill_between(freqs2A, values_mean2A - values_std2A, values_mean2A + values_std2A,
                    color='c', alpha=.1, edgecolor='none')
    ax4.fill_between(freqs2B, values_mean2B - values_std2B, values_mean2B + values_std2B,
                    color='m', alpha=.1, edgecolor='none')

    for clus in freq_clustersA:
        ax1.plot([freqs1A[clus[0]], freqs1A[clus[-1]]], [-48, -48], 'k', linewidth=3)
    for clus in freq_clustersB:
        ax1.plot([freqs1A[clus[0]], freqs1A[clus[-1]]], [-48, -48], 'k', linewidth=3)
    for clus in freq_clusters1:
        ax1.plot([freqs1A[clus[0]], freqs1A[clus[-1]]], [-48, -48], 'k', linewidth=3)
    for clus in freq_clusters2:
        ax1.plot([freqs1A[clus[0]], freqs1A[clus[-1]]], [-48, -48], 'k', linewidth=3)

    ax1.yaxis.set_ticks(np.arange(-50, -9, 10))
    ax2.yaxis.set_ticks(np.arange(-50, -9, 10))
    ax3.yaxis.set_ticks(np.arange(-50, -9, 10))
    ax4.yaxis.set_ticks(np.arange(-50, -9, 10))
    ax1.set_ylim([-50, -10])
    ax2.set_ylim([-50, -10])
    ax3.set_ylim([-50, -10])
    ax4.set_ylim([-50, -10])
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax4.tick_params(axis='both', which='major', labelsize=15)
    legend_properties = {'size':16, 'weight':'bold'}
    ax1.legend(prop=legend_properties)
    ax2.legend(prop=legend_properties)
    ax3.legend(prop=legend_properties)
    ax4.legend(prop=legend_properties)
    ax1.set_title('With Topographical Disorientation (TD)', fontsize = 20, weight='bold')
    ax2.set_title('Without Topographical Disorientation (TD)', fontsize = 20, weight='bold')
    ax3.set_title('Start of the recordings', fontsize = 20, weight='bold')
    ax4.set_title('End of the recordings', fontsize = 20, weight='bold')
    ax3.set_xlabel('Frequency (Hz)', fontsize = 15)
    ax4.set_xlabel('Frequency (Hz)', fontsize = 15)
    ax1.set_ylabel('Power Spectral Density (dB)', fontsize = 15)
    ax3.set_ylabel('Power Spectral Density (dB)', fontsize = 15)
    figure2 = plt.gcf()
    figure2.set_size_inches(24, 12)
    os.makedirs('power_spectrum', exist_ok=True)
    plt.savefig('power_spectrum/'+name+'.png', bbox_inches='tight')


def spatio_temporal_cluster_test_and_plot(psdsX, psdsY, freqs, epochs, name1, name2, p_accept):

    # transpose to have the data with this order: Trials, Time, Channels
    X = [np.transpose(psdsX, (0, 2, 1)), np.transpose(psdsY, (0, 2, 1))]
    
    # Compute the agency matrix
    adjacency, ch_names = find_ch_adjacency(epochs.info, ch_type='eeg')
    mne.viz.plot_ch_adjacency(epochs.info, adjacency, ch_names)

    # Permutation test
    cluster_stats = spatio_temporal_cluster_test(X, n_permutations=1000,
                                                 tail=1 ,
                                                 n_jobs=None, buffer_size=None,
                                                 adjacency=adjacency)
    F_obs, clusters, cluster_pv, _ = cluster_stats

    # *** visualize clusters ***
    good_cluster_inds = np.where(cluster_pv < p_accept)[0]

    # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        freq_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        freq_inds = np.unique(freq_inds)

        # get topography for F stat
        f_map = F_obs[freq_inds, ...].mean(axis=0)

        # get signals at the sensors contributing to the cluster
        sig_freqs = freqs[freq_inds]

        # create spatial mask
        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

        # plot average test statistic and mark significant sensors
        f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs.info, tmin=0)
        f_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='Reds',
                            vlim=(np.min, np.max), show=False,
                            colorbar=False, mask_params=dict(markersize=10))
        image = ax_topo.images[0]

        # remove the title that would otherwise say "0.000 s"
        ax_topo.set_title("")

        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            'Averaged F-map ({:0.0f} - {:0.0f} Hz)'.format(*sig_freqs[[0, -1]]))

        # add new axis for freqs and plot spectrum
        ax_signals = divider.append_axes('right', size='300%', pad=1.2)
        title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
        if len(ch_inds) > 1:
            title += "s (mean)"
        ax_signals.plot(freqs, 10*np.log10(psdsX.mean(axis=0)[ch_inds].mean(axis=0)), label=name1)
        ax_signals.plot(freqs, 10*np.log10(psdsY.mean(axis=0)[ch_inds].mean(axis=0)), label=name2)
        ax_signals.set_xlabel('frequency (Hz)')
        ax_signals.set_ylabel('Power Spectrum Density (dB)')
        ax_signals.set_title(title)
        ax_signals.legend()

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), sig_freqs[0], sig_freqs[-1],
                                color='orange', alpha=0.3)

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)
        plt.savefig('spatio_temporal_cluster_'+name1+'_'+name2+'_Cluster'+str(i_clu)+'.png', bbox_inches='tight')


# *************** MAIN SCRIPT ****************

analysis0 = False
analysis1 = False
analysis2 = True
analysis3 = False
analysis4 = False

# Choice of how to cut the files
beg = 2 # 0.5
end = 10+beg # 5+beg
supertitle = 'remove2_consider10' # 'remove0.5_consider5'

# Path
data_path = '/Users/marinevernet/Documents/lab_Lyon/ARTICLES/5_topo_amnesia/synchronized_files_reref'

# Runs in which the patient reported to be lost
runsA = ['Klinik02.set','Klinik04.set','Klinik05.set','Klinik06.set',\
    'Klinik07.set','Klinik08.set','Klinik12.set','Klinik14.set',\
    'Strecke06.set','Strecke07.set','Strecke08.set','Strecke09.set','Strecke12.set',\
    'Google5.set','Google6.set']

# Runs in which the patient did not report to be lost
runsB = ['Klinik01.set', 'Klinik10.set','Klinik11.set','Klinik13.set','Klinik15.set', \
    'Strecke01.set','Strecke02.set','Strecke03.set','Strecke04.set','Strecke05.set',\
    'Strecke10.set','Strecke11.set','Strecke13.set','Strecke14.set',\
    'Google1.set','Google2.set','Google3.set','Google4.set','Google7.set','Google8.set']

# Loading and concatenating the files
raw1A = list()
raw2A = list()
raw1B = list()
raw2B = list()
for run in runsA:
    run_to_load = op.join(data_path, run)
    rawA = mne.io.read_raw_eeglab(run_to_load)
    raw1A.append(rawA.copy().crop(beg,end)._data)
    raw2A.append(rawA.copy().crop(rawA.times[-1]-end, rawA.times[-1]-beg)._data)
for run in runsB:
    run_to_load = op.join(data_path, run)
    rawB = mne.io.read_raw_eeglab(run_to_load)
    raw1B.append(rawB.copy().crop(beg,end)._data)
    raw2B.append(rawB.copy().crop(rawB.times[-1]-end, rawB.times[-1]-beg)._data)

# Create epochs
info = mne.create_info(ch_names=rawA.ch_names, sfreq=rawA.info['sfreq'], ch_types='eeg')
montage = mne.channels.DigMontage(dig=rawA.info['dig'], ch_names = rawA.ch_names)
epochs1A = create_epoch_with_montage(raw1A, info, montage)
epochs2A = create_epoch_with_montage(raw2A, info, montage)
epochs1B = create_epoch_with_montage(raw1B, info, montage)
epochs2B = create_epoch_with_montage(raw2B, info, montage)

# Calculate power spectrum
spectrum1A, psds1A, freqs1A, psds_band1A = extract_power_in_freq_band(epochs1A)
spectrum2A, psds2A, freqs2A, psds_band2A = extract_power_in_freq_band(epochs2A)
spectrum1B, psds1B, freqs1B, psds_band1B = extract_power_in_freq_band(epochs1B)
spectrum2B, psds2B, freqs2B, psds_band2B = extract_power_in_freq_band(epochs2B)

# ******* Analysis 0: example of plots **********

if analysis0:

    runs = ['Klinik01.set', 'Strecke07.set']
    for run in runs:
        run_to_load = op.join(data_path, run)
        raw = mne.io.read_raw_eeglab(run_to_load)
        dic = dict(mag=1e-12, grad=4e-11, eeg=40e-3, eog=150e-6, ecg=5e-4,
                   emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
                   resp=1, chpi=1e-4, whitened=1e2)
        fig = mne.viz.plot_raw(raw, duration=20, start=raw.times[-1]-20, scalings=dic, show_scrollbars=False, use_opengl=False, overview_mode='channels') #show=False)

# ******* Analysis 1: topomaps in different frequency bands **********

if analysis1:

    # Compute the agency matrix
    adjacency, ch_names = find_ch_adjacency(epochs1A.info, ch_type='eeg')
    mne.viz.plot_ch_adjacency(epochs1A.info, adjacency, ch_names)

    # paired comparison (1A-2A et 1B-2B)
    spatial_clustersA = power_spectrum_stat_comparison_spatial_clusters(psds_band1A, psds_band2A, adjacency, 1)
    spatial_clustersB = power_spectrum_stat_comparison_spatial_clusters(psds_band1B, psds_band2B, adjacency, 1)
    # unpaired comparison (1A-1B et 2A-2B)
    spatial_clusters1 = power_spectrum_stat_comparison_spatial_clusters(psds_band1A, psds_band1B, adjacency, 0)
    spatial_clusters2 = power_spectrum_stat_comparison_spatial_clusters(psds_band2A, psds_band2B, adjacency, 0)

    # spectrum1A.plot_topomap()
    # spectrum2A.plot_topomap()
    # spectrum3 = spectrum1.copy()
    # if spectrum1A.get_data().shape[0] == spectrum2A.get_data().shape[0]:
    #     spectrum3._data = spectrum2A.get_data() - spectrum1A.get_data()
    # else:
    #     spectrum3._data = spectrum2A.get_data().mean(axis=0, keepdims=True) - spectrum1A.get_data().mean(axis=0, keepdims=True)
    # spectrum3.plot_topomap()
    # spectrum3.plot_topomap(normalize=True)
    # spectrum3.plot_topomap(normalize=True, mask=spatial_clusters[0].T, bands={'Delta (0-4 Hz)': (0, 4)})
    # spectrum3.plot_topomap(normalize=True, mask=spatial_clusters[1].T, bands={'Theta (4-8 Hz)': (4, 8)})
    # spectrum3.plot_topomap(normalize=True, mask=spatial_clusters[2].T, bands={'Alpha (8-12 Hz)': (8, 12)})
    # spectrum3.plot_topomap(normalize=True, mask=spatial_clusters[3].T, bands={'Beta (12-30 Hz)': (12, 30)})
    # spectrum3.plot_topomap(normalize=True, mask=spatial_clusters[4].T, bands={'Gamma (30-45 Hz)': (30, 45)})

    # plot_topomap_spectrum_difference(psds_band1A, psds_band2A, psds_band1B, psds_band2B, spatial_clustersA, spatial_clustersB, spatial_clusters1, spatial_clusters2)
    # plot_topomap_spectrum_difference_single_fig(psds_band1A, psds_band2A, psds_band1B, psds_band2B, spatial_clustersA, spatial_clustersB, spatial_clusters1, spatial_clusters2, supertitle)
    plot_topomap_spectrum_difference_reduced_single_fig(psds_band1A, psds_band2A, psds_band1B, psds_band2B, spatial_clustersA, supertitle)


# ******* Analysis 2: spectrum in different electrodes (group of electrodes) **********

if analysis2:

    # channel_choices = ['each_channel']
    channel_choices = ['all_together']
    # channel_choices = ['front_back']
    # channel_choices = ['left_right']
    # channel_choices = ['front_back_left_right']

    psds1Alog, avEp_psds1A, avEp_psds1A_mean, avEp_psds1A_std = averaging_extracting_converting(spectrum1A, psds1A)
    psds2Alog, avEp_psds2A, avEp_psds2A_mean, avEp_psds2A_std = averaging_extracting_converting(spectrum2A, psds2A)
    psds1Blog, avEp_psds1B, avEp_psds1B_mean, avEp_psds1B_std = averaging_extracting_converting(spectrum1B, psds1B)
    psds2Blog, avEp_psds2B, avEp_psds2B_mean, avEp_psds2B_std = averaging_extracting_converting(spectrum2B, psds2B)

    for channel_choice in channel_choices:
        print(channel_choice)

        freq_clustersA = power_spectrum_stat_comparison_freq_clusters(psds1Alog, psds2Alog, 1, channel_choice)
        freq_clustersB = power_spectrum_stat_comparison_freq_clusters(psds1Blog, psds2Blog, 1, channel_choice)
        freq_clusters1 = power_spectrum_stat_comparison_freq_clusters(psds1Alog, psds1Blog, 0, channel_choice)
        freq_clusters2 = power_spectrum_stat_comparison_freq_clusters(psds2Alog, psds2Blog, 0, channel_choice)

        if channel_choice == 'each_channel':
            for chan in np.arange(14):
                plot_figure_power_spectrum_single_plot(freqs1A, freqs2A, freqs1B, freqs2B,
                                                       psds1Alog[:, chan, :].mean(axis=0), psds2Alog[:, chan, :].mean(axis=0), psds1Blog[:, chan, :].mean(axis=0), psds2Blog[:, chan, :].mean(axis=0),
                                                       rawA.ch_names[chan])
                plot_figure_power_spectrum(freqs1A, freqs2A, freqs1B, freqs2B,
                                           psds1Alog[:, chan, :].mean(axis=0), psds2Alog[:, chan, :].mean(axis=0), psds1Blog[:, chan, :].mean(axis=0), psds2Blog[:, chan, :].mean(axis=0),
                                           psds1Alog[:, chan, :].std(axis=0), psds2Alog[:, chan, :].std(axis=0), psds1Blog[:, chan, :].std(axis=0), psds2Blog[:, chan, :].std(axis=0),
                                           freq_clustersA[chan], freq_clustersB[chan], freq_clusters1[chan], freq_clusters2[chan],
                                           rawA.ch_names[chan])


        elif channel_choice == 'all_together':
            plot_figure_power_spectrum_single_plot(freqs1A, freqs2A, freqs1B, freqs2B,
                                                   avEp_psds1A_mean, avEp_psds2A_mean, avEp_psds1B_mean, avEp_psds2B_mean,
                                                   'all_electrodes')
            plot_figure_power_spectrum(freqs1A, freqs2A, freqs1B, freqs2B,
                                       avEp_psds1A_mean, avEp_psds2A_mean, avEp_psds1B_mean, avEp_psds2B_mean,
                                       avEp_psds1A_std, avEp_psds2A_std, avEp_psds1B_std, avEp_psds2B_std,
                                       freq_clustersA, freq_clustersB, freq_clusters1, freq_clusters2,
                                       'figure2_spectrum_all_electrodes_'+supertitle)

        elif channel_choice == 'front_back':
            front_back_chan = [np.arange(0, 7), np.arange(7, 14)]
            front_back_name = ['Frontal electrodes', 'Backward electrodes']
            plot_figure_power_spectrum_single_fig(freqs1A, freqs2A, freqs1B, freqs2B,
                                                  psds1Alog, psds2Alog, psds1Blog, psds2Blog,
                                                  freq_clustersA, freq_clustersB, freq_clusters1, freq_clusters2,
                                                  front_back_chan, front_back_name, channel_choice, supertitle)

        elif channel_choice == 'left_right':
            front_back_chan = [[1, 3, 5, 7, 10, 12], [2, 4, 6, 8, 11, 13]]
            front_back_name = ['Left electrodes', 'Right electrodes']
            plot_figure_power_spectrum_single_fig(freqs1A, freqs2A, freqs1B, freqs2B,
                                                  psds1Alog, psds2Alog, psds1Blog, psds2Blog,
                                                  freq_clustersA, freq_clustersB, freq_clusters1, freq_clusters2,
                                                  front_back_chan, front_back_name, channel_choice, supertitle)
        elif channel_choice == 'front_back_left_right':
            front_back_chan = [[1, 3, 5], [2, 4, 6], [7, 10, 12], [8, 11, 13]]
            front_back_name = ['Front Left electrodes', 'Front Right electrodes', 'Back Left electrodes', 'Back Right electrodes']
            plot_figure_power_spectrum_single_fig(freqs1A, freqs2A, freqs1B, freqs2B,
                                                  psds1Alog, psds2Alog, psds1Blog, psds2Blog,
                                                  freq_clustersA, freq_clustersB, freq_clusters1, freq_clusters2,
                                                  front_back_chan, front_back_name, channel_choice, supertitle)

# # ******* Analysis 3: clustering in 2D **********

if analysis3:

    spatio_temporal_cluster_test_and_plot(psds1A, psds2A, freqs1A, epochs1A, 'with TD, start', 'with TD, end', 0.05)
    spatio_temporal_cluster_test_and_plot(psds1B, psds2B, freqs1A, epochs1A, 'without TD, start', 'without TD, end', 0.05)
    spatio_temporal_cluster_test_and_plot(psds1A, psds1B, freqs1A, epochs1A, 'with TD, start', 'without TD, start', 0.05)
    spatio_temporal_cluster_test_and_plot(psds2A, psds2B, freqs1A, epochs1A, 'with TD, end', 'without TD, end', 0.05)


# # ******* Analysis 4: right/left statistical comparison **********

if analysis4:

    print('right/left stat comparison for front_back_start_end')

    psds1Alog, avEp_psds1A, avEp_psds1A_mean, avEp_psds1A_std = averaging_extracting_converting(spectrum1A, psds1A)
    psds2Alog, avEp_psds2A, avEp_psds2A_mean, avEp_psds2A_std = averaging_extracting_converting(spectrum2A, psds2A)
    psds1Blog, avEp_psds1B, avEp_psds1B_mean, avEp_psds1B_std = averaging_extracting_converting(spectrum1B, psds1B)
    psds2Blog, avEp_psds2B, avEp_psds2B_mean, avEp_psds2B_std = averaging_extracting_converting(spectrum2B, psds2B)

    front_back_chan = [[1, 3, 5], [2, 4, 6], [7, 10, 12], [8, 11, 13]]
    front_back_name = ['Front Left electrodes', 'Front Right electrodes', 'Back Left electrodes', 'Back Right electrodes']

    freq_clusters_1A, freq_clusters_2A, freq_clusters_1B, freq_clusters_2B = power_spectrum_right_left_stat_comparison_freq_clusters(psds1Alog, psds2Alog, psds1Blog, psds2Blog)

    plot_figure_power_spectrum_right_left_single_fig(freqs1A, freqs2A, freqs1B, freqs2B,
                                                     psds1Alog, psds2Alog, psds1Blog, psds2Blog,
                                                     freq_clusters_1A, freq_clusters_2A, freq_clusters_1B, freq_clusters_2B,
                                                     front_back_chan, front_back_name, supertitle)
