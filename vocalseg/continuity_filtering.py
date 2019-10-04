def continuity_segmentation(
    vocalization,
    rate,
    min_level_db=-80,
    min_level_db_floor=-40,
    db_delta=5,
    n_fft=1024,
    hop_length_ms=1,
    win_length_ms=5,
    ref_level_db=20,
    pre=0.97,
    spectral_range=None,
    verbose=False,
    mask_thresh_std=1,
    neighborhood_time_ms=5,
    neighborhood_freq_hz=500,
    neighborhood_thresh=0.5,
    max_element_len_ms=2000,
    min_syllable_length_s=0.1,
    min_silence_for_spec=0.1,
    silence_threshold=0.05,
    max_vocal_for_spec=1.0,
    temporal_neighbor_merge_distance_ms=0.0,
    overlapping_element_merge_thresh=np.inf,
    min_element_size_ms_hz=[0, 0],  # ms, hz
    figsize=(20, 5),
):
    """
    [summary]
    
    Arguments:
        vocalization {[type]} -- [description]
        rate {[type]} -- [description]
    
    Keyword Arguments:
        min_level_db {int} -- [description] (default: {-80})
        min_level_db_floor {int} -- [description] (default: {-40})
        db_delta {int} -- [description] (default: {5})
        n_fft {int} -- [description] (default: {1024})
        hop_length_ms {int} -- [description] (default: {1})
        win_length_ms {int} -- [description] (default: {5})
        ref_level_db {int} -- [description] (default: {20})
        pre {float} -- [description] (default: {0.97})
        spectral_range {[type]} -- [description] (default: {None})
        verbose {bool} -- [description] (default: {False})
        mask_thresh_std {int} -- [description] (default: {1})
        neighborhood_time_ms {int} -- [description] (default: {5})
        neighborhood_freq_hz {int} -- [description] (default: {500})
        neighborhood_thresh {float} -- [description] (default: {0.5})
        max_element_len_ms {int} -- [description] (default: {2000})
        min_syllable_length_s {float} -- [description] (default: {0.1})
        min_silence_for_spec {float} -- [description] (default: {0.1})
        silence_threshold {float} -- [description] (default: {0.05})
        max_vocal_for_spec {float} -- [description] (default: {1.0})
        temporal_neighbor_merge_distance_ms {float} -- [description] (default: {0.0})
        overlapping_element_merge_thresh {[type]} -- [description] (default: {np.inf})
        min_element_size_ms_hz {list} -- [description] (default: {[0, 0]})
        hzfigsize {tuple} -- [description] (default: {(20, 5)})
    
    Returns:
        [type] -- [description]
    """

    def plot_interim(spec, cmap=plt.cm.afmhot, zero_nan=False):
        """
        quickplot of a spectrogram
        """
        fig, ax = plt.subplots(figsize=figsize)
        if zero_nan:
            spec = spec.copy()
            spec[spec == 0] = np.nan
        plot_spec(
            spec,
            fig=fig,
            ax=ax,
            rate=rate,
            hop_len_ms=hop_length_ms,
            show_cbar=False,
            cmap=cmap,
        )
        plt.show()

    results = dynamic_threshold_segmentation(
        vocalization,
        rate,
        n_fft=n_fft,
        hop_length_ms=hop_length_ms,
        win_length_ms=win_length_ms,
        ref_level_db=ref_level_db,
        pre=pre,
        min_level_db=min_level_db,
        db_delta=db_delta,
        silence_threshold=silence_threshold,
        verbose=False,
        spectral_range=spectral_range,
        min_syllable_length_s=min_syllable_length_s,
        min_silence_for_spec=min_silence_for_spec,
        max_vocal_for_spec=max_vocal_for_spec,
    )

    spec = results["spec"]

    # bin width in Hz
    if spectral_range is None:
        spec_bin_hz = (rate / 2) / np.shape(spec)[0]
    else:
        spec_bin_hz = (spectral_range[1] - spectral_range[0]) / np.shape(spec)[0]

    if verbose:
        plot_interim(spec, cmap=plt.cm.Greys)

    ### create a mask
    mask = mask_spectrogram(spec, mask_thresh_std)

    if verbose:
        plot_interim(mask)

    # Create a smoothing filter for the mask in time and frequency
    continuity_filter = make_continuity_filter(
        neighborhood_freq_hz, neighborhood_time_ms, spec_bin_hz, hop_length_ms
    )
    print(np.shape(continuity_filter))
    ### remove non-continuous regions of the mask
    # apply filter
    mask = signal.fftconvolve(
        (1 - mask.astype("float32")), continuity_filter, mode="same"
    )
    # threshold filter
    mask = mask < neighborhood_thresh

    if verbose:
        plot_interim(mask)

    # find continous elements
    elements = segment_mask(mask)

    if verbose:
        plot_interim(elements, cmap=cmap, zero_nan=True)

    # get element timing
    unique_elements, syllable_start_times, syllable_end_times = get_syllable_timing(
        elements, hop_length_ms
    )
    print("unique elements: {}".format(len(unique_elements)))
    # merge elements that are nearby to each other
    if temporal_neighbor_merge_distance_ms > 0:
        elements = merge_temporal_neighbors(
            elements,
            unique_elements,
            syllable_start_times,
            syllable_end_times,
            temporal_neighbor_merge_distance_ms,
        )

        if verbose:
            plot_interim(elements, cmap=cmap, zero_nan=True)
            unique_elements = np.unique(elements[elements != 0].astype(int))
            print("unique elements: {}".format(len(unique_elements)))

    # no reason to merge overlapping if already merging neighbords
    elif overlapping_element_merge_thresh <= 1.0:
        # merge elements that are overlapping in time by some amount
        elements = merge_overlapping_elements(
            elements,
            unique_elements,
            syllable_start_times,
            syllable_end_times,
            overlapping_element_merge_thresh,
        )
        if verbose:
            plot_interim(elements, cmap=cmap, zero_nan=True)
            unique_elements = np.unique(elements[elements != 0].astype(int))
            print("unique elements: {}".format(len(unique_elements)))

    # remove elements that are
    if np.product(min_element_size_ms_hz) > 0:
        min_element_size = int(
            np.product(
                (
                    min_element_size_ms_hz[0] / hop_length_ms,
                    min_element_size_ms_hz[1] / spec_bin_hz,
                )
            )
        )
        if min_element_size > 0:
            elements = remove_small_elements(elements, min_element_size)

            if verbose:
                plot_interim(elements, cmap=cmap, zero_nan=True)
                unique_elements = np.unique(elements[elements != 0].astype(int))
                print("unique elements: {}".format(len(unique_elements)))

    results["elements"] = elements

    return results


def remove_small_elements(elements, min_element_size):
    """
    remove elements that are below some threshold size
    
    Arguments:
        elements {[type]} -- [description]
        min_element_size {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # get unique points
    unique_elements = np.unique(elements[elements != 0].astype(int))

    print(min_element_size)
    for element in unique_elements:
        # if the size of the cluster is smaller than the minimum, remove it
        if np.sum(elements == element) < min_element_size:
            elements[elements == element] = 0

    return elements


def merge_temporal_neighbors(
    elements,
    unique_elements,
    syllable_start_times,
    syllable_end_times,
    temporal_neighbor_merge_distance_ms,
):
    """
    merge elements that are within temporal_neighbor_merge_distance_ms
     ms of each other
    
    Arguments:
        elements {[type]} -- [description]
        unique_elements {[type]} -- [description]
        syllable_start_times {[type]} -- [description]
        syllable_end_times {[type]} -- [description]
        temporal_neighbor_merge_distance_ms {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    merged_elements = {}
    for element, st, et in tqdm(
        zip(unique_elements, syllable_start_times, syllable_end_times),
        total=len(unique_elements),
        desc="merging temporal neighbors",
        leave=False,
    ):
        # if this element has already been merged, ignore it
        if element in merged_elements.keys():
            element = merged_elements[element]
        # get elements that start between the beginning of this element and the
        #    end of this element plus temporal_neighbor_merge_distance_ms
        overlapping_syllables = np.where(
            (syllable_start_times > st)
            & (syllable_start_times < et + (temporal_neighbor_merge_distance_ms))
        )[0]
        # print(overlapping_syllables)
        if len(overlapping_syllables) > 0:
            for overlapping_syllable in overlapping_syllables:
                syll_name = unique_elements[overlapping_syllable]
                merged_elements[syll_name] = element
                elements[elements == syll_name] = element
            # remove from lists
            unique_elements = np.delete(unique_elements, overlapping_syllables)
            syllable_start_times = np.delete(
                syllable_start_times, overlapping_syllables
            )
            syllable_end_times = np.delete(syllable_end_times, overlapping_syllables)

    return elements
