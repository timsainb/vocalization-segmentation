from tqdm.autonotebook import tqdm
from vocalseg.utils import _normalize, spectrogram_nn, norm
import numpy as np
from scipy import ndimage


def contiguous_regions(condition):
    """
    Compute contiguous region of binary value (e.g. silence in waveform) to 
        ensure noise levels are sufficiently low
    
    Arguments:
        condition {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    idx = []
    i = 0
    while i < len(condition):
        x1 = i + condition[i:].argmax()
        try:
            x2 = x1 + condition[x1:].argmin()
        except:
            x2 = x1 + 1
        if x1 == x2:
            if condition[x1] == True:
                x2 = len(condition)
            else:
                break
        idx.append([x1, x2])
        i = x2
    return idx


def dynamic_threshold_segmentation(
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
    silence_threshold=0.05,
    min_silence_for_spec=0.1,
    max_vocal_for_spec=1.0,
    min_syllable_length_s=0.1,
    spectral_range=None,
    verbose=False,
):
    """
    computes a spectrogram from a waveform by iterating through thresholds
         to ensure a consistent noise level
    
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
        silence_threshold {float} -- [description] (default: {0.05})
        min_silence_for_spec {float} -- [description] (default: {0.1})
        max_vocal_for_spec {float} -- [description] (default: {1.0})
        min_syllable_length_s {float} -- [description] (default: {0.1})
        spectral_range {[type]} -- [description] (default: {None})
        verbose {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """

    # does the envelope meet the standards necessary to consider this a bout
    envelope_is_good = False

    # make a copy of the hyperparameters

    # make a copy of the original spectrogram
    spec_orig = spectrogram_nn(
        vocalization,
        rate,
        n_fft=n_fft,
        hop_length_ms=hop_length_ms,
        win_length_ms=win_length_ms,
        ref_level_db=ref_level_db,
        pre=pre,
    )
    fft_rate = 1000 / hop_length_ms

    if spectral_range is not None:
        spec_bin_hz = (rate / 2) / np.shape(spec_orig)[0]
        spec_orig = spec_orig[
            int(spectral_range[0] / spec_bin_hz) : int(spectral_range[1] / spec_bin_hz),
            :,
        ]

    # loop through possible thresholding configurations starting at the highest
    for _, mldb in enumerate(
        tqdm(
            np.arange(min_level_db, min_level_db_floor, db_delta),
            leave=False,
            disable=(not verbose),
        )
    ):
        # set the minimum dB threshold
        min_level_db = mldb
        # normalize the spectrogram
        spec = norm(_normalize(spec_orig, min_level_db=min_level_db))

        # subtract the median
        spec = spec - np.median(spec, axis=1).reshape((len(spec), 1))
        spec[spec < 0] = 0

        # get the vocal envelope
        vocal_envelope = np.max(spec, axis=0) * np.sqrt(np.mean(spec, axis=0))
        # normalize envelope
        vocal_envelope = vocal_envelope / np.max(vocal_envelope)

        # Look at how much silence exists in the signal
        onsets, offsets = onsets_offsets(vocal_envelope > silence_threshold) / fft_rate
        onsets_sil, offsets_sil = (
            onsets_offsets(vocal_envelope <= silence_threshold) / fft_rate
        )

        # if there is a silence of at least min_silence_for_spec length,
        #  and a vocalization of no greater than max_vocal_for_spec length, the env is good
        if len(onsets_sil) > 0:
            # frames per second of spectrogram

            # longest silences and periods of vocalization
            max_silence_len = np.max(offsets_sil - onsets_sil)
            max_vocalization_len = np.max(offsets - onsets)
            if verbose:
                print("longest silence", max_silence_len)
                print("longest vocalization", max_vocalization_len)

            if max_silence_len > min_silence_for_spec:
                if max_vocalization_len < max_vocal_for_spec:
                    envelope_is_good = True
                    break
        if verbose:
            print("Current min_level_db: {}".format(min_level_db))

    if not envelope_is_good:
        return None

    onsets, offsets = onsets_offsets(vocal_envelope > silence_threshold) / fft_rate

    # threshold out short syllables
    length_mask = (offsets - onsets) >= min_syllable_length_s

    return {
        "spec": spec,
        "vocal_envelope": vocal_envelope.astype("float32"),
        "min_level_db": min_level_db,
        "onsets": onsets[length_mask],
        "offsets": offsets[length_mask],
    }


def onsets_offsets(signal):
    """
    [summary]
    
    Arguments:
        signal {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    elements, nelements = ndimage.label(signal)
    if nelements == 0:
        return np.array([[0], [0]])
    onsets, offsets = np.array(
        [
            np.where(elements == element)[0][np.array([0, -1])] + np.array([0, 1])
            for element in np.unique(elements)
            if element != 0
        ]
    ).T
    return np.array([onsets, offsets])
