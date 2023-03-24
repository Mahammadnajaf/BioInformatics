import numpy
import scipy.signal

from bob.bio.base.pipelines import BioAlgorithm


class MiuraMatch(BioAlgorithm):
    """Finger vein matching: match ratio via cross-correlation

    The method is based on "cross-correlation" between a model and a probe image.
    It convolves the binary image(s) representing the model with the binary image
    representing the probe (rotated by 180 degrees), and evaluates how they
    cross-correlate. If the model and probe are very similar, the output of the
    correlation corresponds to a single scalar and approaches a maximum. The
    value is then normalized by the sum of the pixels lit in both binary images.
    Therefore, the output of this method is a floating-point number in the range
    :math:`[0, 0.5]`. The higher, the better match.

    In case model and probe represent images from the same vein structure, but
    are misaligned, the output is not guaranteed to be accurate. To mitigate this
    aspect, Miura et al. proposed to add a *small* cropping factor to the model
    image, assuming not much information is available on the borders (``ch``, for
    the vertical direction and ``cw``, for the horizontal direction). This allows
    the convolution to yield searches for different areas in the probe image. The
    maximum value is then taken from the resulting operation. The convolution
    result is normalized by the pixels lit in both the cropped model image and
    the matching pixels on the probe that yield the maximum on the resulting
    convolution.

    For this to work properly, input images are supposed to be binary in nature,
    with zeros and ones.

    Based on [MNM04]_ and [MNM05]_

    Parameters:

      ch (:py:class:`int`, optional): Maximum search displacement in y-direction.

      cw (:py:class:`int`, optional): Maximum search displacement in x-direction.

    """

    def __init__(
        self,
        ch=80,  # Maximum search displacement in y-direction
        cw=90,  # Maximum search displacement in x-direction
        probes_score_fusion="max",
        enrolls_score_fusion="mean",
        **kwargs,
    ):
        super().__init__(
            probes_score_fusion=probes_score_fusion,
            enrolls_score_fusion=enrolls_score_fusion,
            **kwargs,
        )

        self.ch = ch
        self.cw = cw

[docs]    def create_templates(self, feature_sets, enroll):
        return feature_sets


[docs]    def compare(self, enroll_templates, probe_templates):
        # returns scores NxM where N is the number of enroll templates and M is the number of probe templates
        # enroll_templates is Nx?1xD
        # probe_templates is Mx?2xD
        scores = []
        for enroll in enroll_templates:
            scores.append([])
            for probe in probe_templates:
                s = [[self.score(e, p) for p in probe] for e in enroll]
                s = self.fuse_probe_scores(s, axis=1)
                s = self.fuse_enroll_scores(s, axis=0)
                scores[-1].append(s)
        return numpy.array(scores)


[docs]    def score(self, model, probe):
        """Computes the score between the probe and the model.

        Parameters:

          model (numpy.ndarray): The model of the user to test the probe against

          probe (numpy.ndarray): The probe to test


        Returns:

          list[float]: Value between 0 and 0.5, larger value means a better match

        """

        image_ = probe.astype(numpy.float64)

        md = model
        # erode model by (ch, cw)
        R = md.astype(numpy.float64)
        h, w = R.shape  # same as I
        crop_R = R[self.ch : h - self.ch, self.cw : w - self.cw]

        # correlates using scipy - fastest option available iff the self.ch and
        # self.cw are height (>30). In this case, the number of components
        # returned by the convolution is high and using an FFT-based method
        # yields best results. Otherwise, you may try  the other options bellow
        # -> check our test_correlation() method on the test units for more
        # details and benchmarks.
        Nm = scipy.signal.fftconvolve(image_, numpy.rot90(crop_R, k=2), "valid")
        # 2nd best: use convolve2d or correlate2d directly;
        # Nm = scipy.signal.convolve2d(I, numpy.rot90(crop_R, k=2), 'valid')
        # 3rd best: use correlate2d
        # Nm = scipy.signal.correlate2d(I, crop_R, 'valid')

        # figures out where the maximum is on the resulting matrix
        t0, s0 = numpy.unravel_index(Nm.argmax(), Nm.shape)

        # this is our output
        Nmm = Nm[t0, s0]

        # normalizes the output by the number of pixels lit on the input
        # matrices, taking into consideration the surface that produced the
        # result (i.e., the eroded model and part of the probe)
        score = Nmm / (
            crop_R.sum()
            + image_[t0 : t0 + h - 2 * self.ch, s0 : s0 + w - 2 * self.cw].sum()
        )

        return score
