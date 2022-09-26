import tensorflow as tf
import numpy as np

def add_histogram(writer, tag, values, step):
    """
    Logs the histogram of a list/vector of values.
    From: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    """
    # counts, bin_edges = np.histogram(values, bins=bins)
    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Therefore we drop the start of the first bin
    # bin_edges = bin_edges[1:]

    bins = np.arange(0, len(values)) + 1
    bins = np.linspace(1, len(values), len(values) + 1, endpoint=True)

    # Fill fields of histogram proto
    hist = tf.compat.v1.HistogramProto()
    hist.min = float(np.min(bins))
    hist.max = float(np.max(bins))
    # hist.num = int(np.prod(values.shape))
    # hist.sum = float(np.sum(values))
    # hist.sum_squares = float(np.sum(values ** 2))

    bins = bins[1:]

    for edge in bins:
        hist.bucket_limit.append(edge)
    for c in values:
        d = c * 30.0 / float(len(values))
        hist.bucket.append(d)

    summary = tf.compat.v1.summary.Summary(value=[tf.compat.v1.summary.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)
    writer.flush()