import six


def str_result(result):
    return '\t'.join(['%s=%s' % kv for kv in six.iteritems(result)])
