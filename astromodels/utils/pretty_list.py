from builtins import str
__author__ = 'giacomov'

import yaml
import re


def _process_html(dictionary):

    list_start = '<ul>\n'
    list_stop = '</ul>\n'
    entry_start = '<li>'
    entry_stop = '</li>\n'

    output=[list_start]

    for key,value in list(dictionary.items()):

        if isinstance(value, dict):

            # Check whether the dictionary is empty. In that case, don't print anything
            if len(value)==0:

                continue

            if len(value) > 1 or isinstance(list(value.values())[0], dict):

                output.extend(
                    (
                        entry_start + str(key) + ': ',
                        _process_html(value),
                        entry_stop,
                    )
                )

            else:

                output.append(entry_start + str(key) + ': ' + str(list(value.values())[0]) + entry_stop)

        else:

            output.append(entry_start + str(key) + ': ' + str(value) + entry_stop)

    output.append(list_stop)

    return '\n'.join(output)


def _process_text(dictionary):

    # Obtain YAML representation

    string_repr = yaml.dump(dictionary, default_flow_style=False)

    return re.sub("(\s*)(.+)", "\\1  * \\2", string_repr)


def dict_to_list(dictionary, html=False):
    """
    Convert a dictionary into a unordered list.

    :param dictionary: a dictionary
    :param html: whether to output HTML or simple text (True or False)
    :return: the list
    """

    return _process_html(dictionary) if html else _process_text(dictionary)
