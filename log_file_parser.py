import matplotlib.pyplot as plt
import numpy as np
import os
import re


def get_values_from_string(string, key_phrase):
    non_decimal = re.compile(r'[^\d.]+')

    idx = string.find(key_phrase)
    if idx > 0:
        idx += len(key_phrase)-1
    else:
        return None
    val = string[idx:].split()[1].strip()
    if len(val) > 1:
        return float(non_decimal.sub('', val))
    else:
        None


class LogFile:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.user_param = {}
        self.discriminator_error = []
        self.classifier_accuracy = []
        if file_path:
            self.parse_file(file_path)

    def parse_file(self, file_path):
        # Reset values
        self.discriminator_error = []
        self.classifier_accuracy = []

        with open(file=file_path, mode='r') as of:
            for line in of.readlines():
                if line[0] == '-' and line.find(':') > 0:
                    # Assume it is a user given parameters
                    key, value = line[1:].split(':')
                    self.user_param[key.strip()] = value.strip()
                if line.startswith('Fixed Input:'):
                    # Test data set line
                    # Look for discriminator error
                    search_phrase = 'discriminator error'
                    value = get_values_from_string(line, search_phrase)
                    self.discriminator_error.append(value)
                    # Look for classifier accuracy
                    search_phrase = 'classifier accuracy'
                    value = get_values_from_string(line, search_phrase)
                    self.classifier_accuracy.append(value)


def finish_figure(figure, title, num_vals=None, thresholds=[]) -> None:
    plt.figure(figure)
    plt.title(title)
    for t, s in zip(thresholds, [':', '--', '-']):
        plt.plot([0, num_vals], [t, t], s+'k', zorder=0, label=f'{t:d}%')
    plt.ylim([-1, 101])
    plt.legend()


def main(file_list):

    log_files = []
    for file in file_list:
        if os.path.isdir(file):
            file = os.path.join(file, 'log.txt')
        if not os.path.exists(file):
            continue
        # File exists, create LogFile object and add it to the list
        log_files.append(LogFile(file))

    num_vals = 0
    fig_disc = plt.figure()
    fig_class = plt.figure()
    for log_file in log_files:
        plt.figure(fig_disc)
        plt.plot(log_file.discriminator_error, label=log_file.file_path)
        plt.figure(fig_class)
        plt.plot(log_file.classifier_accuracy, label=log_file.file_path)
        num_vals = max(num_vals, len(log_file.discriminator_error), len(log_file.classifier_accuracy))

    num_vals -= 1

    finish_figure(fig_disc, "Discriminator Error (ideally at 50%)", num_vals, [45, 50])
    finish_figure(fig_class, "Classifier Accuracy (ideally at 100%)", num_vals, [98, 90])

    plt.show()


if __name__ == '__main__':
    res_dir = 'results'
    '''
    all_results = [os.path.join(res_dir, item) for item in sorted(os.listdir(res_dir))
                   if os.path.isdir(os.path.join(res_dir, item))]
    print('\n'.join(all_results))
    main(all_results)
    '''
    '''
    main(['results/diff_lr_deca=01_alpha=0_vec=20',            # <- Good result
          'results/glr=1e-3_dlr=2e-3_nodecay_vec=20',
          'results/glr=2e-3_dlr=2e-3_dalpha=0_vec=20',
          'results/glr=3e-3_dlr=2e-3_dalpha=0_vec=20',
          'results/glr=4e-3_dlr=2e-3_dalpha=0_vec=20',
          'results/glr=4e-3_dlr=3e-3_dalpha=0_vec=20',
          'results/glr=4e-4_dlr=2e-4_dalpha=0_vec=20',
          'results/glr=4e-4_dlr=2e-4_nodecay_vec=20',
          'results/glr=4e-5_dlr=2e-5_nodecay_vec=20',
          'results/lr=2e-3_deca=01_alpha=0_vec=20',
          'results/noclassifier',
          'out/log.txt'])
    '''
    '''
    main(['results/diff_lr_deca=01_alpha=0_vec=20',             # <- TOP1
          'results/glr=4e-3_dlr=2e-3_dalpha=0_vec=20',          # <- TOP2
          'results/glr=4e-4_dlr=2e-4_dalpha=0_vec=20',
          'results/glr=1e-3_dlr=2e-3_nodecay_vec=20',
          'out/log.txt'])
    '''
    # Different learning rate scale
    main(['results/glr=4e-3_dlr=2e-3_dalpha=0_vec=20',          # <- TOP2
          #'results/glr=1e-3_dlr=2e-3_nodecay_vec=20',
          #'results/glr=4e-4_dlr=2e-4_nodecay_vec=20',
          'results/glr=2e-3_dlr=2e-3_update_disc_every_other_batch',
          'results/glr=2e-4_dlr=2e-4_update_disc_every_other_batch',
          'results/glr=2e-5_dlr=2e-5_update_disc_every_other_batch',
          'results/update_disc_every_other_batch',
          'results/glr=2e-3_dlr=2e-3_dalpha=0_vec=20',
          #'results/glr=1e-3_dlr=1e-4_nodecay_vec=20',
          'out/log.txt'])

