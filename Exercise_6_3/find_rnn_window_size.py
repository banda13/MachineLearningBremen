import csv
from Exercise_6_3.rnn import RNN

# starting and ending window size
start_size = 2
end_size = 100

# how many times to try each window size
repetition = 10
file_name = 'experiment.csv'

# NOTE: this calculation are very computational power intensive
if __name__ == '__main__':
    # create the csv and write header
    with open(file_name, 'w') as csvfile:
        fieldnames = ['size', 'train_loss', 'test_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';', lineterminator = '\n')
        writer.writeheader()

    # iterate trough the window sizes
    for x in range(start_size, end_size):
        try:
            print('Experiment for window size: ', x)
            avg_train_loss, avg_test_loss = 0, 0

            # try every window size more, than take the average loss
            for i in range(repetition):
                rnn = RNN(sliding_window_size=x)
                rnn.load_and_preprocess()
                rnn.compile_model()
                rnn.train(epochs=10, batch_size=32, verbose=2) # verbose is 2 to dont print training process
                train_loss, test_loss = rnn.evaluate()
                avg_train_loss += train_loss
                avg_test_loss += test_loss
            avg_test_loss /= repetition
            avg_train_loss /= repetition

            # save the results into the csv
            with open(file_name, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';', lineterminator = '\n')
                writer.writerow({fieldnames[0]: str(x), fieldnames[1]: str(avg_train_loss), fieldnames[2]: str(avg_test_loss)})

        # if some error occur, do not shout down the whole experiment
        except Exception as e:
            print('Failed to make experiment for window size %d' % x)