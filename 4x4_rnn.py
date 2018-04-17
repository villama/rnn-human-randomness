"""4x4_rnn.py."""

from config import get_config, print_usage
import sqlite3
import numpy as np
from tqdm import trange
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from pathlib import Path
import os
import matplotlib.pyplot as plt
from scipy import stats


def main():
    """Main method."""
    # ----------------------------------------
    # Load data from database
    if (config.input_dtype == 'db'):
        times_taken = get_times_taken()
        features, labels = get_moves_db()
    else:
        times_taken = None
        features, labels = get_moves_csv()
    features, labels = preprocess(features, labels)

    # Truncate number of rows if we want to look at less data
    if config.n is not None:
        if config.input_dtype == 'db':
            times_taken = times_taken[:config.n]
        features = features[:config.n]
        labels = labels[:config.n]

    # Split train from test
    fea_trte = np.split(features, [features.shape[1] - config.test_size], 1)
    lab_trte = np.split(labels, [labels.shape[1] - config.test_size], 1)
    fea_tr, lab_tr = fea_trte[0], lab_trte[0]
    fea_te = fea_trte[1]

    print("\nTraining and testing models...")
    predictions = np.zeros((features.shape[0], config.test_size, 4))
    for i in trange(len(fea_tr)):
        # Get a trained model for each i
        predictions[i] = predict(fea_tr[i], lab_tr[i], fea_te[i], i)

    # evaluate(features, labels, predictions, times_taken)
    evaluate_csv(features, labels, predictions, times_taken)

    # Save predictions inside log
    s = ""
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            for k in range(predictions.shape[2]):
                s += "{},".format(predictions[i][j][k])
        s += "\n"
    f = open("logs/predictions.csv", 'w')
    f.write(s)
    f.close()


def get_times_taken():
    # Connect to the db file
    conn = sqlite3.connect(str(config.input))
    c = conn.cursor()
    times = []
    query = 'SELECT timeTaken FROM UserData WHERE block1 IS NOT NULL'
    for row in c.execute(query):
        times += [int(row[0])]

    times = np.asarray(times, dtype=int)
    times = times / 1000 / config.num_moves

    # Return times (139,) as avg seconds per move
    return times


def get_moves_db():
    # Connect to the db file
    conn = sqlite3.connect(str(config.input))
    c = conn.cursor()
    q = 'SELECT block1, block2, block3 FROM UserData WHERE block1 IS NOT NULL'
    d = {'u': 0, 'd': 1, 'l': 2, 'r': 3}
    data = []
    for row in c.execute(q):
        s = str(row[0]) + str(row[1]) + str(row[2])
        s = s.replace(',', '').strip()
        data += [int(x) if x.isdigit() else d[x] for x in s]

    data = np.asarray(data, dtype=int).reshape((-1, config.num_moves, 3))

    # Separate features from labels
    features = np.delete(data, [2], 2)
    labels = np.delete(data, [0, 1], 2).reshape((-1, config.num_moves))

    # Return features (139, 1200, 2) and labels (139, 1200) as ndarrays
    return features, labels


def get_moves_csv():
    s = open(config.input, 'r').read()
    s = s.replace(',', '').strip()
    s = s.split('\n')
    d = {'u': 0, 'd': 1, 'l': 2, 'r': 3}
    data = []
    for row in s:
        data += [int(x) if x.isdigit() else d[x] for x in row]

    data = np.asarray(data, dtype=int).reshape((-1, config.num_moves, 3))

    # Separate features from labels
    features = np.delete(data, [2], 2)
    labels = np.delete(data, [0, 1], 2).reshape((-1, config.num_moves))

    # Return features (139, 1200, 2) and labels (139, 1200) as ndarrays
    return features, labels


def preprocess(features, labels):
    # [0, 1, 2, 3] -> [0, 0.33, 0.66, 0.99]
    features = features / 3

    # Reshape features to be LSTM friendly (N, 1200, 1, 2)
    features = features.reshape(-1, features.shape[1], 1, 2)

    # Convert labels to one-hot
    labels = (np.arange(4) == labels[..., None]).astype(int)

    # Return features normalized and reshaped (N, 1200, 1, 2),
    # labels one-hot (N, 1200, 4)
    return features, labels


def train(features, labels, i):
    # features shape: (1200, 1, 2)
    # labels shape: (1200, 4)

    # Design model
    model = Sequential()
    if (config.num_hidden == 1):
        model.add(LSTM(config.num_neurons,
                       batch_input_shape=(config.batch_size, 1, 2),
                       stateful=True))
    else:
        model.add(LSTM(config.num_neurons,
                       batch_input_shape=(config.batch_size, 1, 2),
                       stateful=True,
                       return_sequences=True))
        for i in range(config.num_hidden - 1):
            if i == config.num_hidden - 2:
                model.add(LSTM(config.num_neurons,
                               stateful=True))
            else:
                model.add(LSTM(config.num_neurons,
                               stateful=True,
                               return_sequences=True))
    model.add(Dense(4))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Checkpoint path
    checkpoint_path = config.log_dir + "/model_{}.hdf5".format(i)

    # Check if we can load a saved weight
    if Path(checkpoint_path).is_file():
        model.load_weights(checkpoint_path)
    else:
        # Create the logs directory if it does not exist
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # Checkpoint
        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=False)

        # Fit network
        for i in range(config.num_epoch):
            model.fit(features, labels, epochs=1,
                      batch_size=config.batch_size, verbose=0,
                      shuffle=False, callbacks=[checkpoint])
            model.reset_states()

    # Redefine network for batch_size = 1
    model_out = Sequential()
    if (config.num_hidden == 1):
        model_out.add(LSTM(config.num_neurons,
                           batch_input_shape=(1, 1, 2),
                           stateful=True))
    else:
        model_out.add(LSTM(config.num_neurons,
                           batch_input_shape=(config.batch_size, 1, 2),
                           stateful=True,
                           return_sequences=True))
        for i in range(config.num_hidden - 1):
            if i == config.num_hidden - 2:
                model_out.add(LSTM(config.num_neurons,
                                   stateful=True))
            else:
                model_out.add(LSTM(config.num_neurons,
                                   stateful=True,
                                   return_sequences=True))
    model_out.add(Dense(4))

    # Copy weights
    old_weights = model.get_weights()
    model_out.set_weights(old_weights)

    model_out.compile(loss='mean_squared_error', optimizer='adam')

    return model_out


def predict(fea_tr, lab_tr, fea_te, i):
    model = train(fea_tr, lab_tr, i)
    predictions = np.zeros((config.test_size, 4))
    for i in range(len(fea_te)):
        predictions[i] = model.predict(fea_te[i].reshape((1, 1, 2)),
                                       batch_size=1)
    del model

    # Return an individual's one-hot predictions (400, 4)
    return predictions


def evaluate(features, labels, te_predictions, times_taken):
    #    features.shape : (N, 1200, 1, 2)
    #      labels.shape : (N, 1200, 4) -> (N, 1200)
    # predictions.shape : (N, 1200, 4) -> (N, 1200)

    # Restore features to ints
    features = np.round(features * 3).astype(int)
    # Argmax on labels and predictions
    labels = np.argmax(labels, axis=2)
    te_predictions = np.argmax(te_predictions, axis=2)
    # Get test features/labels
    te_features = np.split(
        features, [features.shape[1] - config.test_size], 1)[1]
    te_labels = np.split(labels, [labels.shape[1] - config.test_size], 1)[1]
    # Get num participants, num moves
    N = features.shape[0]

    # ----------------------------------------
    # 1. Total accuracy
    total_acc = np.zeros(N)
    for i in range(N):
        total_acc[i] = np.sum(te_labels[i] == te_predictions[i])
    total_acc = total_acc / te_labels.shape[1]

    print("\nTotal accuracies for {} participants:".format(N))
    for i in range(N):
        print("    {:.4f}".format(total_acc[i]))
    print("\nMean total accuracies for {} participants:".format(N))
    print("    {:.4f}({:.4f})".format(total_acc.mean(), total_acc.std()))

    # ----------------------------------------
    # 2. 1st half vs 2nd half accuracy difference
    h_size = int(te_labels.shape[1] / 2)
    num_corr_1 = np.zeros(N)
    num_corr_2 = np.zeros(N)
    for i in range(N):
        for j in range(h_size):
            if te_labels[i][j] == te_predictions[i][j]:
                num_corr_1[i] += 1
        for j in range(te_labels.shape[1] - h_size):
            if te_labels[i][j + h_size] == te_predictions[i][j + h_size]:
                num_corr_2[i] += 1
    halfacc_1 = num_corr_1 / h_size
    halfacc_2 = num_corr_2 / (te_labels.shape[1] - h_size)
    halfacc_diff = halfacc_2 - halfacc_1

    print("\n1st half and 2nd half accuracies for {} participants:".format(N))
    for i in range(N):
        print("    {:.4f}, {:.4f}".format(halfacc_1[i], halfacc_2[i]))
    print("\nMean 2nd - 1st half accuracies for {} participants:".format(N))
    print("    {:.4f}({:.4f})".format(halfacc_diff.mean(), halfacc_diff.std()))

    _, _, r_value, _, _ = stats.linregress(halfacc_1, halfacc_2)
    r_squared = r_value ** 2
    print("\nR Squared for 1st half vs. 2nd half accuracy: {:.4f}".
          format(r_squared))

    plt.plot(halfacc_1, halfacc_2, "ro")
    plt.xlabel('Accuracy for 1st 200 tested moves.')
    plt.ylabel('Accuracy for 2nd 200 tested moves.')
    plt.savefig("half1vshalf2.png")

    # ----------------------------------------
    # 3. Per square accuracies
    correct = np.zeros(N * 16).reshape((N, 4, 4))
    freq = np.zeros(N * 16).reshape((N, 4, 4))
    for i in range(N):
        # Each i is an individual
        for j in range(te_features.shape[1]):
            # Each j is a move
            x, y = te_features.item(i, j, 0, 0), te_features.item(i, j, 0, 1)
            freq[i][x][y] += 1
            if te_labels[i][j] == te_predictions[i][j]:
                correct[i][x][y] += 1
    acc = correct / freq

    print("\nMean per square accuracies for {} participants:".format(N))
    for y in range(4):
        s = "  "
        for x in range(4):
            s += "{:8.4f}({:.4f}) ".format(acc.mean(axis=0)[x][y],
                                           acc.std(axis=0)[x][y])
        print(s)

    # ----------------------------------------
    # 4. Corner/side/mid accuracies
    corner_correct = np.zeros(N)
    corner_freq = np.zeros(N)
    side_correct = np.zeros(N)
    side_freq = np.zeros(N)
    mid_correct = np.zeros(N)
    mid_freq = np.zeros(N)
    for i in range(N):
        # Each i is an individual
        for j in range(te_features.shape[1]):
            # Each j is a move
            coord = [te_features.item(i, j, 0, 0),
                     te_features.item(i, j, 0, 1)]
            coord = coord_to_type(coord)
            if coord == "corner":
                corner_freq[i] += 1
                if te_labels[i][j] == te_predictions[i][j]:
                    corner_correct[i] += 1
            elif coord == "side":
                side_freq[i] += 1
                if te_labels[i][j] == te_predictions[i][j]:
                    side_correct[i] += 1
            elif coord == "middle":
                mid_freq[i] += 1
                if te_labels[i][j] == te_predictions[i][j]:
                    mid_correct[i] += 1
    corner_acc = corner_correct / corner_freq
    side_acc = side_correct / side_freq
    mid_acc = mid_correct / mid_freq

    print("\nMean corner accuracies for {} participants, {} visits:".
          format(N, int(corner_freq.sum())))
    print("    {:.4f}({:.4f})".format(corner_acc.mean(), corner_acc.std()))
    print("\nMean side accuracies for {} participants, {} visits:".
          format(N, int(side_freq.sum())))
    print("    {:.4f}({:.4f})".format(side_acc.mean(), side_acc.std()))
    print("\nMean mid accuracies for {} participants, {} visits:".
          format(N, int(mid_freq.sum())))
    print("    {:.4f}({:.4f})".format(mid_acc.mean(), mid_acc.std()))

    # ----------------------------------------
    # 5. Per square visit distribution
    freq = np.zeros(N * 16).reshape((N, 4, 4))
    for i in range(N):
        # Each i is an individual
        for j in range(features.shape[1]):
            # Each j is a move
            x, y = features.item(i, j, 0, 0), features.item(i, j, 0, 1)
            freq[i][x][y] += 1
    freq = freq / features.shape[1]

    print("\nPer square visit distributions for {} participants:".format(N))
    for y in range(4):
        s = "  "
        for x in range(4):
            s += "{:8.4f}({:.4f}) ".format(freq.mean(axis=0)[x][y],
                                           freq.std(axis=0)[x][y])
        print(s)

    # ----------------------------------------
    # 6. Corner/mid/side visit distribution
    # reuse freq from 4
    corner_freq = np.zeros(N)
    mid_freq = np.zeros(N)
    side_freq = np.zeros(N)
    for i in range(N):
        corner_freq[i] = np.mean([freq[i][0][0],
                                  freq[i][0][3],
                                  freq[i][3][0],
                                  freq[i][3][3]])
        mid_freq[i] = np.mean([freq[i][1][1],
                               freq[i][1][2],
                               freq[i][2][1],
                               freq[i][2][2]])
        side_freq[i] = np.mean([freq[i][0][1],
                                freq[i][0][2],
                                freq[i][1][0],
                                freq[i][1][3],
                                freq[i][2][0],
                                freq[i][2][3],
                                freq[i][3][1],
                                freq[i][3][3]])

    print("\nMean corner distributions for {} participants:".format(N))
    print("    {:.4f}({:.4f})".format(corner_freq.mean(), corner_freq.std()))
    print("\nMean side distributions for {} participants:".format(N))
    print("    {:.4f}({:.4f})".format(side_freq.mean(), side_freq.std()))
    print("\nMean mid distributions for {} participants:".format(N))
    print("    {:.4f}({:.4f})".format(mid_freq.mean(), mid_freq.std()))

    # ----------------------------------------
    # 7. Up/down/left/right distributions
    key_presses = np.zeros(N * 4).reshape((N, 4))
    for i in range(N):
        for j in range(labels.shape[1]):
            key_presses[i][labels[i][j]] += 1

    print("\n Up/down/left/right key press distributions for {} participants:".
          format(N))
    print("    Up    : {:7.4f}({:.4f})".
          format(key_presses.mean(axis=0)[0], key_presses.std(axis=0)[0]))
    print("    Down  : {:7.4f}({:.4f})".
          format(key_presses.mean(axis=0)[1], key_presses.std(axis=0)[1]))
    print("    Left  : {:7.4f}({:.4f})".
          format(key_presses.mean(axis=0)[2], key_presses.std(axis=0)[2]))
    print("    Right : {:7.4f}({:.4f})".
          format(key_presses.mean(axis=0)[3], key_presses.std(axis=0)[3]))

    # ----------------------------------------
    # 8. Per square direction distributions
    persquare_dir = np.zeros(N * 4 * 4 * 4).reshape((N, 4, 4, 4))
    for i in range(N):
        for j in range(features.shape[1]):
            x, y = features.item(i, j, 0, 0), features.item(i, j, 0, 1)
            persquare_dir[i][x][y][labels[i][j]] += 1
    # Make tallies into distributions
    for i in range(N):
        for x in range(4):
            for y in range(4):
                total = (persquare_dir[i][x][y][0] +
                         persquare_dir[i][x][y][1] +
                         persquare_dir[i][x][y][2] +
                         persquare_dir[i][x][y][3])
                persquare_dir[i][x][y][0] /= total
                persquare_dir[i][x][y][1] /= total
                persquare_dir[i][x][y][2] /= total
                persquare_dir[i][x][y][3] /= total

    print("\nPer square direction distributions for {} participants:".
          format(N))
    for y in range(4):
        for d in range(4):
            s = "  "
            for x in range(4):
                s += ("{:8.4f}({:.4f}) ".
                      format(persquare_dir.mean(axis=0)[x][y][d],
                             persquare_dir.std(axis=0)[x][y][d]))
            print(s)
        print()

    # ----------------------------------------
    # 9. 1st order sequential effects
    seq_eff_num = np.zeros(N).reshape((N))
    seq_eff_den = np.zeros(N).reshape((N))
    for i in range(N):
        for j in range(features.shape[1] - 1):
            coord = [features.item(i, j + 1, 0, 0),
                     features.item(i, j + 1, 0, 1)]
            if coord_to_type(coord) == "middle":
                seq_eff_den[i] += 1
                dir_prev = labels[i][j]
                dir_curr = labels[i][j + 1]
                if dir_prev == dir_curr:
                    seq_eff_num[i] += 1
    seq_eff = seq_eff_num / seq_eff_den
    print("\n1st order seq. effect for middle tiles for {} participants:".
          format(N))
    print("    {:.4f}({:.4f})".format(seq_eff.mean(), seq_eff.std()))

    # ----------------------------------------
    # 10. Time taken vs. accuracy
    # Reuse total_acc from 1
    if times_taken is not None:
        _, _, r_value, _, _ = stats.linregress(times_taken, total_acc)
        r_squared = r_value ** 2

        print("\nAvg seconds per move for {} participants:".format(N))
        for i in range(N):
            print("    {:.4f}".format(times_taken[i]))
        print("\nR Squared for time taken vs. accuracy: {:.4f}".
              format(r_squared))

        plt.plot(total_acc, "ro")
        plt.xlabel('Average seconds per move')
        plt.ylabel('Total accuracy')
        plt.savefig("timevsacc.png")

    print()


def evaluate_csv(features, labels, te_predictions, times_taken):
    #    features.shape : (N, 1200, 1, 2)
    #      labels.shape : (N, 1200, 4) -> (N, 1200)
    # predictions.shape : (N, 1200, 4) -> (N, 1200)

    # Restore features to ints
    features = np.round(features * 3).astype(int)
    # Argmax on labels and predictions
    labels = np.argmax(labels, axis=2)
    te_predictions = np.argmax(te_predictions, axis=2)
    # Get test features/labels
    te_features = np.split(features,
                           [features.shape[1] - config.test_size],
                           1)[1]
    te_labels = np.split(labels, [labels.shape[1] - config.test_size], 1)[1]
    # Get num participants, num moves
    N = features.shape[0]

    # ----------------------------------------
    # 0. N
    print("N")
    print(N)
    print()

    # ----------------------------------------
    # 1. Time, accuracy
    total_acc = np.zeros(N)
    for i in range(N):
        total_acc[i] = np.sum(te_labels[i] == te_predictions[i])
    total_acc = total_acc / te_labels.shape[1]

    if config.input_dtype == "db":
        print("time, total accuracy".format(N))
        for i in range(N):
            print("{:.4f},{:.4f}".format(times_taken[i],
                                         total_acc[i]))
    else:
        print("total accuracy")
        for i in range(N):
            print("{:.4f}".format(total_acc[i]))

    # ----------------------------------------
    # 2. 1st half vs 2nd half accuracy
    h_size = int(te_labels.shape[1] / 2)
    num_corr_1 = np.zeros(N)
    num_corr_2 = np.zeros(N)
    for i in range(N):
        for j in range(h_size):
            if te_labels[i][j] == te_predictions[i][j]:
                num_corr_1[i] += 1
        for j in range(te_labels.shape[1] - h_size):
            if te_labels[i][j + h_size] == te_predictions[i][j + h_size]:
                num_corr_2[i] += 1
    halfacc_1 = num_corr_1 / h_size
    halfacc_2 = num_corr_2 / (te_labels.shape[1] - h_size)

    print("\n1st half acc, 2nd half acc")
    for i in range(N):
        print("{:.4f},{:.4f}".format(halfacc_1[i], halfacc_2[i]))

    _, _, r_value, _, _ = stats.linregress(halfacc_1, halfacc_2)
    print("\n1sthalf vs 2ndhalf r value")
    print(r_value)

    # ----------------------------------------
    # 3. Per square accuracies
    correct = np.zeros(N * 16).reshape((N, 4, 4))
    freq = np.zeros(N * 16).reshape((N, 4, 4))
    for i in range(N):
        # Each i is an individual
        for j in range(te_features.shape[1]):
            # Each j is a move
            x, y = te_features.item(i, j, 0, 0), te_features.item(i, j, 0, 1)
            freq[i][x][y] += 1
            if te_labels[i][j] == te_predictions[i][j]:
                correct[i][x][y] += 1
    acc = correct / freq

    print("\nMean, SD per square accuracies (N={})".format(N))
    for y in range(4):
        s = ""
        for x in range(4):
            s += "{:.4f},".format(acc.mean(axis=0)[x][y])
        s += "\n"
        for x in range(4):
            s += "{:.4f},".format(acc.std(axis=0)[x][y])
        s += "\n"
        print(s)
    print()

    # ----------------------------------------
    # 4. Corner/side/mid accuracies
    corner_correct = np.zeros(N)
    corner_freq = np.zeros(N)
    side_correct = np.zeros(N)
    side_freq = np.zeros(N)
    mid_correct = np.zeros(N)
    mid_freq = np.zeros(N)
    for i in range(N):
        # Each i is an individual
        for j in range(te_features.shape[1]):
            # Each j is a move
            coord = [te_features.item(i, j, 0, 0),
                     te_features.item(i, j, 0, 1)]
            coord = coord_to_type(coord)
            if coord == "corner":
                corner_freq[i] += 1
                if te_labels[i][j] == te_predictions[i][j]:
                    corner_correct[i] += 1
            elif coord == "side":
                side_freq[i] += 1
                if te_labels[i][j] == te_predictions[i][j]:
                    side_correct[i] += 1
            elif coord == "middle":
                mid_freq[i] += 1
                if te_labels[i][j] == te_predictions[i][j]:
                    mid_correct[i] += 1
    corner_acc = corner_correct / corner_freq
    side_acc = side_correct / side_freq
    mid_acc = mid_correct / mid_freq

    print("corner accuracies")
    for i in range(N):
        print(corner_acc[i])
    print("\nside accuracies")
    for i in range(N):
        print(side_acc[i])
    print("\nmid accuracies")
    for i in range(N):
        print(mid_acc[i])
    print()

    # ----------------------------------------
    # 5. Per square visit distribution
    freq = np.zeros(N * 16).reshape((N, 4, 4))
    for i in range(N):
        # Each i is an individual
        for j in range(features.shape[1]):
            # Each j is a move
            x, y = features.item(i, j, 0, 0), features.item(i, j, 0, 1)
            freq[i][x][y] += 1
    freq = freq / features.shape[1]

    print("mean, SD visit distribution (N = {})".format(N))
    for y in range(4):
        s = ""
        for x in range(4):
            s += "{:.4f},".format(freq.mean(axis=0)[x][y])
        s += "\n"
        for x in range(4):
            s += "{:.4f},".format(freq.std(axis=0)[x][y])
        s += "\n"
        print(s)
    print()

    # Skip 6. --------------------------------

    # ----------------------------------------
    # 7. Up/down/left/right distributions
    key_presses = np.zeros(N * 4).reshape((N, 4))
    for i in range(N):
        for j in range(labels.shape[1]):
            key_presses[i][labels[i][j]] += 1

    print("Up/down/left/right distributions")
    for i in range(N):
        print("{:.4f},{:.4f},{:.4f},{:.4f}".format(key_presses[i][0],
                                                   key_presses[i][1],
                                                   key_presses[i][2],
                                                   key_presses[i][3]))
    print()

    # ----------------------------------------
    # 8. Per square direction distributions
    persquare_dir = np.zeros(N * 4 * 4 * 4).reshape((N, 4, 4, 4))
    for i in range(N):
        for j in range(features.shape[1]):
            x, y = features.item(i, j, 0, 0), features.item(i, j, 0, 1)
            persquare_dir[i][x][y][labels[i][j]] += 1
    # Make tallies into distributions
    for i in range(N):
        for x in range(4):
            for y in range(4):
                total = (persquare_dir[i][x][y][0] +
                         persquare_dir[i][x][y][1] +
                         persquare_dir[i][x][y][2] +
                         persquare_dir[i][x][y][3])
                persquare_dir[i][x][y][0] /= total
                persquare_dir[i][x][y][1] /= total
                persquare_dir[i][x][y][2] /= total
                persquare_dir[i][x][y][3] /= total

    print("\nper square direction distributions (N = {})".format(N))
    for y in range(4):
        for d in range(4):
            s = ""
            for x in range(4):
                s += ("{:.4f}({:.4f}), ".
                      format(persquare_dir.mean(axis=0)[x][y][d],
                             persquare_dir.std(axis=0)[x][y][d]))
            print(s)
        print()

    # ----------------------------------------
    # 9. 1st order sequential effects
    seq_eff_num = np.zeros(N).reshape((N))
    seq_eff_den = np.zeros(N).reshape((N))
    for i in range(N):
        for j in range(features.shape[1] - 1):
            coord = [features.item(i, j + 1, 0, 0),
                     features.item(i, j + 1, 0, 1)]
            if coord_to_type(coord) == "middle":
                seq_eff_den[i] += 1
                dir_prev = labels[i][j]
                dir_curr = labels[i][j + 1]
                if dir_prev == dir_curr:
                    seq_eff_num[i] += 1
    seq_eff = seq_eff_num / seq_eff_den
    print("1st order seq. effect for middle tiles")
    for i in range(N):
        print("{:.4f}".format(seq_eff[i]))

    # Skip 10. -------------------------------

    # ----------------------------------------
    # 11. Corner/side/mid freq
    corner_freq = np.zeros(N)
    side_freq = np.zeros(N)
    mid_freq = np.zeros(N)
    for i in range(N):
        # Each i is an individual
        for j in range(features.shape[1]):
            # Each j is a move
            coord = [features.item(i, j, 0, 0), features.item(i, j, 0, 1)]
            coord = coord_to_type(coord)
            if coord == "corner":
                corner_freq[i] += 1
            elif coord == "side":
                side_freq[i] += 1
            elif coord == "middle":
                mid_freq[i] += 1

    print("corner freqs")
    for i in range(N):
        print(corner_freq[i])
    print("\nside freqs")
    for i in range(N):
        print(side_freq[i])
    print("\nmid freqs")
    for i in range(N):
        print(mid_freq[i])
    print()


def coord_to_type(coord):
    # Coord is a list of 2 numbers
    corner = [[0, 0],
              [0, 3],
              [3, 0],
              [3, 3]]
    side = [[0, 1],
            [0, 2],
            [1, 0],
            [1, 3],
            [2, 0],
            [2, 3],
            [3, 1],
            [3, 2]]
    mid = [[1, 1],
           [1, 2],
           [2, 1],
           [2, 2]]

    if coord in corner:
        return "corner"
    elif coord in side:
        return "side"
    elif coord in mid:
        return "middle"
    else:
        raise ValueError('Coordinate not recognized for grid.')


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main()
