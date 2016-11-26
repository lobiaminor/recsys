import datetime
import multiprocessing
import threading
import time
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.preprocessing import normalize
from utils import Utils as u
import argparse

"""""
Steps:
1 import data and build the user rating matrix
2 build the similarity matrix from the user rating matrix keeping only the most similar users
_____________________________________________________________________________________________________________
3 evaluate for each user all the items that have been evaluated at least from one of the similar users in order
  to estimate the rating on that item. Choose the best estimated ratings and write them on 2 files
  a. submission file
  b. file with the couples item, estimated rating for possible further use
"""""

parser = argparse.ArgumentParser()
parser.add_argument('--rating_file', type=str, default=None)
parser.add_argument('--target_users', type=str, default=None)
parser.add_argument('--rating_key', type=str, default='rating')
parser.add_argument('--user_item_sep', type=str, default=',')
parser.add_argument('--item_item_sep', type=str, default='\t')
parser.add_argument('--user_key', type=str, default='user_id')
parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--k', type=int, default=50)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--prediction_file', type=str, default=None)
parser.add_argument('--user_bias', type=bool, default=True)
parser.add_argument('--item_bias', type=bool, default=True)
parser.add_argument('--rec_length', type=int, default=5)
parser.add_argument('--verbosity_level', type=str, default="Info")
parser.add_argument('--number_of_cpu', type=int, default=4)
parser.add_argument('--urm_fil', type=str, default=None)
parser.add_argument('--similarity_matrix_file', type=str, default=None)
parser.add_argument('--estimations_file', type=str, default=None)
args = parser.parse_args()


class DataContainer:
    # user number, item number
    user_rating_dictionary = {}
    # number of users, both interactive and non
    number_of_users = {}
    # number of all the rated items
    number_of_items = {}
    # user number -> user_id
    urm_position_to_uid = {}
    # user_id -> user number
    uid_to_urm_position = {}
    # item number -> item_id
    urm_position_to_iid = {}
    # item_id -> item number
    iid_to_urm_position = {}
    # target user number -> user_id
    target_number_uid = {}
    # user_id -> target user number
    uid_target_number = {}
    # list of the ids of all the target users
    target_users = []
    # list of the ids of all the users both interactive and non
    users = []
    # list of only the interacting users
    interacting_users = []
    # list of all the users without interactions
    non_profiled_users = []
    # matrix with a row for every user and a column for every item. The cell contains the vote
    user_rating_matrix = sps.csc_matrix(([0], ([0], [0])), shape=(1, 1))
    similarity_matrix = sps.csc_matrix(([0], ([0], [0])), shape=(1, 1))
    # user_id -> item_number
    # NOTE: item number is not item_id
    user_rated_items = {}
    # item number -> user number
    item_rating_users = {}
    # user_id -> neighbour's number
    user_neighbours = {}
    # Expired items id
    expired_items = []
    # Recommendations dictionary
    rec_dictionary = {}


def urm_computer():
    # Building the user rating matrix starting from the dictionary of interactions

    # Counting to get the elements to average the user
    u.time_print("Calculating the user bias", style="Info")
    user_number = {}
    user_total = {}
    user_average = {}

    for key in DataContainer.user_rating_dictionary.keys():
        user = key[0]
        if user in user_total.keys():
            user_total[user] += DataContainer.user_rating_dictionary[key]
            user_number[user] += 1
        else:
            user_total[user] = DataContainer.user_rating_dictionary[key]
            user_number[user] = 1

    # Computes the average and stores it only if the user_bias arg is set to True
    for user in user_number.keys():
        user_average[user] = user_total[user]/user_number[user] * 1 if args.user_bias else 0

    # Counting to get the elements to average the item
    u.time_print("Calculating the item bias", style="Info")
    item_number = {}
    item_total = {}
    item_average = {}

    for key in DataContainer.user_rating_dictionary.keys():
        item = key[1]
        user = key[0]
        if item in item_total.keys():
            item_total[item] += DataContainer.user_rating_dictionary[key] - user_average[user]
            item_number[item] += 1
        else:
            item_total[item] = DataContainer.user_rating_dictionary[key] - user_average[user]
            item_number[item] = 1

    # Computes the average and stores it only if the item_bias arg is set to True
    for item in item_number.keys():
        item_average[item] = item_total[item]/item_number[item] * 1 if args.item_bias else 0

    # Builds the array necessary to build the sparse matrix
    u.time_print("Converting the urm to a sparse representation", style="Info")
    data = []
    row_indices = []
    col_indices = []

    for key in DataContainer.user_rating_dictionary.keys():
        user = key[0]
        item = key[1]
        row_indices.append(user)
        col_indices.append(item)
        data.append(DataContainer.user_rating_dictionary[(user, item)] - item_average[item] - user_average[user])
        if DataContainer.urm_position_to_uid[user] in DataContainer.user_rated_items.keys():
            DataContainer.user_rated_items[DataContainer.urm_position_to_uid[user]].append(item)
        else:
            DataContainer.user_rated_items[DataContainer.urm_position_to_uid[user]] = [item]
        if DataContainer.urm_position_to_iid[item] in DataContainer.item_rating_users.keys():
            DataContainer.item_rating_users[DataContainer.urm_position_to_iid[item]].append(user)
        else:
            DataContainer.item_rating_users[DataContainer.urm_position_to_iid[item]] = [user]

    DataContainer.user_rating_matrix = \
        sps.csc_matrix((data, (row_indices, col_indices)),
                       shape=(DataContainer.number_of_users, DataContainer.number_of_items))

    u.time_print("Building auxiliary Data structures", style="Info")
    for user in DataContainer.users:
        if user in DataContainer.user_rated_items.keys():
            DataContainer.interacting_users.append(user)
        else:
            DataContainer.non_profiled_users.append(user)

    if args.normalize :
        u.time_print("Starting the row-wise normalization", style="Info")
        DataContainer.user_rating_matrix = normalize(DataContainer.user_rating_matrix, 'l2', 1, copy=False)
        u.time_print("Normalization successfully completed")

    u.time_print("User rating matrix successfully built!")


def similarity_matrix_computer():
    # In order to compute the similarity matrix the procedure to follow
    # is to multiply the urm to its transpose. However, due to the limited
    # amount of available memory it's necessary to perform that product
    # stacking the result of the multiplication of each row of the user rating
    # matrix to the transpose of that matrix. This can be done in parallel.

    # Splits the work between one worker for each of the cpu passed as parameters
    u.time_print("Splitting the work between the workers", style="Info")
    children = args.number_of_cpu
    pool = multiprocessing.Pool()
    tot_rows = len(DataContainer.target_users)
    step = int(tot_rows/children)
    child_processes = []

    # Assigning part of the tas to each worker
    for i in range(0, children):
        if i == children-1:
            child_processes.append(pool.apply_async(row_dealer, (step * i, -1, args.k, i,)))
        else:
            child_processes.append(pool.apply_async(row_dealer, (step*i, step*(i+1), args.k, i,)))

    # Fetching the results from the different workers
    results = []

    for i in range(0, children):
        results.append(child_processes[i].get())

    # Merging the obtained results
    u.time_print("Merging the results", style="Info")
    data = []
    row_indices = []
    col_indices = []

    DataContainer.uid_target_number = dict(results[0][3])
    DataContainer.target_number_uid = dict(results[0][4])
    DataContainer.user_neighbours = dict(results[0][5])

    for i in range(1, children):
        data = np.hstack((results[i - 1][0], results[i][0]))
        row_indices = np.hstack((results[i - 1][1], results[i][1]))
        col_indices = np.hstack((results[i - 1][2], results[i][2]))
        DataContainer.uid_target_number.update(results[i][3])
        DataContainer.target_number_uid.update(results[i][4])
        DataContainer.user_neighbours.update(results[i][5])

    # Building the similarity matrix as a sparse one
    u.time_print("Building sparse representation of the similarity matrix", style="Info")
    DataContainer.similarity_matrix = \
        sps.csc_matrix((data, (row_indices, col_indices)),
                       shape=(len(DataContainer.target_users), DataContainer.number_of_items))
    u.time_print("Similarity matrix built successfully")

    return


def row_dealer(start, end, k, thread_number):
    # Instantiate all the variables needed to collect the results
    data = []
    row_ind = []
    col_ind = []
    user_neighbours = {}
    uid_target_number = {}
    target_number_uid = {}

    # The index will be the new row index in the similarity matrix in which all the
    # rows corresponding to user that are not among the targets are removed
    index = 0
    non_profiled_number = 0
    # Copies the transpose of the user rating matrix
    rating_user_matrix = DataContainer.user_rating_matrix.transpose(copy=True)

    # If the end parameter is negative it means that the end is the end of the target users
    if end > 0:
        targets = DataContainer.target_users[start:end]
    else:
        targets = DataContainer.target_users[start:]

    # Computes the length of the targets
    total = len(targets)

    # For each one of the targets to be analyzed by the worker
    # the dot product is performed with the transposed of the user rating matrix
    for target in targets:
        row = DataContainer.uid_to_urm_position[target]
        uid_target_number[target] = index + start
        target_number_uid[index+start] = target
        sparse_product_line = \
            DataContainer.user_rating_matrix.getrow(row).dot(rating_user_matrix)
        dense_product_line = np.asarray(sparse_product_line.todense()).ravel()
        dense_product_line[DataContainer.uid_to_urm_position[target]] = 0
        # Only the k best indices are kept
        if len(dense_product_line.copy().nonzero()[0]) == 0:
            non_profiled_number += 1
        k_nearest_indices = np.argpartition(dense_product_line, -k)[-k:]
        k_nearest_indices = [k for k in k_nearest_indices if dense_product_line[k] is not 0]
        user_neighbours[target] = k_nearest_indices.copy()
        for i in range(0, k):
            data = dense_product_line[k_nearest_indices[i]]
            col_ind = k_nearest_indices[i]
            row_ind = index+start

        index += 1
        if index % 500 == 0:
            u.time_print("Thread#"+str(thread_number)+":Completion percentage->",
                         str(int(index*100/total)), "%", style="Info")

    u.time_print("Thread#"+str(thread_number)+":Completion percentage->100%. Process completed")
    print(non_profiled_number)
    return data, row_ind, col_ind, uid_target_number, target_number_uid, user_neighbours


def recommender(start, end, rec_length, thread_number):
    # For all users in target users elaborates recommendations producing a dictionary which
    # associates user_id to (item_id, estimated_rating) in order of rating
    dictionary = {}
    counter = 0
    non_rating_targets = 0
    for user in DataContainer.target_users[start:end]:
        rating_user_matrix = DataContainer.user_rating_matrix.transpose(copy=True)
        user_row = DataContainer.similarity_matrix.getrow(DataContainer.uid_target_number[user])
        similarities = np.asarray(user_row.copy().todense()).ravel()
        evaluated_items = non_personalized_init(rec_length, user)
        candidates = []
        already_seen = DataContainer.user_rated_items[user]
        for neighbour in DataContainer.user_neighbours[user]:
            if neighbour in DataContainer.user_rated_items.keys():
                candidates.extend(DataContainer.user_rated_items[neighbour])
        candidates = list(set(candidates))
        candidates_filtered = (x for x in candidates
                               if x not in DataContainer.expired_items and x not in already_seen)
        if user in DataContainer.user_rated_items.keys():
            for item in candidates_filtered:
                item_column = rating_user_matrix.getrow(item).T
                rankers = [x for x in DataContainer.user_neighbours[user] if x in DataContainer.item_rating_users[item]]
                summation = 0
                for ranker in rankers:
                    summation += similarities[ranker]
                estimated_rating_dirty = np.asarray(user_row.dot(item_column).todense()).ravel()
                estimated_rating = estimated_rating_dirty / summation

                evaluated_items.append((item, estimated_rating))

            evaluated_items.sort(key=lambda tup: tup[1], reverse=True)
        else:
            non_rating_targets += 1
            print(non_rating_targets)

        dictionary[user] = evaluated_items[:rec_length].copy()
        counter += 1
        if counter % 200 == 0:
            print(counter)

    return dictionary


def non_personalized_init(rec_length, user):
    # For reasons of time this function is absolutely data-set dependent. TODO make it independent
    top_100_pops = [1053452, 2778525, 1244196, 1386412, 657183, 2791339, 536047, 2002097, 1092821, 784737, 1053542,
                    278589, 79531, 1928254, 1133414, 1162250, 1984327, 343377, 1742926, 1233470, 1140869, 830073,
                    460717, 1576126, 2532610, 1443706, 1201171, 2593483, 1056667, 1754395, 1237071, 1117449, 734196,
                    437245, 266412, 2371338, 823512, 2106311, 1953846, 2413494, 2796479, 1776330, 365608, 1165605,
                    2031981, 2402625, 1679143, 2487208, 315676, 1069281, 818215, 419011, 931519, 470426, 1695664,
                    2795800, 2313894, 1119495, 2091019, 2086041, 84304, 72465, 499178, 2156629, 906846, 468120,
                    1427250, 117018, 471520, 2466095, 1920047, 1830993, 2198329, 335428, 2512859, 1500071, 2037855,
                    434392, 951143, 972388, 1047625, 2350341, 2712481, 542469, 1123592, 152021, 1244787, 1899627,
                    625711, 1330328, 2462072, 1419444, 2590849, 1486097, 1788671, 2175889, 110711,
                    16356, 291669, 313851]

    to_recommend = []
    weight = -2
    counter = 0
    for item in top_100_pops:
        if item not in DataContainer.expired_items and item not in DataContainer.user_rated_items[user]:
            to_recommend.append((item, weight))
            weight -= 1
            counter += 1
            if counter == rec_length:
                break

    for i in range(0, rec_length - counter):
        to_recommend.append((0, weight))
        weight -= 1

    return to_recommend


# TODO
def writing():
    return


def main():
    u.init(int(time.time()*1000), args.verbosity_level, "UB_CF")
    DataContainer.user_rating_dictionary, DataContainer.number_of_users, DataContainer.number_of_items,\
        DataContainer.target_users, DataContainer.users, DataContainer.urm_position_to_uid, \
        DataContainer.uid_to_urm_position, DataContainer.urm_position_to_iid, DataContainer.iid_to_urm_position =\
        u.data_importation(args.rating_file, args.target_users, args.user_key, args.item_key, args.rating_key)
    urm_computer()
    similarity_matrix_computer()
    DataContainer.expired_items = u.check_expiration("data/competition/item_profile.csv")
    recommender(0, len(DataContainer.target_users), args.rec_length, 1)

main()


