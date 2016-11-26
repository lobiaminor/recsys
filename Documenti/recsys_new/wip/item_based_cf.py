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
args = parser.parse_args()


class DataContainer:
    # user number, item number ->
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
    # user_id -> neighbour's numbers
    user_neighbours = {}


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
        user_average[user] = user_total[user] / user_number[user] * 1 if args.user_bias else 0

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
            item_number[user] = 1

    # Computes the average and stores it only if the item_bias arg is set to True
    for item in item_number.keys():
        item_average[item] = item_total[item] / item_number[item] * 1 if args.item_bias else 0

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
        if user in DataContainer.user_rated_items.keys():
            DataContainer.user_rated_items[DataContainer.urm_position_to_uid[user]].append(item)
        else:
            DataContainer.user_rated_items[DataContainer.urm_position_to_uid[user]] = [item]

    DataContainer.user_rating_matrix = \
        sps.csc_matrix((data, (row_indices, col_indices)),
                       shape=(DataContainer.number_of_users, DataContainer.number_of_items))

    u.time_print("Building auxiliary Data structures", style="Info")
    for user in DataContainer.users:
        if user in DataContainer.user_rated_items.keys():
            DataContainer.interacting_users.append(user)
        else:
            DataContainer.non_profiled_users.append(user)

    if args.normalize:
        u.time_print("Starting the row-wise normalization", style="Info")
        DataContainer.user_rating_matrix = normalize(DataContainer.user_rating_matrix, 'l2', 1, copy=False)
        u.time_print("Normalization successfully completed")

    u.time_print("User rating matrix successfully built!")


def main():
    u.init(int(time.time() * 1000), args.verbosity_level)
    DataContainer.user_rating_dictionary, DataContainer.number_of_users, DataContainer.number_of_items, \
    DataContainer.target_users, DataContainer.users, DataContainer.urm_position_to_uid, \
    DataContainer.uid_to_urm_position, DataContainer.urm_position_to_iid, DataContainer.iid_to_urm_position = \
        u.data_importation(args.rating_file, args.target_users, args.user_key, args.item_key, args.rating_key)
    print(DataContainer.user_rating_dictionary, DataContainer.user_rating_dictionary)


main()
