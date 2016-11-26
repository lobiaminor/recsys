import datetime
import multiprocessing
import threading
import time
import yaml
import numpy as np
import pandas as pd
import scipy.sparse as sps
from colorama import Fore, Style
from sklearn.preprocessing import normalize
import itertools as it
import argparse

lock = threading.Lock()
last_time = 0
time_offset = 0

parser = argparse.ArgumentParser()
parser.add_argument('--rating_file', type=str, default="data/competition/interactions.csv")
parser.add_argument('--target_users', type=str, default="data/competition/target_users.csv")
parser.add_argument('--rating_key', type=str, default='rating')
parser.add_argument('--user_item_sep', type=str, default=',')
parser.add_argument('--item_item_sep', type=str, default='\t')
parser.add_argument('--user_key', type=str, default='user_id')
parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--k', type=int, default=63)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--prediction_file', type=str, default="target/competition/"
                                                           + str(datetime.datetime.now()).replace(" ", "").
                                                           replace(".", "").replace(":", "") + "submission.csv")
parser.add_argument('--user_bias', type=bool, default=True)
parser.add_argument('--item_bias', type=bool, default=True)
parser.add_argument('--rec_length', type=int, default=5)
parser.add_argument('--verbosity_level', type=str, default="Info")
parser.add_argument('--number_of_cpu', type=int, default=4)
parser.add_argument('--urm_file', type=str, default=None)
parser.add_argument('--similarity_matrix_file', type=str, default=None)
parser.add_argument('--estimations_file', type=str, default=None)
args = parser.parse_args()

""""
Some useful definitions to understand the code base
Users -> array of all the users: interacting + target
Target_users -> array of the users that are the target of our recommendations
Interacting_users -> array of the users that have at least one interaction

Function Interfaces
def time_print(string1, string2="", string3="", string4="", string5="", string6="", style="Log")
def user_knn(interactions, target_users_file=None, k=50, user_key="user_id", item_key="item_id")
def row_dealer(user_rating_matrix, target_users, uid_position_dic, task_name, offset=0, k=50)
def mapper(array, subject="the given input")
def data_importation(interactions, target_users_file=None, user_key="user_id", item_key="item_id")
def urm_computer(interactions, target_users_file=None, user_key="user_id", item_key="item_id")
def recommend(similarity_matrix, user_rating_matrix, target_users,)
def main(interactions, target_users_file=None, k=50, hold_out_percentage=0.8, prediction_file=None,
             ask_to_go=False, interaction_logic=0, user_key="user_id", item_key="item_id")
"""""


def time_print(string1, string2="", string3="", string4="", string5="", string6="", style="Log"):
    # This function is needed to print also the ex time of the task (approx)
    verbosity_levels = ["None", "Log", "Info", "Tips"]
    lock.acquire()
    global last_time
    millis = int(round(time.time() * 1000))
    diff = millis-last_time
    last_time = millis
    if style == "Log" and \
                verbosity_levels.index(args.verbosity_level) >= verbosity_levels.index("Log"):
        print(Style.BRIGHT + Fore.RED + "[", str((last_time-time_offset)/1000)+"]s ( +[", str(diff)
              + "]ms ) Log @ UB_CF : " + Style.RESET_ALL + Style.BRIGHT + Fore.MAGENTA +
              str(string1)+str(string2)+str(string3)+str(string4)+str(string5)+str(string6)+Style.RESET_ALL)
    elif style == "Info" and \
                  verbosity_levels.index(args.verbosity_level) >= verbosity_levels.index("Info"):
        print(Style.BRIGHT + Fore.BLUE + "[", str((last_time-time_offset)/1000)+"]s ( +[", str(diff)
              + "]ms ) Info @ UB_CF : " + Style.RESET_ALL + Style.BRIGHT + Fore.CYAN +
              str(string1)+str(string2)+str(string3)+str(string4)+str(string5)+str(string6)+Style.RESET_ALL)
    lock.release()


def user_knn(interactions, target_users_file=None, k=50, user_key="user_id", item_key="item_id", rating_key="rating",
             _item_bias_=1, _user_bias_=1):
    # This function is called in order to test or recommend using an user based collaborative filtering approach
    time_print("Building the URM...")

    # User rating matrix : sparse matrix with row_number rows and col_number columns
    # rows are users, columns are items
    # target users are the users for which we want to provide recommendations
    # position uid dic : dictionary associating to a row the corresponding user id
    # uid position : the symmetric
    # position iid dic : dictionary associating to a row the corresponding item id
    # iid position : the symmetric
    # user_rated_items is the dictionary in which for each user are listed all the ids of the items that he rated
    user_rating_matrix, target_users, users, position_uid_dic, uid_position_dic, position_iid_dic, iid_position_dic,\
        row_number, col_number, user_rated_items, item_rating_user = urm_computer(interactions, target_users_file,
                                                                                  user_key, item_key,
                                                                                  _item_bias_=_item_bias_,
                                                                                  rating_key=rating_key,
                                                                                  _user_bias_=_user_bias_)
    time_print("URM Successfully built")

    if args.normalize:
        # Normalization of the matrix. Row wise
        time_print("Starting the row-wise normalization", style="Info")
        user_rating_matrix = normalize(user_rating_matrix, 'l2', 1, copy=False)
        time_print("Normalization successfully completed")

    # Now it starts the computation of the similarity matrix
    # In order to improve performances 4 workers will be used to perform the task
    tot = len(target_users)
    pool = multiprocessing.Pool()
    step = int(tot/4)

    # The matrix is supposed to be squared (all users * all users) but in order to
    # improve performances we eliminate all the unused rows, keeping only the rows
    # corresponding to the target users.

    # The task of computing the matrix is split between 4 child processes
    first_block = pool.apply_async(row_dealer, (user_rating_matrix, target_users[:step],
                                                uid_position_dic, "One", 0, k,))
    second_block = pool.apply_async(row_dealer, (user_rating_matrix, target_users[step:2*step],
                                                 uid_position_dic, "Two", step, k,))
    third_block = pool.apply_async(row_dealer, (user_rating_matrix, target_users[2*step:3*step],
                                                uid_position_dic, "Three", 2*step, k,))
    fourth_block = pool.apply_async(row_dealer, (user_rating_matrix, target_users[3*step:],
                                                 uid_position_dic, "Four", 3*step, k,))

    # Fetching of the results from the workers who run the tasks
    first_data, first_row_ind, first_col_ind, first_new_uid_position_partial, first_similar_users_id\
        = first_block.get()
    second_data, second_row_ind, second_col_ind, second_new_uid_position_partial, second_similar_users_id \
        = second_block.get()
    third_data, third_row_ind, third_col_ind, third_new_uid_position_partial, third_similar_users_id \
        = third_block.get()
    fourth_data, fourth_row_ind, fourth_col_ind, fourth_new_uid_position_partial, fourth_similar_users_id \
        = fourth_block.get()

    # Merging the results obtained by the workers
    time_print("Merging the results obtained by the workers...", style="Info")

    # all the elements of the matrix obtaining stacking the partial results obtained by the w
    final_data = np.hstack((first_data, second_data))
    final_data = np.hstack((final_data, third_data))
    final_data = np.hstack((final_data, fourth_data))
    time_print("Merged values", style="Log")

    # all the row indices of the matrix obtaining stacking the partial results obtained by the w
    final_row_ind = np.hstack((first_row_ind, second_row_ind))
    final_row_ind = np.hstack((final_row_ind, third_row_ind))
    final_row_ind = np.hstack((final_row_ind, fourth_row_ind))
    time_print("Merged row indices", style="Log")

    # all the column indices of the matrix obtaining stacking the partial results obtained by the w
    final_col_ind = np.hstack((first_col_ind, second_col_ind))
    final_col_ind = np.hstack((final_col_ind, third_col_ind))
    final_col_ind = np.hstack((final_col_ind, fourth_col_ind))
    time_print("Merged column indices", style="Log")

    # all the ids of similar users for each user obtained stacking the partial results obtained by the w
    similar_users_id = dict(first_similar_users_id)
    similar_users_id.update(second_similar_users_id)
    similar_users_id.update(third_similar_users_id)
    similar_users_id.update(fourth_similar_users_id)
    time_print("Merged similar users dictionary", style="Log")

    # The new mapping user -> position in which only target users are present
    new_uid_position = dict(first_new_uid_position_partial)
    new_uid_position.update(second_new_uid_position_partial)
    new_uid_position.update(third_new_uid_position_partial)
    new_uid_position.update(fourth_new_uid_position_partial)
    time_print("Merged position dictionary", style="Log")

    time_print("Merging process completed! The matrix is built for ", str(len(new_uid_position)), " users.")
    time_print("Transforming the matrix into a sparse matrix", style="Info")

    # The similarity matrix is the sparse matrix target_users * users in which are stored the similarities
    similarity_matrix = sps.csc_matrix((final_data, (final_row_ind, final_col_ind)), shape=(tot, row_number))

    pool.close()
    pool.terminate()

    return target_users, similarity_matrix, new_uid_position, user_rating_matrix, position_iid_dic,\
        position_uid_dic, iid_position_dic, uid_position_dic, row_number, col_number, similar_users_id,\
        user_rated_items, item_rating_user


def row_dealer(user_rating_matrix, target_users, uid_position_dic, task_name, offset=0, k=50):
    # This function retrieves a set of rows with the similarity between a bunch of users and all the others
    nonzero = 0
    data = []
    row_ind = []
    col_ind = []
    counter = 0
    new_uid_position_partial = {}
    transposed_urm = user_rating_matrix.transpose(copy=True)
    tot = len(target_users)
    similar_users_id = {}
    # For each user computes the similarity with all the other by multiplying it to the transpose of the URM
    for user in target_users:
        # This is needed to have a dictionary with the new association row <-> id
        new_uid_position_partial[user] = offset + counter
        counter += 1
        # Check the row of the table associated to the next target user
        index = uid_position_dic[user]
        row = user_rating_matrix.getrow(index)
        # Performs the dot product and stores it into an array
        similarity_product_array = np.asarray(row.dot(transposed_urm).toarray()[0].copy())
        # We set to zero the similarity between the user and himself
        similarity_product_array[index] = 0
        # Counts the nonzero for analytics purposes
        nonzero += len(similarity_product_array.nonzero()[0])
        # Prints to show progress every 500 targets users processed
        if counter % 500 == 0:
            time_print("[", task_name, "]:", str(counter/tot*100), " % completed.", style="Info")
        # Here we select the k-nearest-neighbours to store
        k_nearest_indices = np.argpartition(similarity_product_array, -k)[-k:]

        # We store everything in 3 different arrays that will be used later to initialize a sparse matrix
        similar_users_id[user] = k_nearest_indices.copy()
        for iteration in range(0, k):
            row_ind.append(new_uid_position_partial[user])
            col_ind.append(k_nearest_indices[iteration])
            data.append(similarity_product_array[k_nearest_indices[iteration]])

    # We print a message to show the completion of the task
    time_print("[", task_name, "]: 100 % completed.")
    # We add info related to the number of average similar items encountered
    time_print("[", task_name, "]: in average were found ", str(nonzero/tot), "elements per row", style="Info")

    return data, row_ind, col_ind, new_uid_position_partial, similar_users_id


def mapper(array, subject="the given input"):
    # returns a dictionary where each value is associated to position
    to_return = {}
    return_to = {}
    for i in range(len(array)):
        to_return[array[i]] = i
    time_print("Completed position-id association for ", subject)
    for key in to_return.keys():
        return_to[to_return[key]] = key
    time_print("Completed id-position association for ", subject)
    return to_return, return_to


def data_importation(interactions, target_users_file=None, user_key="user_id", item_key="item_id", rating_key="rating"):
    # Import the interaction file in a Pandas data frame
    with open(interactions, 'r') as f:
        interactions_reader = pd.read_csv(interactions, delimiter='\t')

    # Here the program fills the array with all the users to be recommended
    # after this if/else target_users contains this information
    time_print("Listing the target users", style="Info")
    interacting_users = interactions_reader[user_key].as_matrix()
    if target_users_file is None:
        target_users = interacting_users
        users = interacting_users
    else:
        with open(target_users_file, 'r') as t:
            target_users_reader = pd.read_csv(target_users_file, delimiter='\t')
        target_users = target_users_reader[user_key].as_matrix()
        users = list(set(np.hstack((target_users, interacting_users))))
        t.close()

    # Computes the size of the user rating matrix using a simple parallel splitting to increase the speed
    time_print("Defining the dimension of the URM and mapping", style="Info")
    items = list(set(interactions_reader[item_key].as_matrix()))
    pool = multiprocessing.Pool()
    position_iid_dic_process = pool.apply_async(mapper, (items, "Items",))
    iid_position_dic, position_iid_dic = position_iid_dic_process.get()
    position_uid_dic_process = pool.apply_async(mapper, (users, "Users",))
    uid_position_dic, position_uid_dic = position_uid_dic_process.get()

    pool.close()
    pool.terminate()

    row_number = len(users)
    col_number = len(items)
    temp_dic = {}

    # Building a dictionary indexed by tuples (u_id,i_id) each one associated to the number of interactions
    # of the user u_id with the item i_id. The rating assigned to each user to an item is equal to the number
    # of interactions that he had with the item
    time_print("Building the temporary dictionary of interactions", style="Info")
    for i, row in interactions_reader.iterrows():
        key = (uid_position_dic[row[user_key]], iid_position_dic[row[item_key]])
        if key in temp_dic.keys():
            temp_dic[key] += row[rating_key]
        else:
            temp_dic[key] = row[rating_key]
    f.close()
    print('FILECHIUSOOOOOOOOOOOOOOOOOO')
    return temp_dic, row_number, col_number, target_users, users, position_uid_dic, \
        uid_position_dic, position_iid_dic, iid_position_dic


def urm_computer(interactions, target_users_file=None, user_key="user_id", item_key="item_id", rating_key="rating",
                 _user_bias_=1, _item_bias_=1):
    # This functions returns the couple, users to recommend and user rating matrix

    (temp_dic, row_number, col_number, target_users, users,
     position_uid_dic, uid_position_dic, position_iid_dic, iid_position_dic) = \
        data_importation(interactions, target_users_file, user_key, item_key, rating_key)

    # Converting dictionary values into integers and subtracting the user bias
    time_print("Computing the user bias for each user", style="Info")

    # For each line summing all the votes, and counting the number
    totals = {}
    row_elements = {}

    for key in temp_dic.keys():
        value = int(temp_dic[key])
        if key[0] in totals.keys():
            totals[key[0]] += value
            row_elements[key[0]] += 1
        else:
            totals[key[0]] = value
            row_elements[key[0]] = 1

    # Computing the bias of each user
    user_average = {}
    for key in temp_dic.keys():
        user_average[key[0]] = totals[key[0]]/row_elements[key[0]]

    # For each column summing all the votes and counting the number
    time_print("Computing the item bias for each item", style="Info")
    totals.clear()
    col_elements = {}

    for key in temp_dic.keys():
        value = int(temp_dic[key]) - user_average[key[0]]*_user_bias_
        if key[1] in totals.keys():
            totals[key[1]] += value
            col_elements[key[1]] += 1
        else:
            totals[key[1]] = value
            col_elements[key[1]] = 1

    # Computing the bias of each item
    item_average = {}
    for key in temp_dic.keys():
        item_average[key[1]] = totals[key[1]] / col_elements[key[1]]

    # Conversion and subtraction of the bias if requested
    data = []
    row_ind = []
    col_ind = []

    # user rated items is a dictionary in which to every user_id is associated the list of the items (indicated by
    # column number) that he rated
    user_rated_items = {}
    item_rating_users = {}
    time_print("Converting values to the right format and subtracting the user bias...", style="Info")
    for key in temp_dic.keys():
        value = int(temp_dic[key])
        data.append(value-user_average[key[0]]*_user_bias_-item_average[key[1]]*_item_bias_)
        row_ind.append(int(key[0]))
        col_ind.append(int(key[1]))
        key_user = position_uid_dic[key[0]]
        if key_user not in user_rated_items.keys():
            user_rated_items[key_user] = []
        if key[1] not in item_rating_users.keys():
            item_rating_users[key[1]] = []
        item_rating_users[key[1]].append((int(key[0])))
        user_rated_items[key_user].append((int(key[1])))

    tot = 0
    c = 0
    zeros = 0
    interactive_targets = 0
    for key in user_rated_items.keys():
        tot += len(user_rated_items[key])
        c += 1
        if key in target_users:
            interactive_targets += 1
        if len(user_rated_items[key]) == 0 and key in target_users:
            zeros += 1

    zeros += len(target_users)-interactive_targets
    time_print("in average every user evaluated ", str(tot/c)+" ", " items and ", str(zeros) +
               " of the target users evaluated no items", style="Info")

    # Create a sparse matrix from the interactions
    time_print("Creating the sparse matrix with the data", style="Info")
    user_rating_matrix = sps.csc_matrix((data, (row_ind, col_ind)), shape=(row_number, col_number))

    #data.clear()
    #row_ind.clear()
    #col_ind.clear()

    # returns the non-normalized User Rating Matrix
    return user_rating_matrix, target_users, users, position_uid_dic, uid_position_dic, position_iid_dic,\
        iid_position_dic, row_number, col_number, user_rated_items, item_rating_users


def recommend(similarity_matrix, user_rating_matrix, target_users, new_uid_position,
              position_iid_dic, position_uid_dic, user_rated_items, rec_length, expired_items,
              name="Generic Rec-Sys", item_rating_user=None):
    # This function takes a sparse similarity matrix, a list of users and the number of desired recommendations
    # and returns a dictionary u_id -> {rec_items_id}
    counter = 0
    non_profiled_users = []
    # variable to count the users for which is not possible to provide user based recommendations
    non_profiled_users_number = 0

    # To each key (user_id) is associated a tuple composed by : (item_id,est_rating)
    rec_dictionary = dict()
    rating_user_matrix = user_rating_matrix.transpose(copy=True)
    # For every user in target_users you have to compute
    for user in target_users:
        # The row of the similarity matrix corresponding to our user
        sim_sparse_row = similarity_matrix.getrow(new_uid_position[user])
        similarity_matrix_row = sim_sparse_row.todense().tolist()[0]

        # For every target user we have to compute the similarity for all the interesting items
        # where interesting means that have been evaluated at least by one of the neighbours

        # Building the interesting_item_list by merging the respective lines in the user rated items list
        interesting_items = []
        neighbours_columns = sim_sparse_row.nonzero()[1]
        for column in neighbours_columns:
            interesting_items.extend(user_rated_items[position_uid_dic[column]])
        interesting_items = list(set(interesting_items))
        recommendable_items = len(interesting_items)

        # marks as non profiled all the users for which there is no possibility to provide recommendations using
        # the collaborative filtering user based technique
        if recommendable_items == 0:
            non_profiled_users_number += 1
            non_profiled_users.append(user)
            user_rated_items[user] = []

        # Provides recommendations for avery one by initializing the recommendation list with the unseen and not expired
        # top pops to which are assigned negative weights. Then for the user for which is possible the collaborative
        # filtering technique will estimate the personalized recommendations
        if counter % 200 == 0:
            time_print("[", name, "] ", str(counter/len(target_users)*100), "% recommendations provided",
                       style="Info")
        counter += 1

        rec_dictionary[user] = \
            non_personalized_recommendation(rec_length, expired_items, user_rated_items[user])

        # Puts the non-recommendable items in a set in order to get O(1) membership check
        non_recommendable_set = set(it.chain(expired_items, user_rated_items[user]))
        # Now for every interested item we compute the estimated rating storing only the @rec_length better ones
        for item_column in interesting_items:
            # Checks the estimated rating only for valid item_ids
            if position_iid_dic[item_column] not in non_recommendable_set:
                weights = 0
                for ranker in item_rating_user[item_column]:
                    weights += similarity_matrix_row[ranker]
                if weights == 0:
                    weights = 1
                estimated_rating_dirty =\
                    np.asarray((sim_sparse_row.dot(rating_user_matrix.getrow(item_column).T).todense())).ravel()
                estimated_rating = estimated_rating_dirty / weights
                # To debug we signal when the estimated rating is not present
                if len(estimated_rating) == 0:
                    print("D'oh")
                    estimated_rating = [0]
                new_tuple = position_iid_dic[item_column], estimated_rating[0]
                rec_dictionary[user].append(new_tuple)

        # Stores in the dictionary only the rec_length best tuples, sorted by estimated rating
        rec_dictionary[user].sort(key=lambda tup: tup[1], reverse=True)

        rec_dictionary[user] = rec_dictionary[user][:rec_length]

    time_print("[", name, "] 100% recommendations provided. "+str(non_profiled_users_number)+"are not profiled",
               style="Info")

    return rec_dictionary, non_profiled_users


def write_recommendations(recommendations_dic, target_users, user_caption="user_id",
                          rec_item_caption="recommended_items",
                          user_items_sep=',', item_item_sep='\t',
                          prediction_file="predictions.csv"):
    # This function takes as input a dictionary composed by lists of tuples (item,estimated_rating). one for each
    # target user. User caption and rec_item_caption are the name to be printed in the header of the file,
    # user_items_sep and item_item_sep are the two separators respectively to divide the user_id from the
    # recommendations and each single recommended item_id. Prediction file is the name of the output file

    # Opening the output file
    output_file = open(prediction_file, "w+")
    dictionary_path = prediction_file.replace(".csv", "[DICTIONARY].txt")
    dictionary_file = open(dictionary_path, "w+")

    # Writing the header
    output_file.write(user_caption+user_items_sep+rec_item_caption+"\n")
    dictionary_file.write("{")
    # Writing the lines per each target user
    for user in target_users:
        # Each line starts with the id of the target user and a user_item_sep
        best_choices = recommendations_dic[user]
        line = str(user) + user_items_sep
        dic_line = str(user) + ":["
        # Then we iterate on all the recommendations for that user adding them to the string
        for recommendation in best_choices:
            line += str(recommendation[0]) + item_item_sep
            dic_line += "("+str(recommendation[0])+","+str(recommendation[1])+"),"

        dic_line += "]"
        # Removes the comma after the last tuple
        dic_line.replace("),]", ")]")

        # At the end we write the line corresponding to that target user on the file
        output_file.write(line+"\n")
        dictionary_file.write(dic_line+"\n")

    # Last token at the end of the dictionary file
    dictionary_file.write("}")
    # Closing the resources
    output_file.close()
    dictionary_file.close()

    time_print("Output operations concluded")


def check_expiration():
    # Returns a list in which all the items that are expired are stored
    # Import the item description file in a Pandas data frame
    filename = "data/competition/item_profile.csv"

    # This list contains the id's of all the expired items
    expired_ids = []
    with open(filename, 'r') as f:
        item_profiles_reader = pd.read_csv(filename, delimiter='\t')

    expired_items = item_profiles_reader[item_profiles_reader.active_during_test == 0]

    for i, row in expired_items.iterrows():
        expired_ids.append(row["id"])

    f.close()

    # return the list of invalid item_id
    return expired_ids


# TODO think about how to set the weights for the top pop suggestions (are they better or worse of an item similar
# but with a rating worse than your average rating?
def non_personalized_recommendation(rec_length, expired_items, user_rated_items):
    # For reasons of time this function is absolutely data-set dependent.
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

    # Builds the non recommendable set as a set to ensure 0(1) membership check
    non_recommendable_set = set(it.chain(expired_items, user_rated_items))

    for item in top_100_pops:
        if item not in non_recommendable_set:
            to_recommend.append((item, weight))
            weight -= 1
            counter += 1
            if counter == rec_length:
                break

    for i in range(0, rec_length-counter):
        to_recommend.append((0, weight))
        weight -= 1

    return to_recommend


def main(interactions, target_users_file=None, k = 60, user_key="user_id", item_key="item_id", rating_key="interaction_type",
         rec_length=5, _item_bias_=1, _user_bias_=1):

    # Setting the timers
    global last_time
    global time_offset
    last_time = int(round(time.time() * 1000))
    time_offset = last_time

    # building the matrices needed in order to recommend the right items
    target_users, similarity_matrix, new_uid_position, user_rating_matrix, position_iid_dic, \
        position_uid_dic, iid_position_dic, uid_position_dic, row_number, col_number, similar_users_columns, \
        user_rated_items, item_rating_user = user_knn(interactions, target_users_file, k, user_key, item_key,
                                                      rating_key, _item_bias_, _user_bias_)

    expired_items = check_expiration()

    # Splits the work between 4 worker in order to exploit all the 4 CPUs of my machine
    pool = multiprocessing.Pool()
    tot = len(target_users)
    step = int(tot/4)

    first_rec_dictionary_process = pool.apply_async(recommend, (similarity_matrix, user_rating_matrix,
                                                                target_users[0:step], new_uid_position,
                                                                position_iid_dic, position_uid_dic,
                                                                user_rated_items, rec_length, expired_items, "One",
                                                                item_rating_user,))
    second_rec_dictionary_process = pool.apply_async(recommend, (similarity_matrix, user_rating_matrix,
                                                                 target_users[step:2*step], new_uid_position,
                                                                 position_iid_dic, position_uid_dic,
                                                                 user_rated_items, rec_length, expired_items, "Two",
                                                                 item_rating_user,))
    third_rec_dictionary_process = pool.apply_async(recommend, (similarity_matrix, user_rating_matrix,
                                                                target_users[2*step:3*step], new_uid_position,
                                                                position_iid_dic, position_uid_dic,
                                                                user_rated_items, rec_length, expired_items, "Three",
                                                                item_rating_user,))
    fourth_rec_dictionary_process = pool.apply_async(recommend, (similarity_matrix, user_rating_matrix,
                                                                 target_users[3*step:4*step], new_uid_position,
                                                                 position_iid_dic, position_uid_dic,
                                                                 user_rated_items, rec_length, expired_items, "Four",
                                                                 item_rating_user,))

    # Fetches the results of the single workers
    first_rec_dic, first_non_int_users = first_rec_dictionary_process.get()
    second_rec_dic, second_non_int_users = second_rec_dictionary_process.get()
    third_rec_dic, third_non_int_users = third_rec_dictionary_process.get()
    fourth_rec_dic, fourth_non_int_users = fourth_rec_dictionary_process.get()

    # Ensembles the results in a single dictionary with recommendations
    non_profiled_users = np.hstack((first_non_int_users, second_non_int_users))
    non_profiled_users = np.hstack((non_profiled_users, third_non_int_users))
    non_profiled_users = np.hstack((non_profiled_users, fourth_non_int_users))

    # Assembling the final complete dictionary
    time_print("Assembling the final complete dictionary")
    rec_dictionary = dict(first_rec_dic)
    rec_dictionary.update(second_rec_dic)
    rec_dictionary.update(third_rec_dic)
    rec_dictionary.update(fourth_rec_dic)

    pool.close()
    pool.terminate()

    # Writing the recommendations file
    time_print("Writing the recommendations files")
    write_recommendations(rec_dictionary, target_users,
                          prediction_file=args.prediction_file, user_items_sep=args.user_item_sep,
                          item_item_sep=args.item_item_sep)

    time_print("All work done!")

    print(len(non_profiled_users))

main(args.rating_file, args.target_users, k=args.k,
     _item_bias_=args.item_bias, _user_bias_=args.user_bias, rating_key=args.rating_key, rec_length=args.rec_length,
     user_key=args.user_key, item_key=args.item_key)
