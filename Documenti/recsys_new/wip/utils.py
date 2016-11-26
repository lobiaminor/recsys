import time
from colorama import Fore, Style
import pandas as pd
import numpy as np
import multiprocessing


class Utils:

    verbosity_levels = ["None", "Log", "Info", "Tips"]
    last_time = 0
    verbosity_level = "Info"
    time_offset = 0
    name = "Printer"

    @staticmethod
    def time_print(string1, string2="", string3="", string4="", string5="", string6="", style="Log"):
        # This function is needed to print also the ex time of the task (approx)
        millis = int(round(time.time() * 1000))
        diff = millis-Utils.last_time
        Utils.last_time = millis
        if style == "Log" and \
                Utils.verbosity_levels.index(Utils.verbosity_level) >= Utils.verbosity_levels.index("Log"):
            print(Style.BRIGHT + Fore.RED + "[", str((Utils.last_time-Utils.time_offset)/1000)+"]s ( +[", str(diff)
                  + "]ms ) Log @ "+Utils.name+" : " + Style.RESET_ALL + Style.BRIGHT + Fore.MAGENTA +
                  str(string1)+str(string2)+str(string3)+str(string4)+str(string5)+str(string6)+Style.RESET_ALL)
        elif style == "Info" and \
                Utils.verbosity_levels.index(Utils.verbosity_level) >= Utils.verbosity_levels.index("Info"):
            print(Style.BRIGHT + Fore.BLUE + "[", str((Utils.last_time-Utils.time_offset)/1000)+"]s ( +[", str(diff)
                  + "]ms ) Info @ "+Utils.name+" : " + Style.RESET_ALL + Style.BRIGHT + Fore.CYAN +
                  str(string1)+str(string2)+str(string3)+str(string4)+str(string5)+str(string6)+Style.RESET_ALL)

    @staticmethod
    def mapper(array, subject="the given input"):
        # returns a dictionary where each value is associated to position
        to_return = {}
        return_to = {}
        for i in range(len(array)):
            to_return[array[i]] = i
        Utils.time_print("Completed position-id association for ", subject)
        for key in to_return.keys():
            return_to[to_return[key]] = key
        Utils.time_print("Completed id-position association for ", subject)
        return to_return, return_to

    @staticmethod
    def data_importation(interactions, target_users_file=None, user_key="user_id", item_key="item_id",
                         rating_key="rating"):
        # Import the interaction file in a Pandas data frame
        with open(interactions, 'r') as f:
            interactions_reader = pd.read_csv(interactions, delimiter='\t')

        # Here the program fills the array with all the users to be recommended
        # after this if/else target_users contains this information
        Utils.time_print("Listing the target users", style="Info")
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
        Utils.time_print("Defining the dimension of the URM and mapping", style="Info")
        items = list(set(interactions_reader[item_key].as_matrix()))
        pool = multiprocessing.Pool()
        position_iid_dic_process = pool.apply_async(Utils.mapper, (items, "Items",))
        iid_position_dic, position_iid_dic = position_iid_dic_process.get()
        position_uid_dic_process = pool.apply_async(Utils.mapper, (users, "Users",))
        uid_position_dic, position_uid_dic = position_uid_dic_process.get()

        pool.close()
        pool.terminate()

        row_number = len(users)
        col_number = len(items)
        temp_dic = {}

        # Building a dictionary indexed by tuples (u_id,i_id) each one associated to the number of interactions
        # of the user u_id with the item i_id. The rating assigned to each user to an item is equal to the number
        # of interactions that he had with the item
        Utils.time_print("Building the temporary dictionary of interactions", style="Info")
        for i, row in interactions_reader.iterrows():
            key = (uid_position_dic[row[user_key]], iid_position_dic[row[item_key]])
            if key in temp_dic.keys():
                temp_dic[key] += row[rating_key]
            else:
                temp_dic[key] = row[rating_key]

        f.close()

        return temp_dic, row_number, col_number, target_users, users, position_uid_dic, \
            uid_position_dic, position_iid_dic, iid_position_dic

    @staticmethod
    def check_expiration(filename):
        # Returns a list in which all the items that are expired are stored
        # Import the item description file in a Pandas data frame
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

    @staticmethod
    def init(time_offset, verbosity_level, name):
        Utils.time_offset = time_offset
        Utils.last_time = time_offset
        Utils.verbosity_level = verbosity_level
        Utils.name = name
