# -*- coding:utf-8 -*-

import pandas as pd
import Init
import csv
import os

class params ():
    def __init__ (self, mission='10C'):
        Init.main (['-s %s' % mission])
        self.configfile = None
        
        self.path_mission = Init.glbGetMissionPath ()
        
        # RM : Implémenter get_configfile() au lieu de mettre les chemins des data en dur
        #self.path_dd = Init.glbGetMissionPath() + r'2 Data/2 Calculs/16 01 26 New derived data/'
        self.path_dd = Init.glbGetMissionPath() + r'2 Data/2 Calculs/New derived data/'
        
        #self.path_ind = self.path_dd + 'Indic/'
        self.path_pd = Init.glbGetMissionPath() + r'2 Data/1 Received/Market data/Base/'
        #self.path_pd = Init.glbGetMissionPath() + r'2 Data/1 Received/Market data/Base - Test/'
        #self.path_excludedvar = Init.glbGetMissionPath() + r'2 Data/1 Received/Market data/'
        self.path_ld = Init.glbGetMissionPath() + r'2 Data\1 Received\Market data\Latest data\\'
        self.path_y = self.path_dd + 'Y/'
        self.path_x = self.path_dd + 'X/'
        #self.path_training = r'3 Appr/Backtest/EW_YF3MCN 01-01-2016/'
        #self.training_name = r'KL_ALL_SELECTED_EW_YF3MCN_01-01-2016_252_all_noQuantPerf.csv'
        
#         self.path_training =  '%s3 Appr/NoMask/20170320/EW_YF3MCN 01-01-2017/' % self.path_mission
#         self.training_name = 'RULES_EW_YF3MCN_01-01-2017_252.csv'
        
        #self.path_training =  '%s3 Appr/NoMask/20170320/EW_YF3MCN 01-01-2016/' % self.path_mission
        #self.training_name = 'KL_ALL_SELECTED_EW_YF3MCN_01-01-2017_252_EsgSignals2017.csv'
        
        #self.path_carbon = self.path_pd + 'carbon Data.csv'
        #self.price_name = 'price.csv'
        #self.sectoreprice_name = 'SectorPrice.csv'
        #self.marketcap_name = 'marketCap_values.csv'
        #self.esgCountry_name = 'esgCountry.csv'
        #self.constituent_name = 'constituents.csv'
        #self.interestRates_name = 'interestRates.csv'
        #self.estimates_name = 'Estimates.csv'
        #self.excludedvar_name = 'ExcludedVarDeriv.csv'
        
        #self.path_price = self.path_pd + self.price_name
        #self.path_SectorPrice = self.path_pd + self.sectoreprice_name
        #self.path_marketCap = self.path_pd + self.marketcap_name
        #self.path_esgCountry = self.path_pd + self.esgCountry_name
        #self.path_constit = self.path_pd + self.constituent_name
        #self.path_interestRates = self.path_pd + self.interestRates_name
        #self.path_EstimatesEps = self.path_pd + self.estimates_name
        #self.full_path_ExcludedVar = self.path_excludedvar + self.excludedvar_name
        #self.path_esgCompany_scores = self.path_pd + 'Company ESG ratings/scores/'
        #self.path_esgCompany_infos = self.path_pd + 'Company ESG ratings/infos/'
        #self.path_esgCompany_IncEvent = self.path_pd + 'Incidents Events/'
        #self.path_graph = self.path_dd + 'Graphes/'
        #self.path_save_esg = self.path_pd + 'Company ESG ratings/'
        #self.EsgCompany_ld_path = self.path_ld + 'Company ESG ratings.csv'
        #self.EsgCompany_base_path = self.path_save_esg + 'Company ESG ratings.csv'
        #self.path_index = self.path_dd + 'Index 2017-03-06.csv' 
        
    def find_delim (self, csv_path, delimiters=',;'):
        '''
        Function used to find a delimiters in a csv file
        
        Parameters:
        csv_path => Path of the csv file
        delimiters => String with different possibles delimiters
                      ex: ',;' means that the function will test ',' and ';'
        nb_bytes => number of bytes reading to find the best delimiter
    
        Return:
        dialect.delimiter => The best delimiter of the csv 
                             among the given delimiters
        '''
        #Test if the file exists
        assert os.path.isfile(csv_path), 'No csv file here %s' % csv_path
        f = open(csv_path, "rb")
        #Creation of a Sniffer object
        csv_sniffer = csv.Sniffer()
        #It reads nb_bytes bytes of the csv file and ...
        #... chooses the best delimiter among the given delimiters
        dialect = csv_sniffer.sniff(f.readline(), 
                        delimiters=delimiters)
        f.close()
        
        return dialect.delimiter
    
    def read_csv_ad (self, csv_path, header=0, index_col=0, sep=None, parse_dates=True, low_memory=False):
        '''
        
        :param csv_path: le chemin de fichier.
        :type csv_path: str.
        :param index_col: la colonne de l'index.
        :type index_col: str ou int.
        :param sep: le seperateur de fichier.
        :type sep: str.
        :param parse_dates: transformer l'index.
        :type parse_dates: boolean.
        :param low_memory:
        :type low_memory:
        '''
        sep_l = ',;'
        if sep is None:
            sep = self.find_delim (csv_path)
        
        df = pd.read_csv (csv_path, header=header, sep=sep, index_col=index_col, parse_dates=parse_dates,
                          low_memory=low_memory)
        
        sep_ope = sep_l.replace (sep, '')
        col0 = df.columns [0]
        nb_sep = col0.count (sep_ope)
        if len(df.columns) == 1 and nb_sep > 2:
            return pd.read_csv (csv_path, sep=sep_ope, index_col=index_col,
                                parse_dates=parse_dates,low_memory=low_memory)
        else:
            return df
        
    # RM : A coder pour récupérer les répertoires de lecture/écriture sur le répertoire 2.Data (Append, dérivation)
    def get_configfile (self):
        pass
    
    def get_mission_path (self):
        if self.configfile is None:
            return self.path_mission
        
    def get_ld_path (self):
        if self.configfile is None:
            return self.path_ld
        
    def get_base_path (self):
        if self.configfile is None:
            return self.path_pd
        
    def get_derived_data_path (self):
        if self.configfile is None:
            return self.path_dd
        
    def get_x_path (self):
        if self.configfile is None:
            return self.path_x
        
    def get_y_path (self):
        if self.configfile is None:
            return self.path_y
        
    def get_indicator_path (self):
        if self.configfile is None:
            return self.path_ind
        
    def get_price_path(self):
        if self.configfile is None:
            return self.path_price

    def get_EsgCompany_save_path (self):
        if self.configfile is None:
            return self.path_save_esg

    def get_EsgCompany_ld_path (self):
        if self.configfile is None:
            return self.EsgCompany_ld_path

    def get_EsgCompany_base_path (self):
        if self.configfile is None:
            return self.EsgCompany_base_path

    def get_SectorPrice_path(self):
        if self.configfile is None:
            return self.path_SectorPrice
        
    def get_Carbon_path(self):
        if self.configfile is None:
            return self.path_carbon
        
    def get_marketCap_path(self):
        if self.configfile is None:
            return self.path_marketCap
        
    def get_esgCountry_path(self):
        if self.configfile is None:
            return self.path_esgCountry

    def get_constit_path(self):
        if self.configfile is None:
            return self.path_constit

    def get_interestRates_path(self):
        if self.configfile is None:
            return self.path_interestRates

    def get_EstimatesEps_path(self):
        if self.configfile is None:
            return self.path_EstimatesEps

    def get_ExcludedVar_path(self):
        if self.configfile is None:
            return self.full_path_ExcludedVar

    def get_esgCompany_scores_path(self):
        if self.configfile is None:
            return self.path_esgCompany_scores

    def get_esgCompany_infos_path(self):
        if self.configfile is None:
            return self.path_esgCompany_infos

    def get_esgCompany_IncEvent_path(self):
        if self.configfile is None:
            return self.path_esgCompany_IncEvent
        
    def get_graph_path(self):
        if self.configfile is None:
            return self.path_graph
    
    def get_trainingFile_path (self):
        if self.configfile is None:
            return self.path_training + self.training_name
    
    def get_path_index (self):
        if self.configfile is None:
            return self.path_index
        self.path_ind