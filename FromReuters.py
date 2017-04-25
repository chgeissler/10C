#coding: utf-8
from ftplib import FTP
import json
import Init
import os
import collections
import pandas as pd
import TDataSet
import numpy as np
import datetime
import codecs
from os import listdir
from os.path import isfile, join
import zipfile
from shutil import copyfile
import re
import unicodedata
import get_params
#import ntpath

class do_fromReuters ():
    
    def __init__ (self, queue=None):
        self.queue = queue
        
        Init.main (['-s global_11StL'])
        
        #fichier des identifiants de FTP
        with open('ftpid.json') as data_file:
            self.data = json.load (data_file)
        
        self.instance_params = get_params.params ()
        #connexion, partie du code supprimée
        #self.ftp_login = FTP (self.data ['ftp'], self.data ['user'], self.data ['pswr'], timeout=5)
        #Init.Log ( "########### Connected to FTP ###########", self.queue)
        
        self.path_latest = self.instance_params.get_ld_path () #Init.glbMissionPath + r'2 Data\1 Received\Market data\Latest data\\'
        
        #self.path_base = self.instance_params.get_ld_path () #Init.glbMissionPath + r'2 Data\1 Received\Market data\Base\\'
        #Correction probablement Erreur Nourredine
        self.get_base_path = self.instance_params.get_base_path()
        
        self.path_archive = self.path_latest + 'Archive/'
        
        # os.path.dirname (os.path.realpath('__file__')) retourne le répertoire parent
        local_path = os.path.dirname (os.path.realpath('__file__'))
        self.path_temp_extrat = ''#local_path #os.path.join (local_path, '___temparchive/').replace ('\\', '/')
        
        # Inutile pour la mission C&M
        #path_constit = self.instance_params.get_constit_path ()
        
        if not os.path.exists (self.path_archive ):
            os.makedirs (self.path_archive)
            
        if os.path.isfile (path_constit):
            self.df_constit_bis = self.instance_params.read_csv_ad(path_constit, index_col=0)
        else:
            self.df_constit_bis = pd.DataFrame ()
        
        self.repo_to_del = ['Old',  'Archive - bkp']
        # Dictionary of all files name
        self.dic_roots = {}
    
    def main (self):
        '''permet de lancer la recuperation des fichiers à partir de FTP'''
#         
        self.downloadFilesInRepo ()
        self.write_file_in_local ()
        self.get_TidyList_names ()
        self.concatNSave_esg_files ()
        self.concatNSave_price_files ()
        self.concatNSave_IncNEv_files ()
        self.concatNSave_eps_files ()
        self.concatNSave_InterstR_files ()
        self.concatNSave_MarketCap_files ()
        self.concatNSave_SectorPrice_files ()
        
        self.clean ()
    
    def replaceSpaceIncolumns (self, df):
        '''supprime les caracters speciaux dans les noms de colonne d'un DataFrame'''
        for col in df.columns:
            col_m = col.replace('\n', '')
            df.rename (columns={col: col_m}, inplace=True)
            
            col_m1 = col_m.replace(' \r', ' ')
            df.rename (columns={col_m: col_m1}, inplace=True)
            
            col_m2 = col_m1.replace('\r ', ' ')
            df.rename (columns={col_m1: col_m2}, inplace=True)
            
            col_m3 = col_m2.replace('\r', ' ')
            df.rename (columns={col_m2: col_m3}, inplace=True)
        return df
    
    def replace_strings_by_nan (self, df_):
        '''garde que les nombres et supprime les autres caracteres dans un DataFrame'''
        def remove_car (x_):
            x = str(x_)
            #on garde que les "," et "." et les chiffres
            x = re.findall(r'[\-\,\.0-9]+', x)
            if len(x) == 1:
                x = x [0]
                x = x.replace (',', '')
                if x == '-':
                    return np.nan
                else:
                    return float(x)
            else:
                return np.nan
            
        df__ = df_.copy ()
        
        columns = df_.columns
        for col in columns:
            res = map(lambda x: remove_car(x), df__ [col])
            df__[col] = res
        return df__#.astype('float') 
      
    def extractzip (self, path_zip, path_output=None):
        '''Permet d'extraire un fichier Zip vers 'path_output'. 'path_zip' le chemin de fichier zip'''
        if path_output is None:
            path_output = self.path_temp_extrat
            
        with zipfile.ZipFile(path_zip, "r") as z:
            if not os.path.isdir (path_output) and path_output != '':
                os.makedirs (path_output)
            z.extractall (path_output)
        z.close ()
        
        
    def zipdir (self, pathdata, pathzip, mod = 'w'):
        '''Permet de ziper les données dans un repertoire.
        pathdata : le chemin des données.
        pathzip : le chemin de sauvgarde de zip.
        mod : 'w' pour ecrire un nouveau zip, 'r' pour lire,
            a' pour ajouter des fichiers a un zip deja existant,
            pour plus de détails : 
            https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile'''
        
        if not pathzip.endswith ('.zip'):
            pathzip += '.zip'
        
        fileinzip = []
        if mod == 'a':
            #extraction des fichiers existant dans le zip
            self.extractzip (pathzip, None)
            zipf = zipfile.ZipFile (pathzip, 'r')
            fileinzip = [i.filename for i in zipf.filelist]
            fileinzip = list ( set (fileinzip))
            zipf.close ()
        
        try:
            zipf = zipfile.ZipFile (pathzip, 'w')
    
            
            
            for root, _, files in os.walk (pathdata):
                #prendre en compte les fichiers deja archivés
                files += fileinzip
                files = list ( set (files))
                for fname in files:
                    fpath_src = os.path.join (root, fname)
                    fpath_dst = os.path.join (self.path_temp_extrat, fname)
                    copyfile (src=fpath_src, dst=fpath_dst)
                    zipf.write (fname)
                    os.unlink (fname)    
                        
            zipf.close ()
        except:
            pass
        
    def get_format_datetime (self, df):
        '''essaie de deviner le format des dates dans un DataFrame'''
        
        date = str(df.index[0])
        if '/' in date:
            sep = '/'
        elif '-' in date:
            sep = '-'
            
        format_dates = "%Y/%m/%d"
        if sep != '/':
            format_dates = format_dates.replace ('/', '-')
            
        if len (df.index) < 12:
            print "Please Check format date. Default format is Y-M-D."
        index_temp = None
        try:
            index_temp = pd.to_datetime(df.index, format=format_dates)
            bol_goodFromat = not len(index_temp) > 12
        except:
            bol_goodFromat = True
        cas=1
        #any (map(lambda x: x.month != current_month, index_temp))
        while bol_goodFromat:
            #print format_dates, cas
        
            if cas == 1 and bol_goodFromat:
                format_dates = "%m/%Y/%d"
            elif cas == 2 and bol_goodFromat:
                format_dates = "%m/%d/%Y"
            elif cas == 3 and bol_goodFromat:
                format_dates = "%d/%Y/%m"
            elif cas == 4 and bol_goodFromat:
                format_dates = "%Y/%d/%m"
            elif cas == 5 and bol_goodFromat:
                format_dates = "%d/%m/%Y"

            if sep != '/':
                format_dates = format_dates.replace ('/', '-')
                
            try:
                index_temp = pd.to_datetime (df.index, format=format_dates)
                bol_goodFromat = not len(index_temp) > 12
            except:
                bol_goodFromat = True
                
            cas += 1
            if cas > 5:
                return df.index.to_datetime()
                break
        return index_temp
    
    def get_datemtimeasStr (self, path_file):
        '''renvoi le date de modification d'un fichier. Fromat de renvoyer : YYYYMMDD en text.
        path_file : le chemin de fichier.'''
        #sc = time.ctime(os.path.getmtime(path_file))
        date = datetime.datetime.utcfromtimestamp(os.path.getmtime(path_file))
        
        day = str(date.day)
        month = str(date.month)
        
        if len(day) == 1:
            day = '0' + day
        if len(month) == 1:
            month = '0' + month
        
        return str(date.year) + month + day
        
    def DownloadFileFtp (self, path_ftp, path_save = None): #, returnDf = True):
        '''telecharge un fichier de ftp et le sauve dans un repertoire.
        path_ftp : le chemin de fichier dans le FTP.
        path_save : l'emplacement pour sauvegarder le fichier en local.
        '''
        if '/' in path_ftp:
            filename = path_ftp.split ('/')[-1]
        else:
            filename = path_ftp
            
        if path_save is None:
            path_save = filename
        else:
            if filename not in path_save:
                path_save = path_save + filename
        
        Init.Log ( 'Opening local file :' + filename, self.queue)
        file_ = open (path_save, 'wb')
        
        Init.Log ( 'Saving in : ' + path_save, self.queue)
        
        self.ftp_login.retrbinary ('RETR %s' % path_ftp, file_.write)
        
        # Clean up time
        Init.Log ( 'Closing file :' + filename, self.queue)
        file_.close ()
    
    def downloadFilesInRepo (self):
        '''rempli self.dic_roots.
        self.dic_roots : dictionnaire, les cles c'est les noms des repertoires dans le FTP.
        Les valeurs la liste des fichiers dans le repertoire'''
        
        Init.Log ( "Get all repository in Ftp", self.queue)
        listParentRoot = self.ftp_login.nlst ()
        for name in self.ftp_login.nlst ():
            if len (name) == 6:
                listParentRoot.remove(name)
                #on rajoute 01, ça permetra de trier les noms par date.
                listParentRoot += [name + '01']
        
        for rep_del in self.repo_to_del:
            if rep_del in listParentRoot:
                listParentRoot.remove(rep_del)
    
        for name in listParentRoot:
            try:
                #recuperation de list des fichiers dans le repertoire 'name'
                list_temp = self.ftp_login.nlst (name)
            except:
                #on supprimer le 01 pour avoir le vrai nom de repertoire name
                name = name[:-2]
                list_temp = self.ftp_login.nlst (name)
                
            list_csv_names = []
            for path_inftp in list_temp:
                #si le ftp renvoi le chemin complet, on recupere que le nom de fichier
                if '/' in path_inftp:
                    list_csv_names += [path_inftp.split('/')[1]]
                #si non juste le nom de fichier.
                else:
                    list_csv_names += [path_inftp]
                
                #list_temp = [i.split('/')[1] for i in list_temp]
                self.dic_roots [name] = list_csv_names
        
    def write_file_in_local (self):
        '''telechargement des fichiers de ftp et leur ecriture sur le disque. Archive egalement les repertoires.''' 
        
        Init.Log ( "Writing files in disk", self.queue)
              
        for reposi_name, files in self.dic_roots.items():
            if len (reposi_name) == 6:
                reposi_name_ = reposi_name + '01'
            else:
                reposi_name_ = reposi_name
                
            for file_ in files:
                dire = self.path_latest + reposi_name_ + '/' 
                if not os.path.exists(dire):
                    os.makedirs(dire)
                pathInFtp = '%s/%s' % (reposi_name , file_)
                #if '.' in file_ :#and not os.path.isfile (dire + file_):
                #la fonction nlst renvoi une liste avec le nom de fichier si on lui passe juste le nom d'un fichier.
                try:
                    if len (self.ftp_login.nlst(pathInFtp)) == 1:# a ameliorer !!
                        #file_ = renamedatefile (file_)
                        #telechargement
                        self.DownloadFileFtp (path_ftp = pathInFtp, path_save = dire + file_)
                except:
                    self.ftp_login = FTP (self.data ['ftp'], self.data ['user'], self.data ['pswr'], timeout=5)
                    if len (self.ftp_login.nlst(pathInFtp)) == 1:# a ameliorer !!
                        self.DownloadFileFtp (path_ftp = pathInFtp, path_save = dire + file_)
                #on archive les repertoires.
                fnamez = self.path_archive + '%s%s' % (reposi_name_ ,'.zip') 
                #si le zip de repertoire existe, on s'assure d'ajouter les nouveaux fichiers
                if not os.path.isfile (fnamez):
                    self.zipdir (pathdata=dire, pathzip=fnamez, mod='w')
#                     except:
#                         self.zipdir (pathdata=dire, pathzip=fnamez, mod='r')
                #si non on fait un nouveau zip
                else:
                    self.zipdir (pathdata=dire, pathzip=fnamez, mod='a')
    
    def get_TidyList_names (self):
        '''on stock les repertoires de latestdata dans self.list_repo.'''
        #on récupere les répertoires 
        self.list_repo = [f for f in listdir(self.path_latest) if not isfile(join(self.path_latest, f))]
        if 'Incidents Events' in self.list_repo:
            self.list_repo.remove ('Incidents Events')
        if 'Archive' in self.list_repo:
            self.list_repo.remove ('Archive')
        
        if type(self.repo_to_del) == str:
            self.list_repo.remove (self.repo_to_del)
        elif type(self.repo_to_del) == list:
            for rep in self.repo_to_del:
                if rep in self.list_repo:
                    self.list_repo.remove (rep)
        ### Audit files
        for repo in self.list_repo:
            list_files = os.listdir (self.path_latest + repo)
            for file_name in list_files:
                file_name= file_name.replace ('Incidents', '')
                if 'CSV' in file_name.upper () or 'XLSX' in file_name.upper ():
                    date = [s for s in file_name.split ('.')[0].split() if s.isdigit ()]
                    if len (date) > 0:
                        date = date[0]
                        if len (date) == 6:
                            date = date + '01'
                        datemfile = self.get_datemtimeasStr (self.path_latest + repo)
                         
                        date = pd.to_datetime (date)
                        datemfile = pd.to_datetime (datemfile)
                         
                        if date > datemfile:
                            Init.Log ( 'problem in :' + self.path_latest + repo + '\\' + file_name, self.queue)
                    else:
                        Init.Log ( 'no date in file name :' + self.path_latest + repo + '\\' + file_name, self.queue)
    
    def get_esg_TidyfilesNames (self):
        '''on recupere les noms des fichiers esg.
        la fonction renvoi un dictionnaire ordonné.
        Les cles c'est le nom d'un repertoire (ex 20161230), les valeurs c'est les chemins des fichiers esg.'''
        
        Init.Log ( "Processing for Scores", self.queue)
        
        name = 'Company ESG Ratings'
        dic_fileSectorEsg = {}
        #p = self.path_latest  + repo + '/Company ESG ratings 20151109.xlsx'
        
        for repo in self.list_repo:
            list_files = os.listdir (self.path_latest + repo)
            for file_name in list_files:
                  
                if name.upper () in file_name.upper () and ('.CSV' in file_name.upper () or '.XLSX' in file_name.upper ()):
                    if 'normalized' not in file_name:
                        file_name_ = file_name.replace('_normalized', '')
                        #on parcour le nom de fichier pour recuprer la date
                        date = [s for s in file_name_.split ('.')[0].split() if s.isdigit ()]
                        #date est une list avec un seul element str.
                        if len (date) > 0: date = date [0]
                        if len (date) == 6: date = date + '01'
                        if len (date) == 4: date = date + '0101'
                        if len (date) > 0:         
                            if date in dic_fileSectorEsg.values():
                                dic_fileSectorEsg [date] = self.path_latest  + repo + '/' + file_name
                            elif not date in dic_fileSectorEsg.values():
                                dic_fileSectorEsg [date] = self.path_latest  + repo + '/' + file_name
                                
        return collections.OrderedDict (sorted (dic_fileSectorEsg.items ()))
    
    def read_fileEsg (self, path_esg, index_col = 0):
        '''fonction special pour lire les fichier ESGs de format STL.'''
        try:
            if '.xlsx' in path_esg:
                df_esg = pd.read_excel (path_esg)
                df_esg.set_index (df_esg.columns [index_col], inplace = True)
                  
            elif '.csv' in path_esg:
                df_esg = pd.read_csv (path_esg, index_col = index_col, parse_dates = True)
            df_esg.index = pd.to_datetime (df_esg.index)
            
            #df_esg = df_esg.replace (r'.*', np.nan, regex = True)
            df_esg.dropna(how = 'all', inplace = True)
            df_esg = self.replaceSpaceIncolumns (df_esg)
            columns_uppers = [i.upper() for i in df_esg.columns]
            if 'ciqid' in df_esg.columns:
                df_esg.rename (columns = {'ciqid' : 'ciqid'.upper()}, inplace=True)
            elif 'CAPITAL IQ' in columns_uppers:
                index_ci = columns_uppers.index ('CAPITAL IQ')
                str_ci = df_esg.columns [index_ci]
                df_esg.rename (columns = {str_ci : 'ciqid'.upper()}, inplace=True)
                
            return df_esg
          
        except Exception as e: #'BadZipfile':
            Init.Log ( str(e) + ' ('+ path_esg + ')', self.queue)

    def unicodeToStr (self, value):
        try:
            str(value)
            return value
        except:
            return unicodedata.normalize('NFKD', value).encode ('ascii','ignore')
    
    def esg_treatement (self, df): 
        keys_to_remos = ['BROADSECTOR', 'BROADPEERGROUP', 'INDEXSECTOR', 'INDEXPEERGROUP']
        for key in keys_to_remos:
            for col in df.columns:
                col_ = col.replace (' ', '')
                if key in col_:
                    del df [col]
        return df
    
    def concatNSave_esg_files (self):
        '''regroupe tout les fichiers ESG dans un seul, et sauvegarde le fichier dans 'self.path_latest' '''
        #od = ordered dictionary
        self.od_esg = self.get_esg_TidyfilesNames ()
        
        first_EsgDf = True
        for path_esg in self.od_esg.values ():
            path_esg = r'' + path_esg
            if first_EsgDf:
                df_esg = self.read_fileEsg (path_esg)
                df_esg = df_esg.dropna (axis=1, how='all')
                ds_esg = TDataSet.TDataSet (df_esg)
                first_EsgDf = False
          
            else:
                df_esg2 = self.read_fileEsg (path_esg)
                for col in ds_esg.columns:
                    if col not in df_esg2:
                        cols_upper_list = [i.upper() for i in df_esg2.columns]
                        if col.upper() in cols_upper_list:
                            indexation2_col = cols_upper_list.index (col.upper())#renvoi l'index de col dans cols_upper_list
                            str2_col = df_esg2.columns [indexation2_col]
                            
                            df_esg2.rename (columns = {str2_col : col}, inplace=True)
                            
                df_esg2 = df_esg2.dropna (axis=1, how='all')
                
                max_date = ds_esg.index [-1]
                df_esg2 = df_esg2.loc [df_esg2.index > max_date]
                if len (df_esg2.dropna (how='all')) > 0:
                    ds_esg2 = TDataSet.TDataSet (df_esg2)
                    ds_esg = ds_esg.patch_append (ds_esg2)
          
                else:
                    Init.Log ( 'No new historical data in:' + path_esg + '\n', self.queue)
                
        if len(self.od_esg) > 0:
            Init.Log ( 'saving  Company ESG ratings file', self.queue)
            
            df_esg = pd.DataFrame(ds_esg)
            ds_esg = self.esg_treatement (ds_esg)
            df_esg ['Company Name'] = df_esg ['Company Name'].apply (lambda x: self.unicodeToStr (x))
            df_esg.to_csv (self.path_latest + 'Company ESG ratings.csv')
            
    def get_price_TidyfilesNames (self):
        '''on recupere les noms des fichiers price.
        la fonction renvoi un dictionnaire ordonné.
        Les cles c'est le nom d'un repertoire (ex 20161230), les valeurs c'est les chemins des fichiers price.'''
        Init.Log ( "Processing for Prices", self.queue)
          
        name = 'Price'
        dic_filePrice = {}
          
        for repo in self.list_repo:
            list_files = os.listdir (self.path_latest + repo)
            for file_name in list_files:
                  
                if name.upper() in file_name.upper() and ('sector price'.upper() not in file_name.upper()) and ('CSV' in file_name.upper() or 'XLSX' in file_name.upper()) :
                    date = [s for s in file_name.split('.')[0].split() if s.isdigit ()]
                    if len(date) > 0: date = date [0]
                    if len (date) == 6: date = date+'01'
                    if len (date) > 0:
                        dic_filePrice [date] = self.path_latest  + repo + '/' + file_name
        return dic_filePrice
      
    def read_fileprice (self, path_price, sheetname = 'price', header = 1, df_constit=None):
        '''fonction special pour lire les fichier prices de format STL.'''
        if '.xlsx' in path_price:
            df_price = pd.read_excel (path_price, sheetname = sheetname, header = header)
            colsToDel = ['ISIN', 'ISIN2', 'Exchange:ticker', 'securiy name']
            
            if len (df_constit) == 0:
                df_constit = df_price [colsToDel+['CIQID']]
                df_constit.set_index ('CIQID', inplace=True)
            else:
                for i in df_price ['CIQID']:
                    if i not in df_constit.index:
                        line = df_price .loc[df_price['CIQID']==i]
                        line = line.set_index ('CIQID')
                        if len (line) > 0:
                            df_constit = df_constit.append(line [df_constit.columns])
                        
            df_constit = df_constit.dropna (how='all')
                
            for col in colsToDel:
                if col in df_price.columns:
                    del df_price [col]
            
            if 'primary security' in df_price.columns:       
                df_price ['primary security'] = df_price ['primary security'].replace (np.nan, 1)
                df_price = df_price.loc [df_price['primary security']==1]
                del df_price ['primary security']
            
            df_price.set_index ('CIQID', inplace = True)
            df_price = df_price.T.dropna (how='all')
            df_price.index = self.get_format_datetime (df_price) #pd.to_datetime (df_price.index)
            
        elif '.csv' in path_price:
            df_price = pd.read_csv (path_price)
              
            if 'CIQID' not in df_price.columns:
                df_price = pd.read_csv (path_price, header = header)
            colsToDel = ['ISIN','ISIN2' , 'Exchange:ticker', 'securiy name']
            colinfo = ['primary security', 'ISIN','ISIN2' , 'Exchange:ticker', 'securiy name', 'CIQID']
            for col in colinfo:
                if col not in df_price.columns:
                    colinfo.remove(col)
            if len (df_constit) == 0:
                df_constit = df_price [colsToDel+['CIQID']]
                df_constit.set_index ('CIQID', inplace=True)
            else:
                for i in df_price ['CIQID']:
                    if i not in df_constit.index:
                        line = df_price .loc[df_price['CIQID']==i]
                        line = line.set_index ('CIQID')
                        if len (line) > 0:
                            cols_inline = np.intersect1d (df_constit.columns.tolist(), line.columns.tolist())
                            df_constit = df_constit.append(line [cols_inline])
            df_constit = df_constit.dropna (how='all')
                
            for col in colsToDel:
                if col in  df_price.columns:
                    del df_price [col]
            if 'primary security' in df_price.columns:
                df_price ['primary security'] = df_price ['primary security'].replace (np.nan, 1)
                df_price = df_price.loc [df_price['primary security']==1]
                del df_price ['primary security']
            
            df_price = df_price.set_index ('CIQID').T
        #drop les colonnes en double
        df_price = df_price.T.groupby(level = 0).first().T
        
        df_price.index = pd.to_datetime (df_price.index, dayfirst=True)
        df_price = df_price.T.dropna (how = 'all').T
        #df_price.index = pd.DatetimeIndex (df_price)
        df_price = df_price.sort().dropna (how = 'all')
        
        #df_price = df_price.replace (r'.*', np.nan, regex = True)
        df_price = df_price.T.drop_duplicates().T
        df_price = self.replace_strings_by_nan (df_price)

        df_price.dropna (how = 'all', inplace = True)
            
        return df_price, df_constit
      
    def get_dic_dfPrice (self):
        '''renvoi un dictionnaire ordiné des prices.
        les cles c'est le dates, les valeurs c'est une tuple ==> (l'historique de price, le chemin de fichier price)'''
        dic_dfprices = {}
        dic_filnameePrice = self.get_price_TidyfilesNames ()
        for date, path_f in dic_filnameePrice.items():
            Init.Log ( 'Reading: ' + path_f, self.queue)
            df_price, self.df_constit_bis = self.read_fileprice (path_f, df_constit=self.df_constit_bis)
            dic_dfprices [date] = (df_price, path_f)
            
        self.df_constit_bis.to_csv (self.path_latest + 'constituents.csv')
        return collections.OrderedDict (sorted(dic_dfprices.items()))
    
    def concatNSave_price_files (self):
        '''regroupe tout les fichiers price dans un seul, et sauvegarde le fichier dans 'self.path_latest' '''
        od_prices = self.get_dic_dfPrice()
        if len(od_prices) > 0:
            first = True
            for v in od_prices.values ():
                df, pathf = v
                if df is not None:
                    df.index = pd.to_datetime (df.index)
                if first and df is not None:
                    df_price = df.copy ()
                    first = False
                    #les colonnes en doubles
                    columns = [x for x, y in collections.Counter (df_price.columns.tolist()).items() if y > 1]
                    
                    for col in columns:
                        if col in df_price.columns:
                            del df_price [col]
                    ds_price = TDataSet.TDataSet (df_price)
                    ds_price.index = pd.to_datetime (ds_price.index)
                elif df is not None and not first:
                    df_price2 = df.copy ()
                    for col in columns:
                        if col in df_price2.columns:
                            del df_price2 [col]
                  
                    max_date = ds_price.index[-1]
                    df_price2 = df_price2.loc [df_price2.index > max_date]
          
                    ds_price2 = TDataSet.TDataSet (df_price2)
                    if len (ds_price2) > 0:
                        #pdb.set_trace ()
                        Init.Log ( 'Appending: ' + pathf, self.queue)
                        ds_price = ds_price.patch_append (ds_price2)
                    else:
                        Init.Log ( 'problem in or old file: ' + pathf, self.queue)
                        
        Init.Log ( "Saving Price file" , self.queue)
        
        df_price = pd.DataFrame(ds_price)
        df_price.to_csv (self.path_latest + 'price.csv')
    
    def get_incidentNev_TidyfilesNames (self):
        '''on recupere les noms des fichiers d'incidents.
        la fonction renvoi un dictionnaire ordonné.
        Les cles c'est le nom d'un repertoire (ex 20161230), les valeurs c'est les chemins des fichiers des incidents.'''
        
        Init.Log ( "Processing for Incidents and events", self.queue)
        name = 'Incidents'
        
        dic_fileIncidentEvents = {}
          
        for repo in self.list_repo:
            list_files = os.listdir (self.path_latest + repo)
            for file_name in list_files:
                  
                if name.upper() in file_name.upper() and ('CSV' in file_name.upper() or 'XLSX' in file_name.upper()) :
                    date = [s for s in file_name.split('.')[0].split() if s.isdigit ()]
                    if len(date) > 0: date = date [0]
                    if len (date) == 6: date = date + '01'
                    if len (date) > 0:
                        dic_fileIncidentEvents [date] = self.path_latest  + repo + '/' + file_name
        
        return dic_fileIncidentEvents
    
    def read_fileIncidents (self, path_incidents, header = 1):
        '''fonction special pour lire les fichier d'incident de format STL.'''
        dic_event = {}
        try:
            if '.csv' in path_incidents:
                df_Incident = pd.read_csv (path_incidents, index_col =0, header = 1, parse_dates=True)
            else:
                df_Incident = pd.read_excel (path_incidents, header = header)
              
            if len(df_Incident.dropna(how='all')) > 0:
                dic_replace = {'Low':0, 'Medium':1, 'High':2,
                               'Neutral':0, 'Moderate':1, 'Severely critical':2,
                               'Category 0':0, 'Category 1':1, 'Category 2':2,
                               'Category 3':3, 'Category 4':4,'Category 5':5}
      
                df_Incident.replace (dic_replace, inplace = True)
                  
                cols = ['Frequency', 'Number of incidents in chain', 'Exceptionality',
                        'Duration', 'Breadth', 'Impact Score', 'Risk Score', 'Answer Category']
                  
                for col in cols:
                      
                    if col in df_Incident.columns:
                        df = df_Incident [[col, 'Capital IQ ID', 'Incident Date']]
                        grouped = df.groupby (['Capital IQ ID','Incident Date'])
                        df_max = grouped.max ()
                        df_max.reset_index (inplace = True)
                        df_max = df_max.pivot ('Incident Date', 'Capital IQ ID', col)
                        if '.csv' in path_incidents:
                            df_max.index = pd.to_datetime (df_max.index, dayfirst=True)
                        else:
                            df_max.index = pd.DatetimeIndex (df_max.index)
                        dic_event [col] = df_max.sort().dropna(how = 'all')
                return dic_event
        except Exception as e:
            #pdb.set_trace ()
            Init.Log ( str(e) + ' ('+ path_incidents+')', self.queue)
      
    def get_dateIncidence (self, dic_):
        '''renvoi la date de permier df de dictionnaire'''
        date =  dic_.items ()[0][1].index[-1]
        month = str (date.month)
        day = str(date.day)
        if len(month) == 1:
            month = '0' + month
        if len(day) == 1:
            day = '0' + day
        return str(date.year) + month + day
    
    def get_dic_dfInciNEv (self):
        '''renvoi un dictionnaire ordiné des incidents.
        les cles c'est le dates, les valeurs c'est une tuple ==> (l'historique de price, le chemin de fichier price)'''
        dic_dfIncidents = {}
        dic_fileIncidentEvents = self.get_incidentNev_TidyfilesNames ()
        for date, path_f in dic_fileIncidentEvents.items ():
            #dic_dfIncidents [date] = read_fileIncidents (path_f)
            dic_ = self.read_fileIncidents (path_f)
            if dic_ is not None :
                if len (dic_) != 0:
                    date1 = self.get_dateIncidence (dic_)
                    dic_dfIncidents [(date1, date)] = dic_
                else:
                    Init.Log ( "problem in or old file" + path_f, self.queue)
            else:
                Init.Log ( "problem in " + path_f, self.queue)
              
        return collections.OrderedDict (sorted (dic_dfIncidents.items()))
    
    def concatNSave_IncNEv_files (self):
        '''regroupe tout les fichiers d'incidents dans un seul, et sauvegarde le fichier dans 'self.path_latest' '''
        first = True
        od_Incidents = self.get_dic_dfInciNEv ()
        for dic_ in od_Incidents.values ():
            #pdb.set_trace ()
            if first:
                dic_whole = dic_.copy ()
                first = False
            else:
                for key, df in dic_whole.items ():
                    if dic_ is not None:
                        if key in dic_.keys ():
                            ds_update = TDataSet.TDataSet (dic_ [key].copy ())
                            ds = TDataSet.TDataSet (dic_whole [key])
                            #date_max = ds.index [-1]
                            if len (ds_update.dropna (how='all')) > 0:
                                dic_whole [key] = ds.patch_append (ds_update)
          
          
        # In[21]:
        Init.Log ( "Saving ", self.queue)
        for name, ds in dic_whole.items ():
            df = pd.DataFrame (ds)
            df.to_csv (self.path_latest+ '\Incidents Events\\' + name + '.csv')
      
    def get_eps_TidyfilesNames (self):
        '''on recupere les noms des fichiers Eps.
        la fonction renvoi un dictionnaire ordonné.
        Les cles c'est le nom d'un repertoire (ex 20161230), les valeurs c'est les chemins des fichiers Eps.'''
        Init.Log ( "Processing for EPS", self.queue)
          
        name = 'Estimates'
        dic_fileEstimates = {}
          
        for repo in self.list_repo:
            list_files = os.listdir (self.path_latest + repo)
            for file_name in list_files:
                  
                if name.upper() in file_name.upper() and ('CSV' in file_name.upper() or 'XLSX' in file_name.upper()) :
                    date = [s for s in file_name.split('.')[0].split() if s.isdigit ()]
                    if len(date) > 0: date = date [0]
                    if len (date) == 6: date = date+'01'
                    if len (date) > 0:
                        dic_fileEstimates [date] = self.path_latest  + repo + '/' + file_name
        return dic_fileEstimates
    
    def read_fileEps (self, path_eps, sheetname = 1, header = 3):
        '''fonction special pour lire les fichier Eps de format STL.'''
        if '.xlsx' in path_eps:
            df_eps = pd.read_excel (path_eps, sheetname = sheetname, header = header)
        
            for col in df_eps.columns:
                cdate = pd.to_datetime(col)
                if type(cdate) != pd.Timestamp and col != 'CIQID':
                    del df_eps [col]
              
            df_eps = df_eps.set_index ('CIQID').T
            df_eps.replace ('#N/A Requesting Data...', np.nan, inplace = True)
            df_eps.replace ('(Capability Needed)', np.nan, inplace = True)
            df_eps.index = pd.DatetimeIndex (df_eps.index)
            df_eps.dropna (how = 'all', inplace = True)
              
        elif '.csv' in path_eps:
            df_eps = pd.read_csv (path_eps)
        
        
            for col in df_eps.columns:
                cdate = pd.to_datetime(col)
                if type(cdate) != pd.Timestamp and col != 'CIQID':
                    del df_eps [col]
                     
            df_eps = df_eps.set_index ('CIQID').T
            df_eps.index = self.get_format_datetime (df_eps)#pd.to_datetime (df_eps.index, dayfirst = True)
              
        duplicates = [x for x,y in collections.Counter (df_eps.columns.tolist ()).items () if y > 1]
        if len (duplicates) > 0:
            for Id in duplicates:
                del df_eps [Id]
                
        df_eps = self.replace_strings_by_nan (df_eps)   
        df_eps.dropna(how = 'all', inplace = True) 
        return df_eps
      
    def get_dic_dfEps (self):
        '''renvoi un dictionnaire ordiné des Eps.
        les cles c'est le dates, les valeurs c'est une tuple ==> (l'historique de price, le chemin de fichier Eps)'''
        dic_fileEstimates = self.get_eps_TidyfilesNames ()
        dic_dfEps = {}
          
        for date, path_f in dic_fileEstimates.items():
            Init.Log ( 'Reading: ' + path_f, self.queue)
            dic_dfEps [date] = self.read_fileEps (path_f), path_f
              
        return collections.OrderedDict (sorted(dic_dfEps.items()))
    
    def concatNSave_eps_files (self):
        '''regroupe tout les fichiers Eps dans un seul, et sauvegarde le fichier dans 'self.path_latest' '''
        od_Eps = self.get_dic_dfEps ()
        if len(od_Eps) > 0:
            first = True
              
            for v in od_Eps.values ():
                df_eps, path_f = v
                if df_eps is not None:
                    if first:
                        ds_eps = TDataSet.TDataSet (df_eps.copy())
                        first = False
                    else:
                        date_max = ds_eps.index[-1]
                        ds = TDataSet.TDataSet (df_eps.loc [df_eps.index > date_max])
         
                        if len (ds) > 0:
                            #pdb.set_trace ()
                            Init.Log ( 'Appending: ' + path_f, self.queue)
                            ds_eps = ds_eps.patch_append (ds)
                        else:
                            Init.Log ( "problem in or old file" + path_f, self.queue)
                      
              
            # In[27]:
            Init.Log ( "Saving", self.queue)
            df_eps = pd.DataFrame (ds_eps)
            df_eps.to_csv(self.path_latest + 'Estimates.csv')
      
    def get_interestR_TidyfilesNames (self):
        '''on recupere les noms des fichiers des taux d'interet.
        la fonction renvoi un dictionnaire ordonné.
        Les cles c'est le nom d'un repertoire (ex 20161230), les valeurs c'est les chemins des fichiers des taux d'interet.'''
        Init.Log ( 'Processing for Interest rates', self.queue)
        name = 'OECD'
        dic_fileInterestRate = {}
          
        for repo in self.list_repo:
            list_files = os.listdir (self.path_latest + repo)
            for file_name in list_files:
                  
                if name.upper() in file_name.upper():# and 'TXT' in file_name.upper() :
                    date = [s for s in file_name.split('.')[0].split() if s.isdigit ()]
                    if len(date) > 0: date = date [0]
                    if len (date) == 6: date = date+'01'
                    if len (date)>0:
                        dic_fileInterestRate [date] = self.path_latest  + repo + '/' + file_name
        #to utf8
        for _, path_f in dic_fileInterestRate.items ():
            if '.txt'.upper() in path_f.upper():
                path_fu = path_f.replace ('.txt', '_.txt')
                BLOCKSIZE = 1048576 # or some other, desired size in bytes
                with codecs.open (path_f, "r", "utf-16") as sourceFile:
                    with codecs.open (path_fu, "w", "utf-8") as targetFile:
                        while True:
                            contents = sourceFile.read (BLOCKSIZE)
                            if not contents:
                                break
                            targetFile.write (contents)
                        
        return dic_fileInterestRate
      
    def read_fileInterestRate (self, path_ir, index_col = 0, sep = "\t"):
        '''fonction special pour lire les fichier des taux d'interet de format STL.'''
        try:
            df_interestRates = pd.read_csv (path_ir, index_col = index_col, sep = sep)
            if len(df_interestRates.columns)==0:
                sep = self.instance_params.find_delim(path_ir)
                df_interestRates = pd.read_csv (path_ir, index_col = index_col, sep = sep)
            df_interestRates = df_interestRates.pivot (index = 'TIME', columns = 'Country', values = 'Value')
            df_interestRates = df_interestRates.replace ('..', np.nan).dropna (how = "all")
              
            def datetime_intersrate (date):
                char1, char2 = date.split ('-')
                try:
                    y = eval(char1)
                    m = char2
                except:
                    y = eval(char2)
                    m = char1
                date = '01-%s-%s' % (m, y)
                return pd.to_datetime (date)#, format'%d-%m-%Y')
              
            df_interestRates.index = [datetime_intersrate (x) for x in df_interestRates.index]
            df_interestRates.index = df_interestRates.index.to_datetime (dayfirst=True)
            df_interestRates.index = df_interestRates.index.to_period('M').to_timestamp('M')
              
            df_interestRates = df_interestRates.resample ('B')
            df_interestRates = df_interestRates.fillna (method = 'ffill').fillna (method = 'bfill')
            df_interestRates = df_interestRates / 100.
             
            
            #df_interestRates = df_interestRates.replace (r'.*', np.nan, regex = True) 
            df_interestRates = self.replace_strings_by_nan (df_interestRates)   
            df_interestRates.dropna(how = 'all', inplace = True)
            
            return df_interestRates
          
        except Exception as e:
            #pdb.set_trace ()
            Init.Log ( str(e) + ' ('+ path_ir + ')', self.queue)
      
    def get_dic_dfInterstR (self):
        '''renvoi un dictionnaire ordiné des Taux d'interet.
        les cles c'est le dates, en valeur l'historique des taux d'interet'''
        dic_dfInterestRate = {}
        dic_fileInterestRate = self.get_interestR_TidyfilesNames ()  
        for date, path_f in dic_fileInterestRate.items():
            if 'forecast' not in path_f:
                path_fu = path_f.replace ('.txt', '_.txt')
                dic_dfInterestRate [date] = self.read_fileInterestRate (path_fu)
              
        return collections.OrderedDict (sorted (dic_dfInterestRate.items()))
    
    def concatNSave_InterstR_files (self):
        '''regroupe tout les fichiers des taux d'interets dans un seul, et sauvegarde le fichier dans 'self.path_latest' '''
        od_InterestR = self.get_dic_dfInterstR ()
        first = True
          
        for df in od_InterestR.values ():
            if first:
                ds_interestRates = TDataSet.TDataSet (df.copy ())
                first = False
            #df_interestRates = df_interestRates.append (df)
            date_max = ds_interestRates.index [-1]
            ds = TDataSet.TDataSet (df.loc [df.index > date_max])
            if len (ds) >0:
                ds_interestRates = ds_interestRates.patch_append (ds)
        
        if len(od_InterestR)>0:
            df_interestRates = pd.DataFrame (ds_interestRates)
            df_interestRates ['date'] = df_interestRates.index
            df_interestRates = df_interestRates.drop_duplicates ('date', take_last = True)
            del df_interestRates ['date']
              
              
            # In[34]:
            Init.Log ( 'Saving', self.queue)
            df_interestRates.to_csv (self.path_latest + 'interestRates.csv')
      
    def get_MarketCap_TidyfilesNames (self):
        '''on recupere les noms des fichiers MarketCap.
        la fonction renvoi un dictionnaire ordonné.
        Les cles c'est le nom d'un repertoire (ex 20161230), les valeurs c'est les chemins des fichiers MarketCap.'''
        Init.Log ( "Processing for Market Cap", self.queue)
          
        name = 'marketcap'
        dic_fileMarketCap = {}
          
        for repo in self.list_repo:
            list_files = os.listdir (self.path_latest + repo)
            for file_name in list_files:
                  
                if name.upper() in file_name.upper() and ('CSV' in file_name.upper() or 'XLSX' in file_name.upper()):
                    date = [s for s in file_name.split('.')[0].split() if s.isdigit ()]
                    if len(date) > 0: date = date [0]
                    if len (date) == 6: date = date+'01'
                    if len (date)>0:
                        dic_fileMarketCap [date] = self.path_latest  + repo + '/' + file_name
                          
        return collections.OrderedDict (sorted (dic_fileMarketCap.items()))
      
    def read_fileMarketCap (self, path_mc, sheetname = 1, header = 2, index_col = 0, df_constit=None):
        '''fonction special pour lire les fichier des markets cap de format STL.'''
        Init.Log ( 'Reading: ' + path_mc, self.queue)
        
        if '.csv' in path_mc:
            df_mc = pd.read_csv (path_mc, header = header, index_col = index_col)
        else:
            df_mc = pd.read_excel (path_mc, sheetname = sheetname, header = header)
        list_columns = ['company name', 'CIQID', 'Sector', 'Weighting', 'Currency', 'Country', 'ISIN', 'Name']
        dfinfo = df_mc [['CIQID', 'Sector', 'Country']]
        dfinfo ['Name'] = dfinfo.index
        df2 = df_mc.copy ()
        for col in list_columns:
            if col in df2:
                del df2 [col]
          
        df2.index = df_mc ['CIQID']
        dfinfo.set_index('CIQID', inplace=True)

        #Les capitalisations par action au cours du temps
        df_mc = df2.T.copy ()
        #Les informations par action
        #df_marketCap_info = df1.set_index ('CIQID')
        if ' =k2' in df_mc.index:
            df_mc.drop (' =K2', inplace = True)
          
        df_mc.dropna(how = 'all', inplace = True)
        df_mc.index = self.get_format_datetime (df_mc) #pd.DatetimeIndex (df_mc.index)
          
        duplicates = [x for x,y in collections.Counter (df_mc.columns.tolist ()).items() if y > 1]
          
        if len (duplicates) > 0:
            for Id in duplicates:
                del df_mc [Id]
        
        #df_mc = df_mc.replace (r'.*', np.nan, regex = True) 
        df_mc = self.replace_strings_by_nan (df_mc)  
        df_mc.dropna(how = 'all', inplace = True)
        
        cols_constit = ['Sector', 'Country', 'Name']
        if df_constit is not None:
            dfinfo = dfinfo.groupby (level=0).first()
            for col in cols_constit:
                if col not in df_constit.columns:
                    df_constit [col] = dfinfo [col]
            for qi in dfinfo.index:
                if qi not in df_constit.index:
                    line = dfinfo.loc [[qi]]
                    df_constit = df_constit.append(line)
            df_constit [['Sector', 'Country', 'Name']] = dfinfo [['Sector', 'Country', 'Name']]
            
        return df_mc, df_constit
          
          
      
    # In[37]:
    
    def get_dic_dfMarketCap (self):
        '''renvoi un dictionnaire ordiné des Market cap.
        les cles c'est le dates, en valeur l'historique des MarketCap'''
        od_mc = self.get_MarketCap_TidyfilesNames()
        dic_dfmc = {}
        
        for date, path_f in od_mc.items():
            dic_dfmc [date], self.df_constit_bis = self.read_fileMarketCap (path_f, df_constit=self.df_constit_bis)
              
        return collections.OrderedDict (sorted (dic_dfmc.items()))
    
    def concatNSave_MarketCap_files (self):
        '''regroupe tout les fichiers market cap dans un seul, et sauvegarde le fichier dans 'self.path_latest' '''
        od_dfmc = self.get_dic_dfMarketCap()
        first = True
        for df in od_dfmc.values ():
            if first:
                ds_mc = TDataSet.TDataSet (df.copy ())
                first = False
            #df_interestRates = df_interestRates.append (df)
            date_max = ds_mc.index [-1]
            ds = TDataSet.TDataSet (df.loc [df.index > date_max])
            if len (ds) > 0:
                ds_mc = ds_mc.patch_append (ds)
                
        if len(od_dfmc) > 0:
            Init.Log ('Saving', self.queue)
            df_mc = pd.DataFrame (ds_mc)
            df_mc.to_csv (self.path_latest + 'marketCap_values.csv')
            self.df_constit_bis.to_csv (self.path_base + 'constituents.csv')
     
    def get_SectorPrice_TidyfilesNames (self):
        '''on recupere les noms des fichiers SectorPrice.
        la fonction renvoi un dictionnaire ordonné.
        Les cles c'est le nom d'un repertoire (ex 20161230), les valeurs c'est les chemins des fichiers SectorPrice.'''
        Init.Log ( 'Processing Price sector', self.queue)
        
        name = 'Sector price'
        dic_fileSectorPrice = {}
         
        for repo in self.list_repo:
            list_files = os.listdir (self.path_latest + repo)
            for file_name in list_files:
                 
                if name.upper() in file_name.upper() and ('CSV' in file_name.upper() or 'XLSX' in file_name.upper()):
                    date = [s for s in file_name.split('.')[0].split() if s.isdigit ()]
                    if len(date) > 0: date = date [0]
                    if len (date) == 6: date = date+'01'
                    if len (date)>0:
                        dic_fileSectorPrice [date] = self.path_latest  + repo + '/' + file_name
                         
        return collections.OrderedDict (sorted (dic_fileSectorPrice.items()))
     
    def read_fileSectorPrice (self, path_sp, sheetname = 1, header = 1, index_col = 0):
        '''fonction special pour lire les fichier Sector price de format STL.'''
        Init.Log ( 'Reading: ' + path_sp, self.queue)

        if '.xlsx' in path_sp:
            df_SectorPrice = pd.read_excel (path_sp, sheetname = sheetname, header = header, index_col = index_col)
            if 'ISIN' in df_SectorPrice.columns:
                del df_SectorPrice ['ISIN']
            if 'ISIN2' in df_SectorPrice.columns:
                del df_SectorPrice ['ISIN2']
            if 'Pricing Date' in df_SectorPrice.columns:
                del df_SectorPrice ['Pricing Date']
            df_SectorPrice = df_SectorPrice.T
             
        elif '.csv' in path_sp:
            df_SectorPrice = pd.read_csv (path_sp)
             
            if 'Sector' not in df_SectorPrice.columns:
                df_SectorPrice = pd.read_csv (path_sp, header=header)
            col_to_del = ['ISIN', 'ISIN2', 'Pricing Date', 'Ticker']
             
            for col in col_to_del:
                if col in df_SectorPrice.columns:
                    del df_SectorPrice [col]
            df_SectorPrice = df_SectorPrice.set_index ('Sector').T
 
        df_SectorPrice.index.name = 'Date'
        df_SectorPrice.index = self.get_format_datetime (df_SectorPrice) #pd.DatetimeIndex (df_SectorPrice.index)
        
        #df_SectorPrice = df_SectorPrice.replace (r'.*', np.nan, regex = True) 
        df_SectorPrice = self.replace_strings_by_nan (df_SectorPrice)  
        df_SectorPrice.dropna (how = 'all', inplace = True)    
        
        return df_SectorPrice
    
    def concatNSave_SectorPrice_files (self):
        '''regroupe tout les fichiers Sector price dans un seul, et sauvegarde le fichier dans 'self.path_latest' '''
        od_filepricesector = self.get_SectorPrice_TidyfilesNames ()
        first = True
        for path_sp in od_filepricesector.values ():
            if first:
                df_SectorPrice = self.read_fileSectorPrice (path_sp)
                ds_SectorPrice = TDataSet.TDataSet (df_SectorPrice)
                first = False
            else:
                df_SectorPrice2 = self.read_fileSectorPrice (path_sp)
                max_date = ds_SectorPrice.index[-1]
                df_SectorPrice2 = df_SectorPrice2.loc [df_SectorPrice2.index > max_date]
                if len (df_SectorPrice2.dropna (how='all')) > 0:
                     
                    Init.Log ( 'Appending: ' + path_sp, self.queue)
                    ds_SectorPrice2 = TDataSet.TDataSet (df_SectorPrice2)
                    ds_SectorPrice = ds_SectorPrice.patch_append (ds_SectorPrice2)
                else:
                    Init.Log ( 'problem in or old file: ' + path_sp, self.queue)
                    
        if len(od_filepricesector)>0:
            Init.Log ( 'Saving', self.queue)
            df_SectorPrice = pd.DataFrame (ds_SectorPrice)
            df_SectorPrice.to_csv (self.path_latest + 'SectorPrice.csv')
            
    def clean (self):
        '''vide la mémoire'''
        Init.Log ( 'Done', self.queue)
        # In[ ]:
        
        all_ = [var for var in globals() if var[0] != "_" and var[0]!='do_fromFtp']
        for var in all_:
            del globals()[var]
     
if __name__ == '__main__':
    inst = do_fromReuters ()
    inst.main ()