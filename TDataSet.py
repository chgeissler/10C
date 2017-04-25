# -*- coding: cp1252 -*-
import Init  # @UnresolvedImport
import numpy as np
import pandas as pd
import os
import shutil
import copy
from datetime import datetime
from datetime import timedelta
import math
import sys
import pdb
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
#import scipy
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

import scipy.fftpack

Init.openLogFile ()

glbNanoSexPerDay = 86400000000000.0
glbFieldSep = '!'
#pour les apprentissages de Juin 13
#glbFieldSep = '.'
#Paramètre de séparateur de champs fixé pour les dérivations
#dans ProOptimConfig.cfg, section [DerivXXX]
glbMetaVarPrefix = '$$'
glbDefaultTime = datetime (1900,1,1,18,30)

def to_idate (adate, dateformat = '%Y-%m-%d'):

    if type (adate) == str:
        dt = datetime.strptime (adate, dateformat)
    else:
        dt = adate
    try:
        res = dt.year * 10000 + dt.month*100 + dt.day
    except:
        res = 0

    return res
        
def find_delim(csv_path, delimiters=',;'):
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

def read_csv_clean (path, filename, delim = None, addextension = True, index_col = 0, \
                    header = 0, parse_dates = True, dayfirst = False, \
                    keep_date_col  = True, numeric_only = True, drop_dups = True, format_date = None):

    '''Lit un fichier .csv structuré et renvoie un DataFrame. Traite les lignes vides. '''

    def to_datetime (index, format_date = format_date):
        return datetime.strptime (index, format_date)
    #nom de fichier sans extension: on rajoute l'extension .csv par défaut
    if filename.find ('.csv') < 0 and addextension:
        filename = filename + '.csv'
    df = None
    
    try:
        #import pdb; pdb.set_trace ()
        with open (path + filename) as f:
            if np.isnan(header):
                header = 0
            
            delim = find_delim (path + filename)
            df = pd.read_csv (f, sep = delim, index_col = index_col,  header = header, \
                                parse_dates = parse_dates, 
                                keep_date_col = keep_date_col, 
                                dayfirst = dayfirst,
                                infer_datetime_format = True)

        f.close ()
        #pdb.set_trace()
        if (df is None):
            return None
        elif len (df.columns) == 0 :
            return None
        #détection de doublons dans les fichiers indexés par une date
        elif type(df.index[0]) in [datetime.date, pd.tslib.Timestamp]:  # @UndefinedVariable
            #Eliminer les doublons
            #pdb.set_trace()
            if drop_dups:
                df['__index'] = df.index
                df = df.loc [map (lambda x: not pd.isnull (x), df.index)]
                error_pandas_v_olderthan_014 = False
                try:
                    df.drop_duplicates (subset = ['__index'], take_last = True, inplace = True)
                except:
                    pass
                    error_pandas_v_olderthan_014 = True
                if error_pandas_v_olderthan_014:
                    df.drop_duplicates (cols = ['__index'], take_last = True, inplace = True)
                del df['__index']
            
            if len (df.index) >   1 + abs ((df.index [-1] - df.index[0]).days):
                Init.Log ("Le fichier {0} contient des doublons de dates".format (path + filename))
                return None            
        #col0 = df.columns [0]
        #ne retenir que les dates où l'une des colonnes est non vide
        #à rendre optionnel ?
        #vecteur logique égal à la présence d'au moins une donnée non vide
        idfcol = pd.DataFrame (index = df.index , columns = ['OK'])
        idfcol ['OK'] = False
        for col in df.columns:
            if all (pd.isnull (df [col])):
                del df [col]
            elif len (df [col]) > 0:
                idfcol ['OK'] = np.logical_or (idfcol ['OK'], pd.notnull (df[col]))
        #pdb.set_trace()
        #ne retenir que les données non vides dans au moins une colonne
        df = df [ idfcol ['OK']]
        if parse_dates and keep_date_col:
            df.index = pd.DatetimeIndex (df.index)
            df.index.name = 'Date'
            #affecte les bons types aux colonnes
            #pdb.set_trace()
            df = df.convert_objects (convert_numeric = True)
            pd.DataFrame.sort_index (df, ascending = True, inplace = True)
        #sélectionne les colonnes numériques uniquement si demandé             
        if numeric_only:
            cols = []
            for i in range (0, len (df.columns)):
                strtyp = str(df.dtypes[i])
                if strtyp.find ('int') == 0 or strtyp.find ('float') == 0:
                    cols.append (df.columns [i])
            if len (cols) < len (df.columns):
                df = df.ix [ : , cols]                  
        #f.close()
        return df
    
    except IOError as e:
        #pdb.set_trace()
        Init.Log ( "Accès en lecture impossible au fichier {0}: erreur {1}".format (path + filename, e.strerror))
        return None
    except:
        Init.Log ("Fichier {0} pas au format de matrice temporelle".format (path + filename))
        return None


def readFromDict (obj, odict, recursedict = False, queue = None):
    '''Charge un objet à partir d'un dictionnaire'''
    '''Si recursedict est True                   '''
    '''les clés du dictionnaire sont interprétées commes des champs de l'objet '''
    #pdb.set_trace()
    try:
        val = None
        key = None
        if 'name' in odict.keys():
            obj._name = odict['name']
        for key  in odict:
            #pdb.set_trace ()
            val = odict [key]
                    
            if type (val) in [float, int]:
                if np.isnan (val):
                    val = ''
            elif type(val) == str:
                #pdb.set_trace () 
                                        
                #if key == "Apprentissage" and val[0] =='[':
                    #Transformation d'un string en liste
                    #list_app = list()
                    #letter = ''
                    
                    #for a in val[1: len(val)]:
                        #if a == "," or a == ']':
                            #list_app = list_app + list([letter.replace(" ","")])
                            #letter = ''
                        #else:    
                            #letter = letter + a
                            
                    #val = list_app
                    #setattr (obj, '_' + key.lower(), val)
                                  
                if val[0] in ('[','{'):
                    #pdb.set_trace ()
                    val = zelist = eval (val)
                    if type (zelist) == dict:
                        #récursion: attribuer individuellement les clés du dico
                        #aux champs de l'objet
                        if '__fields' in zelist and zelist ['__fields'] == True:
                            for zekey in zelist:
                                zeval = zelist [zekey]
                                setattr (obj, '_' + zekey.lower(), zeval)
                        else:
                            #non récursion: attribuer le dico en tant que champ 
                            setattr (obj, '_' + key.lower(), val)
                            
            setattr (obj, '_' + key.lower(), val)

    except:
        Init.Log ('Objet {0}: impossible d''interpréter la valeur {1} du champ {2}.'
                                         .format (obj._name, val, key), doexit = True, queue  = queue)
        return None

def sumgeo (z, x):
    if z == 0.0:
        return x
    else:
        return (1 - z**x)/(1 - z)
    
    

 
class TDataSet (pd.DataFrame):
    '''DataFrame spécialisé pour la synchronisation de données temporelles. '''

#28/12/12 la dérivation de DataFrame n'apporte pas beaucoup de bénéfices, et quelques problèmes:
# l'appel à des méthodes issues de DataFrame renvoie un objet DataFrame et non TDataSet    
# les attributs et méthodes supplémentaires peuvent être rajoutés sans sous-classer

    def __init__(self, data = None, index = None, columns = None, \
                     dtype = None, copy = False, givenname = None, queue = None):
        pd.DataFrame.__init__(self, data = data, index = index, columns = columns, dtype = dtype, copy = copy)
        if givenname is None:
            if not hasattr (self, 'name'):
                self.name = 'DS' + str (datetime.now()).replace (':', '_')
        else:
            self.name  = givenname
        if not hasattr (self, 'inputpath'):
            self._inputpath = ''
        if not hasattr (self, 'outputpath'):
            self._outputpath = ''
        if not hasattr (self, '_properties'):
            self._properties = {}

        self.queue = queue
            
    def _as_TDataSet  (self, df):
        if type (df) in [pd.DataFrame, pd.Series, TDataSet]:
            self.__init__ (data = df.values, index = df.index, columns = df.columns)
            return self

    def copy(self):
        cp = copy.deepcopy (self)
        cp.name = self.name
        cp.inputpath = self.inputpath
        cp.outputpath = self.outputpath
        cp._properties = self._properties
##        cp.importstart  = self.importstart
##        cp.importend = self.importend
        return cp

    @property
    def defaultname (self):
        return 'DS' + str (datetime.now()).replace (':', '_')

    @property
    def name(self):
        '''Le nom de la matrice de données'''
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def inputpath(self):
        '''Le répertoire des séries d'origine'''
        return self._inputpath

    @inputpath.setter
    def inputpath(self, value):
        self._inputpath = value
        
    @property
    def outputpath(self):
        '''Le nom de la matrice de données'''
        return self._outputpath

    @outputpath.setter
    def outputpath(self, value):
        self._outputpath = value

    def mustRebuild (self, dstart, dend, col = None):
        ''' Détermine s'il faut éventuellement reconstruire le DataSet en totalité ou sur une colonne.'''
        if len (self.index) == 0:
            return True
        else:
            if col is None:
                return False
            elif col not in self.columns:
                    return True
            try:
                df = self [dstart : dend]
                if  np.datetime64 (dstart) < np.datetime64 (df.index [0]) or \
                    np.datetime64 (df.index [-1]) < np.datetime64 (dend):
                    return True
                else:
                #return any (pd.isnull (self[col].ix [dstart : dend]))
                    return False
            except:
                return True


    def set_prop (self, cols, prop, value):
        '''Rajoute une propriété à une ou des colonnes.'''
        if type (cols) == str:
            cols = [cols]
        elif type (cols) == int:
            cols = self.columns [cols]
        elif type (cols) == list:
            tabcols = []
            for col in cols:
                if type (col) == str:
                    tabcols = tabcols.append (col)
                elif type (col) == int:
                    tabcols = tabcols.append (self.columns [col])
                else:
                    pass
        else:
            return
        for col in cols:
            if col in self._properties:
                coldict = self._properties [col]
            else:
                coldict = {}
            coldict [prop] = value
            self._properties [col] = coldict
        return

    def get_prop (self, col, prop, default_value = None):
        '''Renvoie la valeur d'une propriété d'une colonne.'''
        if col in self._properties:
            coldict = self.properties [col]
            if prop in coldict:
                return coldict [prop]
            else:
                return default_value
        else:
            return default_value

    def get_columns (self, cols = None, forceuppercase = True):
        '''Renvoie des identifiants de colonnes pour un (vecteur de) int ou str. '''
        if cols is None:
            return self.columns
        else:
            retcolumns = []
            if type (cols) == str:
                retcolumns = [cols]
                return retcolumns
            else:
                #tester si l'argument passé est un nombre
                try:
                    col = int (cols)
                    try:
                        return [self.columns [col]]
                    except:
                        return []
                except:
                    pass
            #cols doit maintenant être un tableau
            try:
                len (cols)
            except:
                return []
            
            ncols = len (self.columns)
  
            for col in cols:
                if type(col) == int:
                    if col in range (0 , ncols):
                        retcolumns.append (self.columns[col])
                elif type(col) == str:
                    if forceuppercase: col = col.upper()
                    if col in self.columns:
                        retcolumns.append (col)
            return retcolumns

    def take_columns (self, cols = None, forceuppercase = True):
        '''Equivalent à l'opérateur []'''
        if cols is None: return self
        columns = self.get_columns (cols = cols, forceuppercase = forceuppercase)
        if len (columns) > 0:
            try:
                ds =  TDataSet (index = self.index, \
                            data =  self [columns], \
                            columns = columns)
                ds.name = self.name
            except:
                return None
        else:
            ds = None
        return ds
    
    def change_freq (self, freq = None):
        '''change la fréquence d'une série'''
        dstart = self.index [0]
        dend = self.index [1]
        if freq is not None:        
            _ = pd.bdate_range (start = dstart, end = dend, freq = freq)        
        pass

    def take_interval (self, dstart  = None, dend = None, inplace = False):
        ''' Prend une tranche temporelle [dstart, dend] '''
        ''' cas normal: type(dstart) == type (dend) == str'''
        #pdb.set_trace()
        if len (self.index) == 0:
            return self
        if dstart is None or dstart == '':
            dstart = self.index [0]
        else:
            dstart = pd.to_datetime(dstart)
            
        if dend is None or dend == '':
            dend = self.index [-1]
        else:
            dend = pd.to_datetime (dend)
        
        if (dstart <= self.index [0]) and (dend >= self.index [-1]):
            return self

        try:
            if inplace:
                #ds = self._as_TDataSet (self [str(dstart) : str(dend)])
                ds = self.loc [self.index <= str(dend)]
                ds = self.loc [self.index >= str(dstart)]
            else:
                ds = TDataSet (data = self [str(dstart) : str(dend)], \
                               givenname = self.name)
        except:
            ds = self
        return ds
    
    def exclude_interval (self, dstart, dend, fmt = '%d/%m/%Y'):
        if len (self.index) == 0:
            return self
        if type(dstart) == str:
            dstart = datetime.strptime(dstart, fmt)
        if type(dend) == str:
            dend = datetime.strptime (dend, fmt)
        ds = self
        try:
            datestokeep = self.index.map (lambda (x): (x < dstart) or (x > dend))
            ds = self._as_TDataSet (self [datestokeep])
        except:
            ds = self
        return ds
    
        
    def set_calendar (self, start, end, freq = 'B', forcereset = False):
        #création d'un index de dates
        rng = pd.bdate_range(start = start, end = end, freq = freq)
        try:
            self [start : end]
            if  forcereset or \
                np.datetime64 (start) < np.datetime64 (self.index [0]) or \
                np.datetime64 (end) > np.datetime64 (self.index [-1]):
                    
                TDataSet.__init__ (self, index = rng)
            else:
                return self
        except:
            TDataSet.__init__ (self, index = rng)

    def set_daytime (self, hm_time, dates = None):
        '''Fixe l'heure pour tout l'index ou pour des dates données '''
        #pdb.set_trace ()
        if type (hm_time) in [datetime, pd.tslib.Timestamp]:  # @UndefinedVariable
            if dates is None:
##                self.index = self.index.map ( \
##                    lambda (x): x + pd.DateOffset (hours = hm_time.hour - x.hour, \
##                                                              minutes = hm_time.minute - x.minute))
                self.index = self.index.map ( \
                    lambda (x): datetime (year = x.year, month = x.month, day = x.day, \
                                                    hour = hm_time.hour, \
                                                    minute = hm_time.minute))                
            else:
                #forcer les heures sur les dates selectionnees
                if type (dates) != list:
                    dates = [dates]
                for dt in dates:
                    self [dt : dt].index.values [0] = \
                         self [dt : dt].index [0] + \
                         pd.DateOffset (hours = hm_time.hour - self [dt : dt].index [0].hour, \
                                             minutes = hm_time.minute - self [dt : dt].index [0].minute)
        

    def add_series (self, series, seriesname = None, method = 'ffill', \
                         timecrit = None, compcrit = None, verbose = False):
        '''Rajoute un objet TimeSeries au TDataSet courant '''
        #pdb.set_trace ()
        if isinstance (series, (pd.TimeSeries, pd.Series)):
            #pdb.set_trace ()
            #Si le DataSet n'a pas encore d'index, c'est la première série qui va le fixer
            if self.index.size == 0:
                self.__init__(self, index = series.index)
            if series.index [0] < self.index [0] or \
                       series.index [-1] > self.index [-1]:
                        #series.take_interval (dstart = self.index [0], \
                        #                     dend = self.index [-1], inplace = True)
                        series = series [str(self.index [0]) : str(self.index [-1])]
            #réindexation de la série à ajouter selon l'index existant
            try:
                if method == 'auto':
                    mindeltas = np.minimum (series.index.asi8.diff(1)) / glbNanoSexPerDay
                    mindelta = np.minimum (self.index.asi8.diff(1)) / glbNanoSexPerDay
                    if mindeltas == mindelta:
                        method = None
                    else:
                        method = 'ffill'
                nseries = series.reindex (index = self.index, method = method)
            except:
                #pdb.set_trace()
                print 'Could not reindex series ' + series.name + 'into current Dataset.'
                return self
            #si un nom est fourni, il nomme la colonne contenant la série
            if not (seriesname is None) :
                sname = seriesname
            else :
                # pas de nom, série sans nom: dénomination automatique en V<n>
                if series.name == None:
                    ncols = self.columns.size
                    sname = 'V' + str(ncols)
                else:
                    sname = series.name
            
            # ajout de la série avec le nom
            timeOK = True
            compOK = True
            ds = TDataSet (index = nseries.index, data = nseries.values)
            #pdb.set_trace()
            if timecrit is not None:
                timeOK =  ds.meet_time_criteria (timecrit)
            if compcrit is not None:
                compOK = ds.meet_compression_criteria (compcrit)
            if timeOK and compOK:
                self [sname] = nseries
                if verbose:
                    Init.Log ('TDataSet {0}: rajouté la série {1}.'.
                              format(self.name, seriesname), queue = self.queue)
            elif not timeOK:
                if verbose:
                    Init.Log  ('TDataSet {0}: rejeté la série {1} (critère de temps non vérifié.)'.
                               format (self.name, seriesname), queue = self.queue)
            else:
                if verbose:
                    Init.Log  ('TDataSet {0}: rejeté la série {1} (critère de compression non vérifié.)'.
                               format (seriesname), queue = self.queue)
            return self

    def add_dataset (self, newds, fillmethod = 'ffill', value = None,\
                    timecrit = None, compcrit = None,
                    crits = None, verbose = False,
                    fieldsep = '', prefixcolumns = True):
        '''Combine un nouveau TDataSet à l'objet courant'''
        '''
        newds: DataSet à combiner
        fillmethod: méthode de remplissage à appliquer
        value: valeur fixe remplaçant les valeurs manquantes
        compcrit: dictionnaire de contraintes de compression valable pour toutes les colonnes
        tcrit: dictionnaire de contraines de temps valables pour toutes les contraintes
        crits: dictionnaire d'autres contraintes
        prefixcolumns: transforme les noms de colonnes du nouveau dataset en les préfixant par le nom du dataset
        fieldsep: caractère pour séparer le nom de colonne de son préfixe
        '''

        def reduce_column_name (col, sep):
            colparts = col.split (sep)
            if len (colparts) <= 2:
                return col
            else:
                return colparts [0] + sep + colparts [-1]
            
        if isinstance (newds, (pd.TimeSeries, pd.Series, pd.DataFrame)):
            #pdb.set_trace()
            if fieldsep == '':
                fieldsep = glbFieldSep
            #renommer les colonnes
            ncols = len(newds.columns)
            #pdb.set_trace()
            #Une colonne seulement: on lui attribue le nom du dataset
            if ncols == 1:
                newds.columns = [newds.name.upper()]
            else:
                tabcols = []
                #pdb.set_trace()
                #tabcols = self.columns
                prefix = (newds.name + fieldsep).upper()
                for col in newds.columns:
                    #les colonnes sont préfixées par le nom du dataset suivi du séparateur 
                    #si la colonne contient déjà le nom du dataset suivi du séparateur,
                    #ou si la colonne est une métavariable,
                    #on ne préfixe pas: le nom de colonne n'est pas modifié
                    col = col.upper ()
                    if prefixcolumns:
                        if col.find (prefix) >= 0 or \
                            col.find (glbMetaVarPrefix) >= 0:
                            newcol = col
                        else:
                            newcol = prefix + col
                    else:
                        newcol = col
                    #on retient le dernier segment comme suffixe
                    newcol = reduce_column_name (newcol, fieldsep)
                    #si le nom de colonne ainsi formé est déjà présent:
                    if (newcol in tabcols) or (newcol in self.columns):
                        #pdb.set_trace ()
                        newcol = prefix + col
                    tabcols.append (newcol)
                newds.columns = tabcols

            #pdb.set_trace ()
            # Vérification des critères

            if len (self.index) == 0:
                self = newds.copy ()
                for col in self.columns:
                    #self [col].delete ()
                    del self [col]
            else:    
                # df = self.join (newds,  how = 'outer')
                # pdb.set_trace ()
                if newds.index [0] < self.index [0] or \
                   newds.index [-1] > self.index [-1]:
                    newdend = newds.index [-1]
                    selfend = self.index [-1]
                    if newdend.day == selfend.day and \
                       newdend.month == selfend.month and \
                       newdend.year == selfend.year:
                        
                        newdend = newdend + pd.Timedelta (hours = selfend.hour - newdend.hour,
                                                          minutes = selfend.minute - newdend.minute,
                                                          seconds = selfend.second - newdend.second)
                        
                    newds.take_interval (dstart = self.index [0], \
                                         dend = newdend, inplace=True)
                   
            try:
                            
                if fillmethod == 'auto':
                    mindeltas = newds.min_day_interval ()
                    mindelta = self.min_day_interval()
                    # Réindexation dans une série existante: on compare les fréquences
                    
                    if mindeltas == mindelta:
                        # Si la nouvelle série a la même fréquence que la série hôte
                        if pd.isnull(value):
                            # pas de valeur de remplissage: ajuster les index en ne remontant pas au delà de la fréquence native
                            # Commencer par être sévère pour empêcher le remplissage de vraies valeurs manquantes
                            #CG 26 Oct 15
                            #mindelta au lieu de mindelta - 1 
                            newds2 = newds.reindex (index=self.index, method='ffill', limit=mindelta)
                            newds2[newds2.index.date > newds.index [-1].date()] = np.nan
                            #newds2[newds2.index > newds.index [-1]] = np.nan
                            #CG 8 Dec 15
                            #empêcher le prolongement de la série au-delà de la série d'origine
                            # si on perd trop de données, cas d'une série japonaise 9h30: aller chercher les données du matin
                            if len(newds2.dropna()) < len (newds.index) / 2.0:
                                newds2 = newds.reindex (index=self.index, method='ffill')    
                        else:

                            newds2 = newds.reindex (index=self.index, fill_value=value)
                    else:
                        if pd.isnull(value):
                            newds2 = newds.reindex (index=self.index, method='ffill')
                        else:
                            newds2 = newds.reindex (index=self.index, fill_value=value)
                elif pd.isnull (value):
                    newds2 = newds.reindex (index=self.index, method=fillmethod)
                else:
                    newds2 = newds.reindex (index=self.index, fill_value=value)
                if len (newds2.dropna()) == 0:
                    pass
            except:
                nl = len(newds.index)
                nul = len(np.unique (newds.index.values))
                Init.Log ('L\'index de {0} est de longueur {1} et a {2} doublons'\
                          .format (newds.name, nl, nl - nul),
                          doexit=True)
                return self
                    
            for col in newds.columns:
                     
                    timeOK = True
                    compOK = True
                    critsOK = True
                    doadd = True

                    #si un critère de filtre est demandé:
                    #evaluer les critères par colonne
                    if (timecrit is not None) or (compcrit is not None):
                        #ds = TDataSet (index = nseries.index, data = nseries.values)
                        if timecrit is not None:
                            timecrit ['col'] = col
                            timeOK =  newds.meet_time_criteria (timecrit)
                        if compcrit is not None:
                            compcrit ['col'] = col
                            compOK = newds.meet_compression_criteria (compcrit)
                        if not timeOK:
                            if verbose:
                                Init.Log  ('Rejected {0}[{1}] (Time)'.format (newds.name, col))
                                doadd = False
                        if not compOK:
                            if verbose:
                                Init.Log  ('Rejected {0}[{1}] (compression)'.format (newds.name, col))
                                doadd = False 
                        
                    if crits is not None:
                        for crit in crits.keys ():
                            if crit == 'adf':
                                pvalue = crits [crit]
                                res_pvalue = newds.do_adftest (col=col)
                                if res_pvalue > pvalue:
                                    critsOK = False   
                                if not critsOK:
                                    if verbose:
                                        Init.Log  ('Rejected {0}[{1}] (Adf)'.format (newds.name, col))
                                        doadd = False            
                    
                    if doadd:
                        self [col] = newds2 [col]
                        if verbose:
                            Init.Log ('Added  {0} [{1}]'.format(newds.name, col))

            return self
    

    def add_series_from_file (self, filename, path = None, delim = ',',
                                    header = 0, dayfirst = False,
                                    method = 'ffill', verbose = False,
                                    addextension = True, value = None, 
                                    givenname = None, cols = None,
                                    timecrit = None, compcrit = None,
                                    crits = None,
                                    fieldsep = '', prefixcolumns = True,
                                    numeric_only = True, format_date = None):
        '''Crée un TDataSet à partir d'un fichier et le rajoute aux données courantes. '''
        #pas de chemin fourni: on prend le chemin input par défaut
        if path is None:
            path = self.inputpath
        #pdb.set_trace ()
        if fieldsep == '':
            fieldsep = glbFieldSep
        lastdate = pd.datetime.date (pd.datetime (1900,1,1))
        df = read_csv_clean (path = path, filename = filename,
                                     delim = delim, addextension = addextension,
                                     header = header, parse_dates = True, keep_date_col = True,
                                     dayfirst = dayfirst, numeric_only = numeric_only, format_date = format_date)
        
        if (df is None): return self, lastdate
        elif len (df.index) == 0:
            return self, lastdate
        else:
            lastdate = pd.datetime.date (max (df.index))
        
        newds = TDataSet (data = df)
        #nom de la série: soit le nom passé en argument, soit le nom du fichier
        if givenname is None:
            newds.name = filename
        else:
            newds.name = givenname
        
        #Enlever le chemin et l'extension
        newds.name = newds.name.split ('/') [-1].split('.')[0]
        #pdb.set_trace()
        self = self.add_dataset (newds, fillmethod = method, value = value,
                                         verbose = verbose, crits = crits,
                                         timecrit = timecrit, compcrit = compcrit,
                                         fieldsep = fieldsep, prefixcolumns = prefixcolumns)
        return self, lastdate

    def add_series_from_fileslist (self, filelist, path = None,  
                                    exclusionlist = ['confli'],
                                    method = 'ffill',
                                    value = None, delim = ',', 
                                    namestart = None, nameend = None,
                                    verbose = True, addextension = True,
                                    timecrit = None, compcrit = None,
                                    crits = None, fieldsep = '',
                                    prefixcolumns = True, numeric_only = True):
        '''Rajoute des séries lues à partir d'un ensemble de noms de fichiers du même répertoire.'''
        '''Filtre les noms de fichiers dans un intervalle alphabétique. '''
        def isexcluded (fname, exclusionlist):
            #Détermine si un fichier contient une chaîne interdite
            if exclusionlist is None:
                return False
            if type(exclusionlist) == str:
                exclusionlist = [exclusionlist]
            if len(exclusionlist) > 0:
                for item in exclusionlist:
                    if fname.lower().find(str(item).lower()) > 0:
                        return True
            return False
                
            
        if path is None: path = self.inputpath
        if fieldsep == '':
            fieldsep = glbFieldSep
        lastdate = pd.datetime.date (pd.datetime (1900,1,1))
        for f in filelist:
            if (namestart is None or str(f).upper() >= str(namestart).upper()) and \
               (nameend is None or str(f).upper() <= str(nameend).upper()):

                #pdb.set_trace()
                if not os.path.isdir(path + f) and not isexcluded (f, exclusionlist):
                    try:
                        self, flastdate = self.add_series_from_file (path = path, filename = f,
                                                        timecrit = timecrit, compcrit = compcrit,
                                                        crits = crits,
                                                        method = method, verbose = verbose,
                                                        addextension = addextension, \
                                                        value = value, delim = delim,
                                                        fieldsep = fieldsep, prefixcolumns = prefixcolumns,
                                                        numeric_only = numeric_only)

                    except:
                        #pdb.set_trace ()
                        self, flastdate = self.add_series_from_file (path = path, filename = f,
                                                        timecrit = timecrit, compcrit = compcrit, 
                                                        crits = crits,
                                                        method = method, verbose = verbose,
                                                        addextension = addextension, \
                                                        value = value, delim = delim,
                                                        fieldsep = fieldsep, prefixcolumns = prefixcolumns,
                                                        numeric_only = numeric_only)
                        Init.Log ('Erreur au moment de l''ajout du fichier {0}'.format (f), queue = self.queue)
                        pass
                    
                    if flastdate > lastdate:
                            lastdate = flastdate
        #pdb.set_trace()           
        return self, lastdate

    def add_series_from_dir (self, path, exclusionlist = ['confli'],
                                    timecrit = None, compcrit = None,
                                    crits = None, addextension = True,
                                     namestart = None, nameend = None, verbose = True,
                                     method = 'ffill', value = None,
                                     fieldsep = '', prefixcolumns = True,
                                     numeric_only = True):
        '''Charge tous les fichiers texte d'un répertoire donné '''
        #pdb.set_trace()
        Init.Log ("Ajout à la matrice {0} des séries du répertoire {1}.".format (self.name, path), queue = self.queue)
        if fieldsep == '':
            fieldsep = glbFieldSep

        self, lastdate = self.add_series_from_fileslist (filelist = os.listdir(path), path = path,
                                                        exclusionlist = ['confli'],
                                                        addextension = addextension,
                                                        namestart = namestart, nameend = nameend, 
                                                        verbose = verbose, timecrit = timecrit,
                                                        compcrit = compcrit, crits = crits,
                                                        method = method, value = value,
                                                        fieldsep = fieldsep, prefixcolumns = prefixcolumns,
                                                        numeric_only = numeric_only)
 
        return self, lastdate

    def to_csv (self, outpath = None, filename = None, float_format = None, sep = ',',
                qfstyle = False, append = False, archive = False, queue = None):
        '''Ecrit les données dans un fichier texte. Surcharge la méthode DataFrame.to_csv. '''
        if not os.path.isdir(outpath):
            os.makedirs(outpath, mode = 0777) 
              
        if filename is None:
            name = self.name + '.csv'
        else:
            if filename.find ('.csv') < 0:
                name = filename + '.csv'
            else:
                name = filename
        
        if outpath is None: outpath = self.outputpath

        exists = os.path.exists (outpath + name)
        #compter le nombre de variables (hors index)
        ncol = len (self.columns)
        try:
            if append:
                mode = 'ab'
            else:
                mode = 'wb'
            with open (outpath + name, mode) as f:
                #Ecriture de trois lignes d'en tete
                if qfstyle:
                    #CG 4/11/15 pour une production QF, délimiteur forcé
                    sep = ';'
                    header1 = [''] + ['C'] * ncol
                    header2 = [''] * (ncol + 1)
#                     for _ in range (0, ncol + 1):
#                         header1.append ('C')
#                         header2.append ('')
                    #pdb.set_trace()
                    #str1 = (str (header1) [1:-1]).replace (',', sep).replace("'",'') + '\n'
                    str1 = sep.join (header1) + '\n'
                    #str2 = (str (header2) [1:-1]).replace (',', sep).replace("'",'') + '\n'
                    str2 = sep.join (header2) + '\n'
                    
                    f.write (str1)
                    f.write (str2)
                    f.write (str2)
                  
                pd.DataFrame.to_csv(self, path_or_buf = f, \
                                    float_format = float_format, sep = sep)
                f.close()


                
        except IOError as e:
            Init.Log ( "Accès en écriture impossible au fichier {0}: erreur {1}". \
                       format (outpath + name, e.strerror), queue = self.queue)
            return None
        except:
            Init.Log ( "Erreur {1} imprévue lors de l'écriture dans le fichier au fichier {0}.". \
                       format (outpath + name, sys.exc_info()[0]), queue = self.queue)
            return None
        
        if exists and archive:
            # pdb.set_trace ()
            dtnow = datetime.now ()
            dstr = datetime.strftime(dtnow, '%Y-%m-%d %H-%M')
            datedname = name.split('.csv')[0] + '(' + dstr + ')' + '.csv'
            try:
                shutil.copy (outpath + name, outpath + datedname)
            except IOError as e:
                Init.Log ( "Accès en écriture impossible au fichier {0}: erreur {1}". \
                       format (outpath + datedname, e.strerror), queue = self.queue)
            return None
        
        Init.Log ("Ecriture réussie du fichier {0}".format (outpath + name), queue = queue)
        return self

    def from_csv (self, inpath = None, filename = None, index_col = 0, delim = ',', header = 0):
        '''Lit les données d'un DataSet depuis un fichier texte.  '''

        #pdb.set_trace()    
        if filename is None:
            name = self.name + '.csv'
        else:
            name = filename.split('.csv')[0] + '.csv'
        if inpath is None: inpath = self.inputpath
        df = read_csv_clean (path = inpath, filename = filename, \
                            index_col = index_col, delim = delim, header = header)
        if df is None:
            return None
        else:
            self._as_TDataSet (df)
            self.name = name
        return self
    
    def get_freq (self):
        '''retourne la frequence d'une série'''
        ds = self.copy()
        ds ['freq'] = ds.index
        ds ['freq'] = ds['freq'].diff().astype('timedelta64[D]')
        #On prend la fréquence majoritaire
        if self.name.upper () in self.columns:
            col = self.name.upper ()
        else:
            col = self.columns [0]
        ds_freq = ds.groupby ('freq').count() [col]
        max_ = max (ds.groupby ('freq').count() [col])
        freq = int (ds_freq [ds_freq == max_].index [0])
        #freq = int (freq / np.timedelta64 (1, 'D'))
        
        return freq
    
    def patch_append (self, newdata, override_old = False, override_depth = 0, inplace = True, check_overlap = True):
        '''Recolle un nouveau morceau de données sur un historique antérieur.'''
        '''newdata: ensemble de données ayant des colonnes isomorphes à self.columns
          overrideold: booléen indiquant si les données anciennes sont à effacer dans la plage de chevauchement des dates.
          overridedepth: décalage par rapport à la date la plus récente des nouvelles données pour le remplacement forcé.
          inplace: concaténation sur place ou dans un nouvel objet.
          checkoverlap: si Vrai, déclenche une erreur en cas de non-chevauchement des données (éviter les trous).'''
        #pdb.set_trace()
        #il faut trier les données par dates croissantes
        self.sort_index (inplace = True)
        if len(newdata.index) == 0:
            Init.Log ("Attention, les nouvelles données de: " + self.name + " sont vides.")
            return self    
        if len(newdata.columns) == 0:
            Init.Log ("Attention, les nouvelles données de: " + self.name + " n''ont pas de colonnes.")
            return self            
        if newdata.index [-1] <= self.index [-1]:
            Init.Log ("Attention, les nouvelles données sont déjà dans l'historique de: " + self.name)
            return self
        #la freq de la serie
        try:
            freq = self.get_freq ()
        except:
            freq = 1
        
        if (type(newdata) != type (self)):
            Init.Log ('patch_append: l''objet passé en argument n''est pas un TDataSet.', queue = self.queue)
            return self
        newdata.sort_index (inplace = True)
        latest_old_date = self.index [-1]
        override_depth = int (override_depth)
        override_depth_ = min (len (newdata.index) - 1, override_depth)
        earliest_new_date = newdata.index [0]
        #la première date de remplacement
        if override_old:
            first_replacement_date = self.index [-1 - override_depth] + timedelta (days = 1)
        else:
            first_replacement_date = min (latest_old_date + timedelta(days = 1),
                                                    newdata.index [-1 - override_depth_] )
        #latest_new_date = newdata.index [-1]
        #dif = (earliest_new_date - latest_old_date) / np.timedelta64 (1, 'D')
        dif = (earliest_new_date - latest_old_date).days
        dif = int (dif)
        #overlap = (earliest_new_date <= latest_old_date)
        if dif >= 0:
            try:
                np.testing.assert_approx_equal (freq, dif, 1)
                overlap = True
            except:
                overlap = False
        else:
            overlap = True
            
        dopatch = True
        ds = None
        if overlap:
            #on réduit l'historique en l'arrêtant un jour avant la première date de remplacement
            #pdb.set_trace()   
            #dtrepl = first_replacement_date + timedelta (days = -1)
            #history = self.take_interval (dend = dtrepl, inplace = inplace)
            history = self.take_interval (dend = first_replacement_date - timedelta (days = 1))
            patch = newdata.take_interval (dstart = first_replacement_date)
            dopatch = True
        else:
            #pas de chevauchement
            if check_overlap:
                Init.Log ('Risque de trou lors de la completion de l''historique de {0}\
                             a partir de la date {1}'.format (self.name, latest_old_date)) #, queue = self.queue)
                #raise ()
            else:
                dopatch = True
            if inplace:
                history = self
            else:
                history = self.copy()
            patch = newdata
        try:
            if dopatch:
                ds = history.append (patch)
            else:
                ds = history
        except:
            Init.Log ('Problème lors du recollement des matrices {0} et {1}.' \
                      .format (self.name, newdata.name))
            return None
        ds.name = self.name
        return ds
    
    def extendToDate (self, todate = None, freq = 'B', limit = 5):
        '''Etend un dataset jusqu'à une date plus récente en prolongeant les valeurs. '''
        if todate is None:
            todate = datetime.now ()
        else:
            todate = pd.to_datetime (todate)
        dt0 = self.index [-1] + timedelta (days = 1)    
        if dt0 <= todate:
            #dt0 = self.index [-1] + timedelta (days = 1)
            dtindex = pd.bdate_range (start = dt0, end = todate, freq = freq)
            newds = TDataSet (index = dtindex, columns = self.columns)
            for col in self.columns:
                newds.ix [0, col] = self.ix [-1, col]
                newds [col] = newds [col].fillna (method = 'ffill', limit = limit)
            self = TDataSet (self.append (newds))
            
        return self
        
    def getDeltaT (self, period = 1):
        #Renvoie la série des intervalles en nombre de jours
        deltatds = TDataSet (index = self.index, columns = ['Dt'],
                                     data = (self.index.asi8 / glbNanoSexPerDay ))
        deltatds = deltatds.diff (period)
        return deltatds
    
    def time_since_last_event (self, col, eventsign = 0, thresh = 0.0):
        '''Renvoie une série temporelle de changements. '''
        coldf  = self.take_columns(col)
        if coldf is None: return None
        thresh = abs (thresh)
        if thresh == 0.0:
            coldf ['_roundval'] = coldf [coldf.columns [0]]
        else:
            coldf ['_roundval'] = np.rint (coldf [coldf.columns [0]] / thresh) * thresh
       
        ddf = coldf ['_roundval'].diff (1)
        if eventsign == 0:
            chgval = coldf [ddf <> 0.0]
        elif eventsign == 1:
            chgval = coldf [ddf > 0.0]
        elif eventsign == -1:
            chgval = coldf [ddf < 0.0]
        else:
            pass
        
        changedf = coldf.ix [chgval.index]
        changedf ['deltat'] = 0
        changedf ['deltat'].iloc [1:] = (changedf.index.asi8 [1 : ] - changedf.index.asi8 [ : -1]) / glbNanoSexPerDay          
        return changedf
    
    def apply_hysteresis (self, col, thresh = 0.0, nbdays = 1, inplace = True):
        '''Modifie une série temporelle de changements. '''
        
        if col not in self.columns: 
            return None
        
        if inplace:
            newdf = self
        else:
            newdf = self.copy()

        thresh = abs (thresh)
        if thresh == 0.0:
            newdf ['_roundval'] = newdf [col]
        else:
            newdf ['_roundval'] = np.rint (newdf [col] / thresh) * thresh
            
        chgval = newdf.take_columns([col, '_roundval'], forceuppercase = False)
        
        newdf ['idate'] = range(0, len(self.index))

        chgval ['chgidx'] = range (0, len (newdf.index))
        chgval = chgval [abs(chgval['_roundval'].diff(1)) >= thresh]
        chgval = chgval.reindex (index = newdf.index, method = 'ffill')
        newdf ['chgidx'] = chgval ['chgidx']
        newdf ['deltaidx'] = (newdf ['idate'] - newdf ['chgidx'])
        newdf ['newval'] = chgval [col]
        newdf ['targetval'] = np.nan
        newdf ['targetval'].iloc [0] = newdf ['_roundval'].iloc [0]
        subidx = newdf [np.logical_and (newdf.deltaidx >= nbdays, 
                                        newdf.deltaidx <= nbdays+1)].index
        #on récupère la première valeur arrondie non na 
        #dès que le nombre de jours dépasse le seuil
        newdf.loc [subidx, 'targetval'] = newdf.loc [subidx, '_roundval']
        newdf ['targetval'].fillna (method = 'ffill', inplace = True)
        return newdf
    
    def min_day_interval (self):
        deltadays = (self.index.asi8 [1 : ] - self.index.asi8 [ : -1]) / glbNanoSexPerDay
        return np.rint(deltadays).min()
    
    def max_hole_date_size (self, col = None):
        '''Renvoie les dates et la taille du plus grand segment vide.'''
        if col is None:
            zecol = [self.columns [0]]
        else:
            zecol = self.get_columns(col)
        
        try:
            df = self.dropna (axis = 'index', subset = zecol, inplace = False)
            deltadays = (df.index.asi8 [1 : ] - df.index.asi8 [ : -1]) / glbNanoSexPerDay
            imax = deltadays.argmax ()
            return max(deltadays), df.index [imax], df.index [imax + 1]
        except:
            return np.nan, np.nan, np.nan
                
    
    def estimate_nat_freq (self, col):
        '''Estime la fréquence naturelle d'une série: la fréquence des changements de valeur '''
        self.dropna()
        self.sort_index (inplace = True)
        fl = float ((self.index.asi8 [-1] - self.index.asi8 [0]) / glbNanoSexPerDay)
        #fl = float(len (self.values))
        try:
            if type (col) == int:
                coldf = self [self.columns [col]]
            elif type (col ) == str:
                coldf = self [col]
            else:
                return {}
        except:
            return {}

        #série des différences
        ddf = coldf.diff(1)
        #série des différences non nulles  
        ddf = self [ddf != 0]
        #rajouter une colonne pour les différences de dates
        ddf ['deltat'] = 0
        ddf.deltat [1:] = (ddf.index.asi8 [1 : ] - ddf.index.asi8 [ : -1]) / glbNanoSexPerDay
        mind = 0; 
        #trier les intervalles entre changements de dates
        lastdelta = ddf.ix [-1]
        ddf.sort (columns = 'deltat', inplace = True)
        l = len(ddf)
        deltat = ddf.deltat [1:]
        fdict = {}
        #pdb.set_trace()
        if l > 1:
            fdict ['last'] = lastdelta
            fdict ['min'] = mind = deltat.min()
            fdict ['datemin'] = deltat.idxmin ()
            fdict ['pct5'] = mind
            fdict ['pct10'] = mind
            fdict ['pct25'] = mind
            fdict ['median'] = deltat.ix [int (0.5 * l) - 1]
            fdict ['max'] = maxd = deltat.max()
            fdict ['datemax'] = deltat.idxmax ()
            fdict ['pct95'] = maxd
            fdict ['pct90'] = maxd
            fdict ['pct75'] = maxd
            fdict ['n1'] = len (deltat [deltat >= 1])
            fdict ['r1'] = fdict ['n1'] / fl            
            fdict ['n5'] = len (deltat [deltat >= 5])
            fdict ['r5'] = fdict ['n5'] / (fl / 5)
            fdict ['n10'] = len (deltat [deltat >= 10])
            fdict ['r10'] = fdict ['n10'] / (fl / 10)
            fdict ['n20'] = len (deltat [deltat >= 20])
            fdict ['r20'] = fdict ['n20'] / (fl / 20)                       
            if l > 4:
                fdict ['pct25']  = deltat.ix [int (0.25 * l) - 1]
                fdict ['pct75'] = deltat.ix [int (0.75 * l) - 1]                    
                if l > 10:
                    fdict ['pct10'] = deltat.ix [int (0.1 * l) - 1]
                    fdict ['pct90'] = deltat.ix [int (0.9 * l) - 1]
                    if l > 20:
                        fdict ['pct5'] =  deltat.ix [int (0.05 * l) - 1]
                        fdict ['pct95'] = deltat.ix [int (0.95 * l) - 1]
        return fdict
    
    def extract_changes (self, col = None):
        ds = self.take_columns (col)
        if ds is None:
            return None
        #série des différences non nulles  
        dds = ds [ds.diff(1) != 0]
        return TDataSet (data = dds)

    def compression_rate (self, col = None, freq = 'B'):
        '''Calcule le taux de compression par rapport à une fréquence donnée'''
        try:
            if type (col) == int:
                coldf = self [self.columns [col]]
            elif type (col ) == str:
                coldf = self [col]
            else:
                return 0
        except:
            return 0
        #construction de l'index de dates pour la fréquence donnée
        didx = pd.bdate_range (self.index[0], self.index[-1], freq = freq)
        if len (didx) > 0:
            return float ( coldf.nunique() ) / len (didx)
        else:
            return 1
  
    def meet_time_criteria (self, timecrit = None):
        '''Détermine si une colonne d'un DF vérifie des critères de temps propre donné par un dictionnaire '''
        #pdb.set_trace()
        if timecrit is None:
            return True
        col = 0
        if 'col' in timecrit:
            col = timecrit ['col']
        dstart = self.index [0]
        if 'periodstart' in timecrit:
            dstart = timecrit ['periodstart']
        dend = str (self.index [-1])
        if 'periodend' in timecrit:
            dend = timecrit ['periodend']
        #récupérer la tranche temporelle
        ds = TDataSet (data = self.ix [str(dstart) : str(dend)])    
        tdict = ds.intrinsic_time (col)
        bsatisfied = True
        for crit in timecrit:
            if crit in tdict:
                fminval = -1000000000
                fmaxval = 1000000000
                try:
                    fminval = float (timecrit [crit][0])
                    fmaxval = float (timecrit [crit][1])
                except:
                    pass
                fcritval = tdict [crit]
                bsatisfied = bsatisfied and (fminval <= fcritval) and (fcritval <= fmaxval)
        return bsatisfied       

    def do_adftest (self, dstart=None, dend=None, maxlag=1, regression='nc', col=None):
        '''
        maxlag : int
        Maximum lag which is included in test, default 12*(nobs/100)^{1/4}
        regression : str {'c','ct','ctt','nc'}
        Constant and trend order to include in regression
        * 'c' : constant only (default)
        * 'ct' : constant and trend
        * 'ctt' : constant, and linear and quadratic trend
        * 'nc' : no constant, no trend
        
        -----
        return pvalue
        '''
        if dstart is None:
            dstart = self.index [0]
        if dend is None:
            dend = self.index [-1]
            
        ds = TDataSet (data = self.ix [str(dstart) : str(dend)]).dropna()
        if col is None:
            if self.name in ds.columns:
                col = self.name
            else:
                col = ds.columns[0]
        subserie = ds [col]
        res_pvalue = ts.adfuller (subserie, maxlag=maxlag, regression=regression)[1]
        return res_pvalue
        
    def meet_compression_criteria (self, compcrit = None):
        '''Détermine si une colonne d'un DF vérifie des critères de compression donné par un dictionnaire '''
        #pdb.set_trace()
        if compcrit is None:
            return True
        col = 0
        if 'col' in compcrit:
            col = compcrit ['col']
        dstart = self.index [0]
        if 'periodstart' in compcrit:
            dstart = compcrit ['periodstart']
        dend = self.index [-1]
        if 'periodend' in compcrit:
            dend = compcrit ['periodend']
        #récupérer la tranche temporelle
        ds = TDataSet (data = self.ix [str(dstart) : str(dend)])               
        cprate = ds.compression_rate (col)
        bsatisfied = True
        fminval = -1000000000
        fmaxval = 1000000000
        try:
            fminval = float (compcrit ['cprate'][0])
            fmaxval = float (compcrit ['cprate'][1])
        except:
            pass
        bsatisfied = bsatisfied and (fminval <= cprate) and (cprate<= fmaxval)
        return bsatisfied
    
    def intrinsic_time (self, col):
        '''Donne une statistique sur les changements de position par rapport à la valeur médiane.

        '''
        #en réalité une méthode monocolonne, donc à rattacher à TimeSeries
        #enlever les lignes des valeurs manquantes si nécessaire
        self.dropna ()
        #pdb.set_trace()
        self.sort_index (inplace = True)
        try:
            if type (col) == int:
                coldf = self [self.columns [col]]
            elif type (col ) == str:
                coldf = self [col]
            else:
                return {}
        except:
            return {}        
        #pdb.set_trace()
        #calculer la valeur médiane
        medv = coldf.median()
        sgnfunc = lambda(x): (np.sign(x - medv))
        #cast en DataFrame, sinon TimeSeries ne répond pas
        positiondf = TDataSet (index = self.index, data = coldf.apply (sgnfunc))
        return positiondf.estimate_nat_freq (0)

    def apply_timeshift (self, shift = 1, freq = 'B', ownfreq = None, refdate = None):
        '''Renvoie une copie de l'objet courant, avec dates translatées d'un délai. '''
        '''Les noms de colonnes de l'objet courant ne sont pas modifiés.'''
        '''freq représente l''unité de compte du décalage'''
        '''ownfreq représente la fréquence finale (propre) de la série.'''
        '''refdate: date de calcul. si fournie, les décalages sont limités à cette date'''
        '''Exemple: décaler de 20j une série trimestrielle'''

        #pdb.set_trace ()
        newdataset = self.copy()

        #pas de décalage: on peut changer la fréquence
        #if freq <> 'B':
        if ownfreq is not None and ownfreq != freq:
            pass
            #newdataset = newdataset.change_freq (freq = ownfreq)
        if shift == 0:
            return newdataset
        
        if refdate is None:
            refdate = datetime.now()
        else:
            refdate = pd.to_datetime (refdate)
            
#             else:
#                 return newdataset
        
        #Attention: tshift renvoie un DataFrame
        #import pdb; pdb.set_trace()
        ndf = newdataset.tshift (shift, freq)

        #sous-typer en TDataSet
        #Vérifier que l'on ne décale pas au-delà d'aujourd'hui
        lastdate = ndf.index [-1]
        if lastdate > refdate:
            lastline = ndf.ix [-1]
            ndf = ndf [ndf.index < refdate]
            newline = pd.DataFrame (index = [refdate], data = lastline.values, columns = self.columns)
#             for col in newline.columns:
#                 newline.ix [refdate, col] = lastline [0] [col]
            ndf = ndf.append (newline)
            
        newdataset._as_TDataSet (ndf)
        
        if self._name is None:
            self.name = self.defaultname
        if shift > 0:
            newdataset.name = self.name + '_t+' + str(shift) + str(freq)
        else:
            newdataset.name = self.name + '_t-' + str(abs(shift)) + str(freq)
        return newdataset

    def apply_lag (self, lag = 1, freq = None, inplace = False, cols = None):
        '''Renvoie une copie de l'objet courant, avec valeurs décalées d'un retard et réalignées. '''
        '''Les noms de colonnes de l'objet courant ne sont pas modifiés.'''
        if inplace:
            newdataset = self
        else:
            newdataset = None

        if lag == 0:
            return newdataset

        cols = self.get_columns(cols)
        #if not (cols in self.columns) : return newdataset
        dfcols = pd.DataFrame (data = self [cols], columns = cols)
        #la série décalée
        #import pdb; pdb.set_trace()
        laggedseries = dfcols.shift (periods = lag, freq = freq, inplace = inplace)
        #nommer le dataset résultant s'il est différent
        if lag > 0:
            laggedseriesname = self.name + '>>' + str(lag) + str(freq)
        else:
            laggedseriesname = self.name + '<<' + str(abs(lag)) + str(freq)

        # en cas de copie: on renvoie un dataset contenant les colonne décalées
        if inplace:
            for col in cols:
                newdataset [col] = laggedseries [col]
        else:
            newdataset = TDataSet (index = laggedseries.index, \
                                   data = laggedseries.values,
                                   columns = cols)
            newdataset.name = laggedseriesname
        return newdataset    

    def take_diff (self, period = 1, inplace  = False, cols = None, inpct = True,\
                   fieldsep = '', alldays = True, ownfreq = None):
        '''Renvoie la série des différences d'une colonne pour un décalage donné. 
           En vue d'une synchronisation ultérieure dans une matrice, il faut pré-remplir les différences
           par des zéros à toutes les dates ne figurant pas dans l'index.
           #CG 14/6/2: introduction de l'argument ownfreq représentant la fréquence naturelle de la série
           Celle-ci est nécessaire dans le cas d'une série mensuelle présentée quotidiennement,
           avec donc un saut par mois.
        '''
        
        if inplace:
            newdataset = self
        else:
            newdataset = None
        if fieldsep == '':
            fieldsep = glbFieldSep
        cols = self.get_columns(cols)
        #if not (col in self.columns) : return newdataset
        #import pdb; pdb.set_trace()
        #datacols = pd.DataFrame (data = self [cols], columns = cols)
        datacols = self [cols]
        #shifteddata = datacols.shift (period, freq)
        if inpct:
            strope = 'D%'
        else:
            strope = 'D-'
        deltadataname = self.name + fieldsep + strope + '@' + str(period)
        
        #l'instruction suivante renvoie un DataFrame
        #Calculer la série des différences dans l'unité naturelle de la série
        if not inpct:
            #deltadata = datacols.diff (period)
            deltadata = datacols.diff (period)    
        else:
            #deltadata = datacols.pct_change (period)
            deltadata = datacols.pct_change (period)
        
        #dsdelta = TDataSet (index = deltadata.index, data = deltadata)
        if alldays and ownfreq is not None:
            #zerotime = datetime.time(hour = 0, minute = 0, second = 0)
            #dsdelta.set_daytime(zerotime)
            # Prendre les dates sans heure de la série d'origine 
            deltadata.index = deltadata.index.map ( \
                                    lambda (x): datetime (year = x.year, month = x.month, day = x.day, \
                                                    hour = 0, minute = 0))   
            idx_all = pd.bdate_range(start = self.index[0], end = self.index [-1], freq = ownfreq)
            if (ownfreq == 'B' or ownfreq == 'D'):
                #pas de remplissage
                #deltadata = deltadata.reindex (index = idx_all, fill_value = 0.0)
                deltadata = deltadata.reindex (index = idx_all, method = None)
            elif (ownfreq != 'B' and ownfreq != 'D'):
                #cas d'une série mensuelle suréchantillonnée quotidiennement: 
                # on prolonge la dernière variation calculée
                
                deltadata = deltadata.reindex (index = idx_all, method = 'pad')
            else:
                pass
    
        #import pdb; pdb.set_trace()
        # en cas de copie d'objet: on ne renvoie que la colonne résultat
        newcols =  range (len (cols))
        for icol, col in enumerate (cols):
            newcols [icol] = col.split (fieldsep)[0] + strope + '@' + str (period)
        
        if inplace:
            for col in cols:
                newdataset [col] = deltadata [col]
        else:
            #si l'objet diffseries est un pd.Series, il n'a pas de champ columns
            newdataset = TDataSet (index = deltadata.index, 
                                   data = deltadata.values, 
                                   columns = cols)
        newdataset.columns = newcols    
        newdataset.name = deltadataname
        return newdataset
    

    def rolling_returns (self, maincol, substcol, rollfreq, iday, iweek, effectiveroll_lag = 0, 
                         inpct = True, inplace  = False, fieldsep = ''):
        '''Renvoie la série des différences d'une colonne pour un décalage donné.
        Dans le calcul de V (t) / V(t - p), V est la série principale self [maincol].
        Par exception, aux dates spécifiées par la règle rolldate, on calcule V(t) / Vsubst (t-p),
        où Vsubst représente la série self [substcol] 
        '''
        
        assert type (maincol) in [int, str]
        assert type (substcol) in [int, str]
        assert type (rollfreq) == str
        assert iday >= 1
        assert iweek >= 1
        period = 1
        assert type (period) == int
        assert period > 0
        assert effectiveroll_lag in [0,1] 
        
        if inplace:
            newdataset = self
        else:
            newdataset = None
            
        if fieldsep == '':
            fieldsep = glbFieldSep
        #inpct = True
        
        cols = self.get_columns([maincol, substcol])
        if cols is None:
            return None 
        #if not (col in self.columns) : return newdataset
        #import pdb; pdb.set_trace()
        #datacols = pd.DataFrame (data = self [cols], columns = cols)
        maincol = cols [0]
        substcol = cols [1]
        datacols = self [maincol]
        
        #shifteddata = datacols.shift (period, freq)
        if inpct:
            strope = 'D%'
        else:
            strope = 'D-'
        retdataname = self.name + fieldsep + strope + '@' + str(period)
       
        #l'instruction suivante renvoie un DataFrame
        #Calculer la série des différences dans l'unité naturelle de la série
        if not inpct:
            #deltadata = datacols.diff (period)
            retdata = TDataSet (datacols.diff (period))    
        else:
            #deltadata = datacols.pct_change (period)
            retdata = TDataSet (datacols.pct_change (period))
        #Enlever les heures
#         retdata.index = retdata.index.map (lambda (x): datetime (year = x.year, 
#                                                                  month = x.month, 
#                                                                  day = x.day,
#                                                                  hour = 0,
#                                                                  minute = 0,
#                                                                  second = 0))   
        idx_all = pd.bdate_range(start = self.index[0], end = self.index [-1], freq = 'B')
        #assert len (retdata.index) >= len (idx_all) - period 
        #remettre l'heure à 0 pour pouvoir piocher dans un ensemble de dates
        retdata.set_daytime(datetime (2000,1,1))
        self.set_daytime (datetime (2000,1,1))
        #élargir le calendrier pour inclure les dates de rolls de façon certaine
        retdata = retdata.reindex (index = idx_all, method = None)
  
        #générer la série des dates de roll
#         if rollfreq [1:].find ('BS') < 0:
#             rollfreq = rollfreq + 'BS'
        rolldates = pd.bdate_range(start = self.index[0], end = self.index [-1], freq = rollfreq)
        rolldates = rolldates + pd.datetools.WeekOfMonth (week = iweek - 1, weekday = iday - 1)
        #Ne garder que les dates de roll antérieures aux données courantes
        rolldates = rolldates [rolldates <= retdata.index [-1]]
        daybefore_rolldates = rolldates + pd.datetools.BDay( -period)
        dayafter_rolldates = rolldates + pd.datetools.BDay( period)
        
        #timeidx = self.index
        #Contrat M (front) coté jusqu'à TRoll, traité jusqu'en TRoll-1, roulé en TRoll-1
        #Contrat M+1 (next), coté jusqu'à TRoll, devient le front en TRoll + 1, traité en TRoll-1
        #Returns: 
        # en TRoll, Close(F2, TRoll)/ Close (F2, TRoll-1) - 1
        # en TRoll + 1, Close(F1, TRoll+1)/ Close (F2, TRoll-1) - 1
        #dayafter_roll_contract = maincol
        #cas de UX
        if effectiveroll_lag == 0:
            roll_contract = maincol
            daybefore_roll_contract = maincol
        #cas de FVS
        else:
            roll_contract = substcol
            daybefore_roll_contract = substcol            
         
        if inpct:
            rollday_returns  = self.loc [rolldates, roll_contract].values / \
                                self.loc [daybefore_rolldates, daybefore_roll_contract].values - 1
            dayafter_returns = self.loc  [dayafter_rolldates, maincol].values / \
                                self.loc [rolldates, substcol].values - 1
        else:
            rollday_returns  = self.loc [rolldates, roll_contract].values - \
                                    self.loc [daybefore_rolldates, daybefore_roll_contract].values 
            dayafter_returns = self.loc  [dayafter_rolldates, maincol].values - \
                                    self.loc [rolldates, substcol].values
            
#        for idt, dt in enumerate (rolldates) :
        retdata.loc [rolldates, maincol]  = rollday_returns
        retdata.loc [dayafter_rolldates, maincol]  = dayafter_returns
        newcol = maincol.split (fieldsep)[0] + strope + '@' + str (period)
        
        if inplace:
            newdataset [maincol] = retdata () 
        else:
            #si l'objet diffseries est un pd.Series, il n'a pas de champ columns
            newdataset = TDataSet (index = retdata.index, \
                                   data = retdata.values, \
                                   columns = [newcol])
        #newdataset.index = timeidx
        #revenir au calendrier restreint   
        newdataset = TDataSet (newdataset.dropna ()) 
        newdataset.columns = [newcol]    
        newdataset.name = retdataname
        return newdataset

    def fill_missing_values (self, idxmain = 0,  idxsubst = 1, dfsubst = None):
        '''Remplit les valeurs manquantes de la colonne idxmain par la colonne idxsubst '''
        if dfsubst is None:
            df2 = self
        else:
            df2 = dfsubst
        #pdb.set_trace ()
        try:
            maincol = self.get_columns (idxmain) [0]
            substcol = df2.get_columns (idxsubst) [0]
            if dfsubst is not None:
                self [substcol] = df2 [substcol]
          
            self [maincol] [pd.isnull (self [maincol])] = \
                     self [substcol] [pd.isnull (self [maincol])]
        except:
            pass
        return self        

    def take_combi (self, idx1 = 0, coeff1 = 1, \
                          idx2 = 1, coeff2 = 0, constant = 0,\
                          inplace  = False, islinear = True, combiname = ''):
        '''Renvoie la combinaison linéaire ou exponentielle de deux colonnes. '''

        #pdb.set_trace()
        if inplace:
            newdataset = self
        else:
            newdataset = None

        cols1 = self.get_columns(idx1)
        if len (cols1) > 0 :
            col1 = cols1 [0]
            datacol1 = self [col1]
        else: datacol1 = None
        
        cols2 = self.get_columns(idx2)
        if len (cols2) > 0 :
            col2 = cols2 [0]
            datacol2 = self [col2]
        else: datacol2 = None 

        #pdb.set_trace()
        c1null = c1one = c1neg = c2null = c2one = c2neg = False
        if coeff1 == 0 or datacol1 is None:
            c1null = True
        elif abs(coeff1) == 1:
            c1one = True
        if coeff1 < 0:
            c1neg = True
        if coeff2 == 0 or datacol2 is None:
            c2null = True
        elif abs(coeff2) == 1:
            c2one = True
        if coeff2 < 0:
            c2neg = True           
        #pdb.set_trace ()
        if islinear:
            
            combiarray = np.zeros (len (self.index)) + constant
            if not c1null:
                combiarray = datacol1.values * coeff1 + constant
            if not c2null:
                combiarray = combiarray + datacol2.values * coeff2
        else:
            
            #constante égale à 0 en multiplicatif: on la prend pour 1
            if constant == 0:
                constant = 1
            combiarray = np.ones (len (self.index)) * constant
            if  (datacol1 is not None):
                combiarray = np.power (datacol1.values , coeff1) * constant
            if  (datacol2 is not None):
                combiarray = combiarray  *  np.power (datacol2.values, coeff2)         
        #import pdb; pdb.set_trace()
        # en cas de copie d'objet: on ne renvoie que la colonne résultat
        strc1 = strsgn1 = strc2 = strsgn2 =''
        #pas de nom fourni pour la combinaison:
        if combiname == '':
            #détermination du nom de la série
            if islinear:
                if not c1one: strc1 = str (coeff1)
                if c1neg: strsgn1 = '-'
                if not c2one: strc2 = str (coeff2)
                if c2neg: strsgn2 = '-'
                elif not c1null: strsgn2 = '+'
                if c1null: namepart1 = ''
                else: namepart1 = strsgn1 + strc1 + col1
                if c2null: namepart2 = ''
                else: namepart2 =  strc2 + col2
                colname = namepart1
                if len(namepart2) > 0:
                    colname  = colname + strsgn2 + namepart2
            else:
                if not c1null: strc1 = str (coeff1)
                if c1neg: strsgn1 = '-'
                if not c2null: strc2 = str (coeff2)
                if c2neg: strsgn2 = '-'
                if c1null: namepart1 = ''
                else: namepart1 = col1 + '^' + strsgn1 + str( abs(coeff1))
                if c2null: namepart2 = ''
                else: namepart2 = col2 + '^' + strsgn2 + str (abs (coeff2))
                colname = namepart1
                if len (namepart2) > 0:
                    colname = colname + 'x' + namepart2
        else:
            colname = combiname
        #pdb.set_trace()    
        if inplace:
            newdataset [colname] = combiarray
        else:
            #si l'objet diffseries est un pd.Series, il n'a pas de champ columns
            newdataset = TDataSet (index = self.index, \
                                    data = combiarray, \
                                    columns = ['VALUE'])
            #newdataset  = newdataset.ix [dstart, dend]
            newdataset.name = colname
        return newdataset
    
    def take_ema (self, emadecay = None, span = 1, \
                  inplace  = True, cols = None, wres = True, normalize = True, \
                  histoemadata = None, overridedepth = 0, fieldsep = ''):
        '''Renvoie la série des ema d'un ensemble de colonnes pour une pseudo-durée (span) donnée.'''
        ''' self: contient la totalité des données primaires dont on veut calculer la moyenne
            emadecay: coefficient d'atténuation de la moyenne (proche de 1). Prioritaire si fourni.
           span: 2/emadecay - 1
           cols: groupe de colonnes dont on calcule l'ewma.
           wres: si True, on calcule également le résidu
           normalize: si True, on calcule aussi le Z-Score (résidu / ewmastd (même span))
           histoemadata: série facultative contenant les valeurs historiques de l'ewma sur des dates
               normalement antérieures aux données primaires.
           overridedepth: nombre de jours passés (à partir de la donnée la plus récente) à recalculer
           '''
        
        #pdb.set_trace ()
        usehistoforinit = False
        if fieldsep == '':
            fieldsep = glbFieldSep
        if (histoemadata is not None) \
           and (type (histoemadata) == type(self)) \
           and (len(histoemadata.index) > 0) \
           and np.array_equiv (self.columns, histoemadata.columns):
            if (histoemadata.index[0] <= self.index [-1 - overridedepth]) and (histoemadata.index [-1] >= self.index [-1 - overridedepth]):
                usehistoforinit = True        

        self.sort_index (inplace = True)
        if usehistoforinit:
            #cas où on fournit un historique des ema
            histoemadata.sort_index (inplace = True)
            
        cols = self.get_columns (cols)
        #if not (col in self.columns) : return newdataset
        #import pdb; pdb.set_trace()
        #extraction des données à moyenner dans un DataFrame
        datacols = pd.DataFrame (data = self [cols])

        if inplace:
            newdataset = self
        else:
            newdataset = self.copy ().take_columns (cols)

#     #calculer la période synthétique correspondant au coeff s'il est fourni
        if type (emadecay) in [int, float]:
            if emadecay > 0:
                span = 2.0 / emadecay - 1

        if usehistoforinit:
            #historique d'ema fourni
            dhistolast = histoemadata.index [-1]
            dnewlast = self.index [-1]
            #si plus d'historique que de données nouvelles, rien à faire
            if dhistolast >= dnewlast:
                return histoemadata
            if type (dhistolast) == int:
                dfirstnewema = dhistolast + 1
            else:
                dfirstnewema = dhistolast + timedelta (days = 1)
            #extraction du segment de nouvelles données
            datacols = datacols.ix  [dfirstnewema : dnewlast]
            #calcul de l'ema
            newemadata = pd.ewma (datacols, span = span, wres = wres, normalize = normalize)
            #recollement des nouvelles données
            emadata = histoemadata
            emadata.patch_append (newemadata, check_overlap = True)
        else:
            #recalcul de la totalité des données de l'ema
            emadata = pd.ewma (datacols, span = span, adjust = True)
            
        emaname = self.name + fieldsep + 'EMA@' + str(span) 

        #calcul du résidu
        if wres:
            rescols = self [cols] - emadata
            #calcul du ZScore
            if normalize:
                stdevcols = pd.ewmstd (rescols, span = span)
                zcols = rescols * 0.0
                zcols [stdevcols > 0] = rescols [stdevcols > 0] / stdevcols [stdevcols > 0]
                                   
        for col in cols:
            colname = 'EMA_' + col
            newdataset [colname] = emadata [col]
            if wres:
                colname = 'RES_' + col
                newdataset [colname] = rescols [col]
                if normalize:
                    colname = 'Z_' + col
                    newdataset [colname] = zcols [col]                   
                    
        newdataset.name = emaname
        return newdataset

    def take_hp (self, int_lambda = 1600, window = 0, int_depart=6,\
                  inplace  = True, col = None, wres = True, normalize = True, fieldsep=''):
        
        hpname = self.name + fieldsep + 'HP@' + str(int_lambda)
        self.sort_index (inplace = True)
        col = self.get_columns (col)[0]
        
        if inplace:
            newdataset = self
        else:
            newdataset = self.copy()

        
        #faire un lissage sur une fenetre
        if window!=0:
            pente = pd.DataFrame(index=self.index, columns=[col])
            hpdata = pd.DataFrame(index=self.index, columns=[col])
    
            #le lissage pour le int_depart
            #pdb.set_trace ()
            s = sm.tsa.filters.hpfilter (newdataset[col].iloc[0:int_depart], lamb = int_lambda)  # @UndefinedVariable
            hpdata[col].iloc[0:int_depart] = s[1]
            #calcule de la pente
            pente[col].iloc[0:int_depart] = hpdata[col].iloc[0:int_depart].diff()
            for d in range(int(int_depart+1), newdataset.shape[0]):
                #le lissage apres la periode de int_depart
                lissage = sm.tsa.filters.hpfilter(newdataset[col].iloc[max(0, d - window):d], lamb = int_lambda)[1]  # @UndefinedVariable
                hpdata [col].iloc [max(0, d - window):d] [-1] = lissage [-1]
                #la pente
                pente[col].iloc[max(0, d - window):d][-1] = lissage.diff() [-1]
        #faire un lissage avec int_depart fixe
        elif window==0:
            hpdata = pd.DataFrame(np.nan, index=self.index, columns=[col])
            pente = pd.DataFrame(np.nan, index=self.index, columns=[col])

            #le lissage pour int_depart
            #hpdata[col][0:int_depart] = sm.tsa.filters.hpfilter (hpdata[0:int_depart], lamb = int_lambda)[1][col]
            hpdata[col].iloc[0:int_depart] = sm.tsa.filters.hpfilter (hpdata[0:int_depart], lamb = int_lambda)[1]  # @UndefinedVariable
            #la pente
            pente[col].iloc[0:int_depart] = hpdata[col].iloc[0:int_depart].diff()
            for d in range(int_depart, newdataset.shape[0]):
                #le lissage apres la periode de int_depart 
                lissage = sm.tsa.filters.hpfilter (newdataset[col].iloc[newdataset.index[2]:newdataset.index[d]], lamb = int_lambda)[1]  # @UndefinedVariable
                hpdata[col].iloc[hpdata.index[d]] = lissage[-1]
                #la pente
                pente[col].iloc[hpdata.index[d]] = lissage.diff()[-1]
                
        colname = 'HP_' + col
        newdataset [colname] = hpdata[col]

        newdataset ['pente_hp'] = pente[col]

        #calcul du résidu
        if wres:
            #pdb.set_trace ()
            rescol = self [col] - hpdata[col]
            colname = 'RES_' + col
            newdataset [colname] = rescol

            #calcul du ZScore
            if normalize:
                if window!=0:
                    stdevcol = pd.ewmstd (rescol, span = window)
                elif window==0:
                    stdevcol = pd.ewmstd (rescol, span = 100)
                zcol = rescol * 0.0
                zcol [stdevcol > 0] = rescol [stdevcol > 0] / stdevcol [stdevcol > 0]
                
                colname = 'Z_' + col
                newdataset [colname] = zcol                  

        newdataset = TDataSet(newdataset)
        newdataset.name = hpname
        return newdataset

    def calc_coefs_reg(self, col, X=None, int_h_rolling_ou = 200, int_min_p = 20, lag = 1, inplace=True):
        ''' regresse self [col] sur son lag a renvoi les paramatres de la regression suivante: self [col(t+1)] = a.self [col]+b '''
        self.sort_index ( inplace = True )
        if inplace:
            newdataset = self
        else:
            #newdataset = pd.DataFrame(data=self[self.str_columns].values, str_columns=[self.str_columns], index=self.index)
            newdataset = self.copy()

        if X is None:
            #dataframe pour stocker les resultats provisoire
            DataF = pd.DataFrame(index=newdataset.index)

            #nombre d'observations pour chaque date
            DataF['n'] = range(0,len(newdataset))
            #pdb.set_trace()
            #la variable explicative retarde
            str_colLag = newdataset[col].shift(lag)

            #regression non glissante
            if int_h_rolling_ou == 0:
                DataF['Sx'] = np.cumsum( str_colLag )
                DataF['Sy'] = np.cumsum( newdataset[col] )
                DataF['Sxx'] = np.cumsum( str_colLag**2 )
                DataF['Sxy'] = np.cumsum( str_colLag * newdataset[col] )
                DataF['Syy'] = np.cumsum( newdataset[col] **2 )
               
            #regression glissante
            elif int_h_rolling_ou !=0 :
                DataF['Sx'] = pd.rolling_sum( str_colLag, window = int_h_rolling_ou, min_periods = int_min_p )
                DataF['Sy'] = pd.rolling_sum( newdataset[col], window = int_h_rolling_ou, min_periods = int_min_p  )
                DataF['Sxx'] = pd.rolling_sum( str_colLag**2, window = int_h_rolling_ou, min_periods = int_min_p  )
                DataF['Sxy'] = pd.rolling_sum( str_colLag * newdataset[col] , window = int_h_rolling_ou, min_periods = int_min_p  )
                DataF['Syy'] = pd.rolling_sum( newdataset[col]**2, window = int_h_rolling_ou, min_periods = int_min_p  )
                
            #regression multivarier (a completer avec sciket learn)
            elif X is not None:
                if int_h_rolling_ou == 0: 
                    DataF['Sx'] = np.cumsum( DataF[X] )
                    DataF['Sy'] = np.cumsum( DataF[col] )
                    DataF['Sxx'] = np.cumsum( DataF[X] ** 2 )
                    DataF['Sxy'] = np.cumsum( DataF[X] * DataF[col] )
                    DataF['Syy'] = np.cumsum( DataF[col] ** 2 )
                                                         
                elif int_h_rolling_ou!=0:   
                    DataF['Sx'] = pd.rolling_sum( DataF[X], window = int_h_rolling_ou, min_periods = int_min_p )
                    DataF['Sy'] = pd.rolling_sum( DataF[col], window = int_h_rolling_ou, min_periods = int_min_p  )
                    DataF['Sxx'] = pd.rolling_sum( DataF[X] ** 2, window = int_h_rolling_ou, min_periods = int_min_p )
                    DataF['Sxy'] = pd.rolling_sum( DataF[X] * DataF[col] , window = int_h_rolling_ou, min_periods = int_min_p  )
                    DataF['Syy'] = pd.rolling_sum( DataF[col] ** 2, window = int_h_rolling_ou, min_periods = int_min_p  )
                    
            #calcule des params
            ##least squares regression:

            #le coef      
            a = ( DataF['n'] * DataF['Sxy'] - DataF['Sx'] * DataF['Sy']) / ( DataF['n'] * DataF['Sxx'] - DataF['Sx']**2 )
            #la constante
            b = (DataF['Sy'] - a * DataF['Sx']) / DataF['n']
                        
            #moyenne mobile sur la variable a expliquer
            if int_h_rolling_ou==0:
                _ = pd.rolling_mean( newdataset[col], window = len(newdataset[col]), min_periods = 2 )#mu
            #mu = b / ( 1 - a )

            newdataset['a'] = a
            newdataset['b'] = b
            #newdataset['mu'] = mu

            newdataset = TDataSet(newdataset)

            return newdataset

    def calc_params_ou (self, col_a, col_b, delta=1, inplace=True):
        if inplace:
            newdataset = self
        else:
            newdataset = self.copy ()

        #la moyenne
        mu = newdataset[col_b] / ( 1 - newdataset[col_a] )

        #la vitesse de retour vers la moyenne
        Lambda = - np.log(newdataset[col_a]) / delta

        newdataset['lambda'] = Lambda
        newdataset['mu'] = mu

        return newdataset

    def calc_esp_mr(self, col, i_h_pred, col_lambda, col_mu, inplace=True):
        ''' calcule l'esperance d'un processus mean reverting'''
        if inplace:
            newdataset = self
        else:
            newdataset = self.copy ()
            
        col_mu = newdataset [col_mu]
        col_lambda = newdataset [col_lambda]
        
        #calcule d'esperance d'un processus de retour a la moyenne
        esp = newdataset[col] * np.exp( - col_lambda * i_h_pred ) + col_mu * (1 - np.exp(- col_lambda * i_h_pred))
    
        newdataset['esp_mr'] = esp
        
        return newdataset

    def pred_extrap (self, col_lissage, col_pente, i_h_pred, inplace=True):
        if inplace:
            newdataset = self
        else:
            newdataset = self.copy ()

        #la prediction:
        newdataset['pred_extrap'] = newdataset[col_lissage] + i_h_pred * newdataset[col_pente] 

        return newdataset

    def predict_hp (self, col, i_h_pred, int_lambda, window, coef_amor=0, int_h_rolling_ou=200, lag=1,
                 int_delta_ou=1, int_h_lissage_pente=0, bo_extrapolation=True, with_esp_res=True, inplace=True):
        '''retourne la colonne 'prediction' '''
        if inplace:
            newdataset = self
        else:
            newdataset = self.copy ()
        #ajouter la colonne du lissagee et residu et la pente
        newdataset.take_hp (wres = True, normalize = True, fieldsep='',int_lambda = int_lambda, window = window,
                             int_depart = i_h_pred,inplace = inplace,col=col)
        
        if with_esp_res:
            #calcule des coefs d'ou
            newdataset.calc_coefs_reg (col='RES_'+col, X=None, int_h_rolling_ou=int_h_rolling_ou, int_min_p=20, lag = lag, inplace=True)
            #calcule d'esperance de residu
            #calcule du lambda et du mu
            newdataset.calc_params_ou (col_a='a', col_b='b', delta=int_delta_ou, inplace=True)
            #on garde que les lambdas positifs ou nul
            newdataset['lambda'] = scipy.maximum(newdataset['lambda'], 0)
            ds_esp = newdataset.calc_esp_mr (col = 'RES_'+col, i_h_pred = i_h_pred, col_lambda = 'lambda', col_mu = 'mu', inplace = inplace)['esp_mr']
        else:
            ds_esp = coef_amor * newdataset['RES_'+col]
        if bo_extrapolation:
            #prediction extrap
            if int_h_lissage_pente != 0:
                newdataset ['pente_hp'] = pd.rolling_mean ( newdataset['pente_hp'], int_h_lissage_pente)
            ds_extrapolation = newdataset.pred_extrap (col_lissage = 'HP_'+col, col_pente = 'pente_hp', i_h_pred = i_h_pred, inplace = inplace)['pred_extrap']
        else:
            ds_extrapolation = 0

        #prediction de t+h a la date t
        newdataset['prediction'] = ( ds_extrapolation + ds_esp )

        return newdataset
    
    def predict_ewma (self, col, i_h_pred, coef_amor = 0, int_h_rolling_ou = 200, lag = 1, int_delta_ou = 1, int_h_lissage_pente = 0,
                     bo_extrapolation = True, with_esp_res = True, ReturnD = False, inplace = True):
        if inplace:
            newdataset = self
        else:
            newdataset = self .copy()
        
        #perf sur deltaT
        newdataset [col+str(i_h_pred)] = pd.rolling_sum (newdataset [col], i_h_pred)
            
        #la tendance 
        lissage_ewma = pd.ewma (newdataset [col+str(i_h_pred)], i_h_pred)
        newdataset ['lissage'] = lissage_ewma
        newdataset ['residu'] = newdataset [col+str(i_h_pred)] - lissage_ewma

        
        if with_esp_res == True:
            #calcule d'esperance de residu
            newdataset.calc_coefs_reg (col = 'residu', X = None, int_h_rolling_ou = int_h_rolling_ou, int_min_p = 20, lag = lag, inplace = True)
            newdataset.calc_params_ou ('a', 'b', delta = int_delta_ou, inplace = True)
            #garder les lambda positifs
            newdataset ['lambda'] = scipy.maximum ( newdataset ['lambda'] , 0 )
            #camort
            newdataset ['esp_mr'] = newdataset.calc_esp_mr ('residu', i_h_pred = i_h_pred, col_lambda = 'lambda', col_mu = 'mu', inplace = True) ['esp_mr']
            #avec ou
            if bo_extrapolation == True:
                if int_h_lissage_pente == 0:
                    ds = lissage_ewma.diff () * i_h_pred + newdataset ['esp_mr']
                       
                elif int_h_lissage_pente != 0:
                    pente = pd.rolling_mean (lissage_ewma.diff (), int_h_lissage_pente)
                    ds = pente * i_h_pred + newdataset ['esp_mr']
            newdataset ['prediction'] = lissage_ewma + ds
        
        else:
            #sans ou
            if bo_extrapolation == True:
                if int_h_lissage_pente == 0:
                    ds = lissage_ewma.diff () * i_h_pred + coef_amor * newdataset ['residu']
                    
                elif int_h_lissage_pente != 0:
                    pente = pd.rolling_mean (lissage_ewma.diff (), int_h_lissage_pente)
                    ds = pente * i_h_pred + coef_amor * newdataset ['residu']

            newdataset ['prediction'] = lissage_ewma + ds
             
        #pdb.set_trace ()
        return newdataset

    def predict_test (self, col, i_h_pred, i_c_court, inplace = True ):
        
        if inplace:
            newdataset = self
        elif inplace == False:
            newdataset = self.copy()
         
        newdataset ['esp'] = pd.ewma (newdataset [col], i_h_pred)
        newdataset ['esp'] = pd.ewma (newdataset ['esp'], i_c_court)
#         newdataset ['residu'] = pd.ewma (newdataset [col], i_h_pred) - newdataset ['esp']
#         
#         indexR = newdataset ['residu'][newdataset ['residu']<=0].index
#         newdataset ['esp'][indexR] = pd.ewma (newdataset [col], i_h_pred)[indexR]
        
        
        return newdataset


    
    def ewma_amort ( self, col, quantile_max, quantile_min, window, h_pred, h_lissage, inplace = True):
        #nbs_amort = len (self)
        if inplace:
            newdataset = self
        else:
            newdataset = self.copy ()
        newdataset ['lissage'] = pd.ewma ( newdataset [col], h_pred)
        
        if h_lissage != 0 :
            newdataset ['lissage'] = pd.ewma ( newdataset ['lissage'], h_lissage)
        elif h_lissage == 0:
            pass
        
        newdataset ['pred'] = np.nan
        for i in range (window, len (newdataset)):
            
            #on recupere la fentre des donnees 
            index_i = newdataset ['lissage'][max ( i-window, 0): i].index
            ds_i = pd.DataFrame (newdataset ['lissage'][max ( i-window, 0): i], index = index_i, columns = ['lissage'])
            #on trie
            valeurs_decr = ds_i.sort ( columns = 'lissage', ascending = False)

            #on recupere les valeurs
            valeur_max = valeurs_decr.quantile (quantile_max).values [0]
            valeur_min = valeurs_decr.quantile (quantile_min).values [0]

            if valeur_max != valeur_min:
                #on recupere les dates des valeurs max et min
                date_max = valeurs_decr.index [valeurs_decr ['lissage'] < valeur_max] [0]
                date_min = valeurs_decr.index [valeurs_decr ['lissage'] < valeur_min] [0]
            
                #pdb.set_trace ()
                #remplacer les valeurs max min dans ds
                valeurs_decr ['lissage'][:date_max] = valeur_max
                valeurs_decr ['lissage'][date_min:] = valeur_min
                
                #on trie par date
                valeurs = valeurs_decr.sort_index ()
                #newdataset ['pred'][index_i] = valeurs ['lissage'][index_i]
                newdataset ['pred'][index_i [-1]] = (valeurs ['lissage'] ) [index_i [-1]]
                
            else:
                newdataset ['pred'] = newdataset ['lissage']
                break
                        
        return newdataset
                
    def count_saturation (self, contraintes, epsilon = 1e-1):
        contraintes_ = [i.upper() for i in contraintes.keys() ]
        
        res_dic = {}
        if 'DD'in contraintes_:
            contraintes['DDCur'] = contraintes.pop ('DD')
            
        if 'ACTIONS'in contraintes_:
            contraintes['ExpoActions'] = contraintes.pop ('Actions')
    
        if 'NOMINAL'in contraintes_:
            contraintes['Exposure'] = contraintes.pop ('Nominal')
    
        for col in contraintes.keys ():
            l_interval_float = contraintes [col][0] * (1 + epsilon)
            u_interval_float = contraintes [col][1] * (1 - epsilon)
            
            col_ds = self [col]
            self [col+'_saturated'] = np.logical_or (col_ds < l_interval_float, col_ds > u_interval_float)
            self [col+'_satisfied'] = np.logical_and (col_ds >= contraintes [col][0], col_ds <= contraintes [col][1])
            
            self [col+'_saturated'] = np.uint (self [col+'_saturated'])
            self [col+'_satisfied'] = np.uint (self [col+'_satisfied'])
            
            res_dic[col+'_saturated'] = float ( sum (self [col+'_saturated']) ) / len (self)
            res_dic [col+'_satisfied'] = float ( sum (self [col+'_satisfied']) ) / len (self)
            
        first = True
        self ['atleastone_constraint_sat'] = np.zeros (len(self [col])) 
        self ['all_constraints_sat'] = np.zeros (len (self [col]))
        for i in range ( 1, len (contraintes.keys ()) ):
            col = contraintes.keys ()[i] + '_saturated'
            
            if first:
                first = False
                col0 = contraintes.keys ()[0] + '_saturated'
                
                self ['atleastone_constraint_sat'] = np.maximum (self [col0], self [col]) 
                self ['all_constraints_sat'] = np.multiply  (self [col0], self [col])
                
            else:
                
                self ['atleastone_constraint_sat'] = np.maximum (self ['atleastone_constraint_sat'], self [col])
                self ['all_constraints_sat'] = np.multiply  (self ['all_constraints_sat'], self [col]) 
        
        AuMoins_contrainte_sature = sum (self ['atleastone_constraint_sat'])
        Toute_contraintes_sature = sum (self ['all_constraints_sat'])
        
        AuMoins_contrainte_sature = float (AuMoins_contrainte_sature) / len (self)
        Toute_contraintes_sature = float (Toute_contraintes_sature) / len (self)
        
        res_dic['atleastone_constraint_sat'] = AuMoins_contrainte_sature
        res_dic['all_constraints_sat'] = Toute_contraintes_sature
        
        return res_dic
    
    
    def calc_conditioned_ema (self, cdecay, ycols, coeffcol, \
                              inplace = True, initdend = None, lag = 1):

        ycols = self.get_columns (ycols)
        if ycols is None:
            return None

        coeffcol = self.get_columns (coeffcol)
        if coeffcol is None or (type (coeffcol) == list and len (coeffcol) > 1):
            return None        

        if inplace:
            gemads = self
            coloffset = len (self.columns)
        else:
            gemads = TDataSet (index = self.index)
            coloffset = 0

        try:
            ydata = self.take_columns (ycols, forceuppercase = False)
            cdata = self.take_columns (coeffcol, forceuppercase = False)

        except:
            return None
        
        if ydata is None or cdata is None:
            return None

        #initdstart = gemads.index [0]
        if initdend is None:
            initdend  = gemads.index [-1]

        initydata = ydata.take_interval (dstart = None, dend = initdend)
        initcdata = cdata.take_interval (dstart = None, dend = initdend)

        gemacols = []
        for icol, col in enumerate (ydata.columns):
            #pdb.set_trace ()
            gemacol = 'GEma_' + col
            gemacols.append (gemacol)
            gemads [gemacol] = np.nan

        uemads = TDataSet (index = self.index, columns = gemacols)

        #pdb.set_trace ()
        for icol, ycol in enumerate (ydata.columns):
            col = 'GEma_' + ycol
            ucol = uemads [col]
            ydatacol = ydata [ycol]
            #gcol = gemads [col]
            
            uemads.iat [0 , icol]  = float (initydata [initcdata.values > 0] [ycol].mean ())
            gemads.iat [0, icol + coloffset]  = ucol.iloc [0]

            #cydata = ydata [cdata.values > 0] [ycol]

            #cemaydata = pd.ewma (ycols, span = 2 / cdecay - 1, adjust = True)
            if cdecay > 0:
                for idx in range (1, len(self.index)):
                    if idx < lag:
                        uemads.iat [idx, icol] = ucol.iat [0]
                        gemads.iat [idx, icol + coloffset] = ucol.iat [0]
                    else:
                        coeff = cdecay * float (cdata.iloc [idx - lag])
##                        if coeff > 0:
##                            pdb.set_trace ()
                        uemads.iat [idx, icol] = coeff * float (ydatacol.iat [idx])  \
                                               + (1 - coeff) * (ucol.iat [idx - lag])
                if lag > 1:
                    #pdb.set_trace ()
                    gemads [col] = pd.rolling_mean (ucol, lag)
                    gemads [col].fillna (method = 'bfill', inplace = True)
                else:
                    #pdb.set_trace ()
                    gemads [col] = ucol
            else:
                gemads [col] = ucol.iat [0]

        return gemads


    def calc_running_regression (self, cdecay, ycol, xcols = None, inplace = True, ylag = 0, \
                                 initdstart = None, initdend = None, add_constant = True, ifreq = 1):
        '''Calcule une régression glissante. 0 < cdecay = 2 / (span  + 1) << 1  '''
        #pdb.set_trace ()
        ylag = max (0, ylag)
        ycol = self.get_columns (ycol)
        
        if ycol is None or (type (ycol) == list and len (ycol) > 1):
            return None

        if inplace:
            regds = self
            #coloffset = len (self.columns)
        else:
            regds = TDataSet (index = self.index)
            #coloffset = 0

        try:
            xcols = self.get_columns (xcols)
            xdata = self.take_columns (xcols, forceuppercase = False)
            ydata = self.take_columns (ycol, forceuppercase = False)

        except:
            return None 

        if xdata is None:
            return None
        elif add_constant:
            xdata ['_Xone' ] = 1

        if initdstart is None:
            initdstart = regds.index [0]
        else:
            initdstart = pd.to_datetime (initdstart)

        if initdend is None:
            initdend  = regds.index [-1]
        if type(initdend) == type(str()):
            initdend = pd.to_datetime (initdend)
        if type(initdend) == type(list()):
            initdend = initdend[0]
        #création des colonnes spécifiques à la régression
        datacols = xcols
        datacols.append (ycol [0])
        regcols = []
        regcols.append ('Estim_' + ycol [0])
        regcols.append ('Res_'  + ycol [0])
        #nc = len (xdata.columns)
        for irow, row in enumerate (xdata.columns):
            #pdb.set_trace ()
            regcols.append ('C' + ycol [0] + '_' + row)
        for irow, row in enumerate (xdata.columns):        
            for icol, col in enumerate (xdata.columns):
                if icol >= irow:
                    regcols.append ('C' + row + '_' + col)
        for irow, row in enumerate (xdata.columns):                            
            regcols.append ( 'Alpha_' + row)


        #pdb.set_trace ()
        for col in regcols:
            regds [col] = float (0)
        
        #Initialisation de la régression
        initxdata = xdata.take_interval (dstart = initdstart, dend = initdend)
        initxdata = initxdata.iloc [ : -ylag]
        initydata = ydata.take_interval (dstart = initdstart, dend = initdend)
        initydata = initydata.iloc [ylag : ]
        #nr = len (initxdata.index)

        nxcols = len (xdata.columns)
        
        covmat = np.matrix (np.ndarray (shape = (nxcols, nxcols)))
        yxvect = np.matrix (np.ndarray (shape = (nxcols, 1)))

        #yestimcol = regds['Estim_' + ycol [0]]
        #yrescol = regds ['Res_'  + ycol [0]]
        
        for irow, row in enumerate (initxdata.columns):
            #pdb.set_trace ()
            yxvect [irow, 0] = np.inner (initxdata [row].values, \
                                         initydata.values.reshape (initxdata [row].values.shape))
            regds ['C' + ycol [0] + '_' + row].iloc [0] = yxvect [irow, 0]
            for icol, col in enumerate (xdata.columns):
                if icol >= irow:
                    covmat [irow, icol] = np.inner (initxdata [row].values, initxdata [col].values)
                    regds ['C' + row + '_' + col].iloc [0]  = covmat  [irow, icol]
                else:
                    covmat [irow, icol] = covmat [icol, irow]
        try:
            #pdb.set_trace ()
            invcovmat = np.linalg.inv (covmat)
        except:
            return regds
        #Le vecteur des coefficients alpha de régression
        avect = np.dot (invcovmat, yxvect)
        #pdb.set_trace ()
        #Fixer les alpha initiaux
        for irow, row in enumerate (xdata.columns):
            regds ['Alpha_' + row].iloc [0] = avect [irow, 0]      
        #pdb.set_trace ()
##        yestim = np.inner (avect.reshape ((nxcols, )), initxdata.ix [0].values)
##        yestimcol.iloc [0]  = float (yestim [0])
##        #residue = initydata.ix [0] - yestim [0]
##        #pdb.set_trace ()
##        yrescol.iloc [0] = initydata.iloc [0, 0] - float (yestim [0])

        #Récurrence
        if cdecay > 0:
#             yxvectavg = yxvect
#             covmatavg = covmat
#             span = 1 / cdecay - 1
            for idx in range (1 + ylag, len(self.index)):
                #ne calculer qu'après la période d'initialisation
                #et pour la fréquence donnée
                if self.index [idx] >= initdend:
                    if idx % ifreq == 0:
                        #pdb.set_trace ()
                        for irow, row in enumerate (xdata.columns):
                            # On calcule le produit ^X[t-lag] . Y[t]
                            yxvect [irow, 0] = (1 - cdecay)* ydata.iloc [idx] * xdata.iloc [idx - ylag, irow] + \
                                                  (1 - cdecay) * regds ['C' + ycol [0] + '_' + row].ix [idx - 1]
                            regds ['C' + ycol [0] + '_' + row].iloc [idx] =  yxvect [irow, 0]
                            for icol, col in enumerate (xdata.columns):
                                if icol >= irow:
                                    covmat [irow, icol] =  (1 - cdecay) * xdata.iloc [idx, irow] * xdata.iloc [idx, icol] + \
                                                            (1 - cdecay) * regds ['C' + row + '_' + col].values [idx - 1]
                                    regds ['C' + row + '_' + col].iloc [idx]  = covmat  [irow, icol]
                                else:
                                    covmat [irow, icol] = covmat [icol, irow] 
                            #covmat = covmatavg * span               
                        try:            
                            invcovmat = np.linalg.inv (covmat)
                            avect = np.dot (invcovmat, yxvect)
                        except:
                            pass
                        #pdb.set_trace ()
                        for irow, row in enumerate (xdata.columns):
                            regds ['Alpha_' + row].iloc [idx]  = avect [irow, 0]      
    ##                    #pdb.set_trace ()
    ##                    yestim = np.inner (avect.reshape ((nxcols, )), xdata.ix [idx].values)
    ##                    yestimcol .iloc [idx]= yestim [0]
    ##                    #pdb.set_trace ()
    ##                    #residue = ydata.iloc [idx, 0] - yestim [0]
    ##                    yrescol.iloc [idx] = ydata.iloc [idx, 0] - float (yestim [0])
                    else:
                        # en dehors de la fréquence spécifiée: prolonger
                        #pdb.set_trace ()
                        regds.ix [idx, regcols] = regds.ix [idx - 1, regcols]
                else:
                    #à l'intérieur de la période d'initalisation, garder les valeurs initiales
                    regds.ix [idx, regcols] = regds.ix [0, regcols]
        else:
        #cdecay==0 : valeurs constantes
            for col in regcols:
                regds [col].iloc [1: len(self.index)] = regds [col].ix [0]
            #regds [regcols].iloc [1: len(self.index)] = regds [regcols].ix [0]
            #pdb.set_trace ()
        #Finalisation: calcul des estimations
        for col in xdata.columns:
            prod = regds ['Alpha_' + col] * xdata [col]
            regds ['Estim_'  + ycol [0]] = prod + regds ['Estim_'  + ycol [0]]

        #yrescol = ydata[ycol [0]] - regds ['Estim_'  + ycol [0]]
                        
        return regds

    def take_vol (self, period = 1,
                  window = 20, inplace  = True,
                  annualize = True, 
                  fillinit = True,
                  inpct = True, 
                  cols = None,
                  fieldsep = ''):
        '''Renvoie la série des volatilités de rendements '''
        if inplace:
            newdataset = self
        else:
            newdataset = self.copy ()
        if fieldsep == '':
            fieldsep = glbFieldSep
        cols = self.get_columns (cols)
        #attention, le diff peut changer l'index
        if period == 0:
            diffdata = newdataset.take_cols (cols)
        else:
            diffdata = self.take_diff (period = period, cols = cols, inpct = inpct, alldays = False)
            
        voldata = pd.rolling_std (diffdata, window = window)
        if fillinit:
            voldata [0 : window] = voldata [0 : window + 1].fillna (method = 'bfill') 
        voldata = voldata.dropna ()
        annfactor = 1
        #pdb.set_trace()
        #voldata = pd.rolling_mean (diffdata, window = window)
        volname = self.name + glbFieldSep + 'VOL@' + str(window)
        newcols = range (len (cols))
        for icol, col in enumerate (cols):
            if annualize:
                nfreqdict = self.estimate_nat_freq (col)
                nfreq = max (1, nfreqdict ['min'])
                annfactor = math.sqrt( 260 / nfreq)
            else:
                annfactor = 1
            newdataset [col] = voldata [voldata.columns[icol]] * annfactor
            #nom de la colonne: radical + Vol
            newcols [icol] = col.split(fieldsep)[0] + fieldsep + 'VOL@'+ str(window)

        newdataset.name = volname
        newdataset.columns = newcols
        return newdataset
    
   
    def take_OHLC_vol (self,
                       window = 20,
                       annualize = True, 
                       fillinit = True,
                       inpct = True, 
                       OHLCcols = None,
                       algo = 'yang', 
                       fieldsep = ''):
        '''Renvoie la série des volatilités de rendements '''
        newdataset = TDataSet (index = self.index)  # @UnusedVariable
        
        if fieldsep == '':
            fieldsep = glbFieldSep
        
        cols = self.get_columns (OHLCcols)
        if len (cols) < 4:
            return None
    
        ocol = cols [0]
        hcol = cols [1]
        lcol = cols [2]
        ccol = cols [3]    
        if inpct:
            oreturn  =  self[ocol] / self[ccol].shift (1) - 1
            creturn =   self[ccol] / self[ocol] - 1
            hreturn = self[hcol] / self[ocol] - 1
            lreturn = self[lcol] / self[ocol] - 1
        else:
            oreturn = self[ocol] - self[ccol].shift (1)
            creturn = self[ccol] - self[ocol]
            hreturn = self[hcol] - self[ocol]
            lreturn = self[lcol] - self[ocol]
        
        ovar = pd.rolling_var (oreturn, window = window)  
        closevar = pd.rolling_var (oreturn + creturn, window = window) 
        
        if algo == 'park':
            lhvar = pd.rolling_var (hreturn - lreturn, window = window)
            retvar = lhvar / (4 * math.log(2))
        else :
            sqret = hreturn * (hreturn - creturn) + lreturn * (lreturn - creturn)
            if algo == 'rogers':
                retvar = pd.rolling_mean (sqret, window = window)
            elif algo == 'yang':
                k = 0.0783
                sqret = hreturn * (hreturn - creturn) + lreturn * (lreturn - creturn)
                retvar = ovar + k * closevar + (1 - k) * pd.rolling_mean (sqret, window = window)
        
        voldata = np.sqrt (retvar)
        if fillinit:
            voldata [0 : window + 1] = voldata [0 : window + 1].fillna (method = 'bfill') 
        voldata = voldata.dropna ()
        annfactor = 1
        #pdb.set_trace()
        #voldata = pd.rolling_mean (diffdata, window = window)
        volname = self.name + glbFieldSep + 'VOL' + algo [0] + '@' + str(window)
        if annualize:
            nfreqdict = self.estimate_nat_freq (cols [0])
            nfreq = max (1, nfreqdict ['min'])
            annfactor = math.sqrt( 260 / nfreq)
        else:
            annfactor = 1
        
        newdataset = TDataSet (voldata * annfactor)
        #nom de la colonne: radical + Vol
        newdataset.name = volname
        newdataset.columns = [volname]
        return newdataset
 
    

    def take_corr (self, period = 1,
                   span = 20, exponential = True, inplace  = True,
                   inpct = True, cols = None, lag2 = 0,
                   fieldsep = ''):
        '''Renvoie la série des corrélations entre deux colonnes d'un Dataset
           period: si 0, corrélation des valeurs, si > 0, corrélation des variations sur period
           lag2: retard sur la seconde colonne
           cols: spécifications de 1 ou 2 colonnes
        '''
        if inplace:
            newdataset = self
        else:
            newdataset = TDataSet (index = self.index)
            
        if fieldsep == '':
            fieldsep = glbFieldSep
        cols = self.get_columns (cols)
        if len (cols) == 1:
            col1 = cols [0]
            col2 = col1
        else:
            col1 = cols [0]
            col2 = cols [1]
        ## si period == 0 c'est l'autocorrélation des valeurs 
        ## et non des variations qui est calculée
        startval = period + lag2 * period
        if period == 0:
            data1 = self [col1]
            data2 = self [col2].shift (periods = lag2)
        else:
            if inpct:
                data1 = self [col1].pct_change (period) [startval : ]
                data2 = self [col2].pct_change (period).shift (periods = lag2 * period) [startval : ]
            else:
                data1 = self [col1].diff (period) [startval : ]
                data2 = self [col2].diff (period).shift (periods = lag2 * period) [startval : ]
        
        if exponential:
            corrdata = pd.ewmcorr (data1 [startval : ], data2 [startval : ], span = span)
        else:
            corrdata = pd.rolling_corr (data1, data2, window = span)
        #pdb.set_trace()
        #voldata = pd.rolling_mean (diffdata, window = window)
        corrname = self.name + glbFieldSep + 'CORR'
        corrname = corrname + '[' +  str(col1) + ',' + str(col2) + ',' + str(span) + ']'   
        corrdata = corrdata.dropna()   
        newdataset ['Corr'] = corrdata
        #nom de la colonne: radical + Vol
        #newcols [icol] = col.split(fieldsep)[0] + fieldsep + 'VOL@'+ str(window)
        newdataset.name = corrname
        #newdataset.columns = newcols
        return newdataset  


    def categorize   (self, quantilize = False, \
                        levels = 2, inplace  = True, cols = None, \
                        dstart = None, dend = None, fieldsep = ''):
        '''Renvoie la série des colonnes catégorisées; levels: entier ou tableau.'''
        
        if inplace:
            newdataset = self
        else:
            newdataset = self.copy ()
        if fieldsep == '':
            fieldsep = glbFieldSep
        #pdb.set_trace ()
        cols = self.get_columns (cols)
##        if dstart is None: dstart = self.index [0]
##        if dend is None: dend = self.index [-1]
        
        if type (levels) == int:
            nlevels = levels
            if nlevels > 0:
                levels = np.array (range (0, 1 + nlevels, 1)) * float (1.0 / nlevels)
        else:
            nlevels = len (levels)
        catname = self.name + fieldsep + 'MOD@' + str (nlevels)

        for col in cols:
            if quantilize:
                #subcatcol = pd.qcut (newdataset [col][dstart:dend], qlevels)
                #qlevels = np.zeros (nlevels + 1)
                #pdb.set_trace()
                subseries = newdataset.take_columns (col)
                subseries = subseries.take_interval (dstart = dstart, dend = dend)#, inplace = True)
                subseries = subseries.dropna ()
                #subseries = newdataset [col] [dstart:dend]
                levels [0] = -1000000000.0
                levels [-1] = 1000000000.0
                for ilevel in range (1, nlevels):
                    levels[ilevel] = subseries.quantile (float( ilevel) / nlevels)
                #Patch CG 7Nov14: lorsque la distrib est irrégulière, on peut récolter des doublons
                levels = np.unique (levels)
                #levels = subcatcol.levels
            #pdb.set_trace()
            #colname = col.split(fieldsep)[-1] + fieldsep + 'M' + str (nlevels)
            #Patch CG 12Dec13
            colname = col.split(fieldsep)[-1] + '_M' + str (nlevels)
            #df = newdataset [col].dropna ()

            catcol = pd.cut (x = newdataset [col], bins = levels, labels = False)
            #newdataset [colname] = -1
            newdataset [colname] = catcol
            newdataset [newdataset[colname] < 0] = np.NaN

        newdataset.name = catname
        return newdataset
    
    def categorize_expending   (self, quantilize = False, \
                        levels = 2, inplace  = True, cols = None, \
                        dstart = None, dend = None, fieldsep = ''):
        '''Renvoie la série des colonnes catégorisées; levels: entier ou tableau.'''
        
        if inplace:
            newdataset = self
        else:
            newdataset = self.copy ()
        if fieldsep == '':
            fieldsep = glbFieldSep
        #pdb.set_trace ()
        cols = self.get_columns (cols)
##        if dstart is None: dstart = self.index [0]
##        if dend is None: dend = self.index [-1]
        
        if type (levels) == int:
            nlevels = levels
            if nlevels > 0:
                levels = np.array (range (0, 1 + nlevels, 1)) * float (1.0 / nlevels)
        else:
            nlevels = len (levels)
        catname = self.name + fieldsep + 'MOD@' + str (nlevels)

        for col in cols:
            colname = col.split(fieldsep)[-1] + '_M' + str (nlevels)
            #df = newdataset [col].dropna ()
            
            
            def pdCut_For_expend (data_tocut, levels=levels, nlevels=nlevels):
                levels [0] = -1000000000.0
                levels [-1] = 1000000000.0
                for ilevel in range (1, nlevels):
                    data_tocut = pd.DataFrame (data_tocut)
                    levels[ilevel] = data_tocut.quantile (float( ilevel) / nlevels)
                levels = np.unique (levels)
                return pd.cut (x = data_tocut, bins = levels, labels = False) [-1]
            
            catcol = pd.expanding_apply (newdataset [col], pdCut_For_expend, 15)
            
            
            #catcol = pd.cut (x = newdataset [col], bins = levels, labels = False)
            #newdataset [colname] = -1
            newdataset [colname] = catcol
            newdataset [newdataset[colname] < 0] = np.NaN

        newdataset.name = catname
        return newdataset


    def auto_categorize (self, mod = 10, level_date = None, date_end = None, minR = 0.02):
        ''' Renvoie une liste : (ds catégorisée ou None si on à moins de deux modalitées, nombre de modalités, les bins)'''
        #pdb.set_trace ()
        ds_copy = self.copy ()
        if date_end is not None:
            if ds_copy.index.nlevels == 1:
                ds_copy = ds_copy.loc [:date_end]
            #cas d'un multi index    
            elif ds_copy.index.nlevels == 2:
                ds_copy = ds_copy.loc [ds_copy.index.get_level_values(level_date) <= date_end]
                ds_copy = ds_copy.stack()
            
        df_q = [ds_copy.quantile (q = i/float(mod)) for i in range (0, int(mod)+1, 1)]
        df_q = pd.DataFrame (df_q)
        bins = list (np.unique (df_q.dropna(how='all')))
    
        if len (bins) > 2:
            bins [0] = -10000000.
            bins [-1] = 10000000.
            
            def qq (serie, bins):
                '''catégorise les colonnes d'un ds '''
                res = pd.cut (serie, bins = bins, right = False, retbins = True, labels = False)[0]
                return res 
            
            def checkDf (ds, minR = minR, bins = bins):
                ''' vérification si toutes le modalitées couvre au moins minR%. Si ok on renvoi les bins d'origine, sinon on finsionne les modalités et en renvoi les bins'''
                bins_ = bins [:]
                total = float (len (ds.dropna(how ='all')))
                #pdb.set_trace()
                for i in range (len(bins)-1):
                    j = i + 1
                    try:
                        v = len (ds.loc [(ds>=bins[i]) & (ds<bins[j])].dropna (how ='all'))
                    except:
                        col_ = ds.columns[0]
                        v = len (ds[col_].loc [(ds[col_]>=bins[i]) & (ds[col_]<bins[j])].dropna())
                    if (v/total) < minR:
                        bins_.remove (bins[j])
                return bins_
            
            bins = checkDf (ds_copy, minR = minR)
            if len (bins) > 2:
                ds = TDataSet (self.copy ().apply (lambda x: qq (x, bins)))
                return ds#, len(bins)-1, bins
            else:
                return np.nan#, len (bins)-1, bins
        else:
            return np.nan#, len(bins)-1, bins
    
    def take_cumulative_return (self, inplace  = True, cols = None, timeweight = False, fieldsep = ''):
        '''Renvoie le cumul composé des rendements'''
        '''AMBIGU quand inplace = True, cols <> None'''
##        
##        #pdb.set_trace()
##        if inplace:
##            newdataset = self
##        else:
##            newdataset = self.copy ()
##        #récupérer la liste de libellés de colonnes
        if fieldsep == '':
            fieldsep= glbFieldSep
        cols = self.get_columns (cols)
        #pdb.set_trace()
        #datacols = pd.DataFrame (data = self [cols])
        if timeweight == True:
            deltatime = pd.Series(self.index.asi8)
            deltatime = deltatime.diff(1) / glbNanoSexPerDay
            deltatime.fillna (value = 0.0, inplace = True)
            deltatime = deltatime / 365.25
            deltatime = deltatime.reshape (len (self.index), 1)
            self [cols] = self [cols] * deltatime
        navdata = np.log (1 + self [cols])
        navdata = pd.expanding_sum (navdata)
        navdata = np.exp (navdata)
        navname = self.name.split (fieldsep)[0] + '_CMPD'
        newcols = range (len (cols))

        if inplace:
            newdataset = self
        else:
            newdataset = TDataSet (data = navdata)
        for icol, col in enumerate (cols):
            if inplace:
                newdataset [col] = navdata [col]
            newcols [icol] = col.split(fieldsep)[0] + fieldsep + 'NAV'
                      
        newdataset.name = navname
        newdataset.columns = newcols
        
        return newdataset

    def calc_modified_duration (self, n, cols = None, fieldsep = ''):
        ''' Renvoie la série des sensibilités = modified duration'''
        ''' Pour une série de taux et une maturité '''

        if fieldsep == '':
            fieldsep = glbFieldSep
            
        cols = self.take_columns (cols)
        cols = np.maximum (cols, 1e-5)
        zc = 1.0 / (1.0 + cols)
        zcn = zc**n
        res = 1.0
        res -= (n + 1.0) * zcn
        u = (1.0 - zcn)
        u /= (1.0 - zc)
        u *= zc
        res += u 
        res /= zc
        res += n * zcn / zc
        res *=  - zc * zc
        
        tabcols = self.columns.values
        for icol, col in enumerate (tabcols):
            tabcols [icol] = col + fieldsep + 'SENSI'
        res.columns = tabcols
        return TDataSet (res)
        

    def reindex_as_idate (self):
        '''Remplace l'index de dates par une colonne d'entiers calculés à partir des dates'''
        self.index  = self.index.year * 10000 + \
                         self.index.month * 100 + \
                         self.index.day
        self.index.name = 'IDate'

    def time_columns (self):
        '''Calcule les variables de saisonnalité directes et décalées '''
        ds = TDataSet (index = self.index, columns = [glbMetaVarPrefix + 'DATE','MOIS', 'MOIS_', 'JMOIS', 'JMOIS_', 'JSEM', 'JSEM_'])
        ds [glbMetaVarPrefix + 'DATE'] = self.index.year * 10000 + self.index.month * 100 + self.index.day
        ds ['MOIS'] = self.index.month
        ds ['MOIS_'] = (self.index.month + 6) % 12
        ds ['JMOIS'] = self.index.day
        ds ['JMOIS_'] = (self.index.day + 15) % 31
        ds ['JSEM'] = (np.rint (self.index.asi8 / glbNanoSexPerDay)) % 7
        ds ['JSEM_'] = (np.rint (self.index.asi8 / glbNanoSexPerDay) + 3) % 7
        return ds

        
    def sort_index (self, ascending = True, inplace = False):
        '''Trie selon l'index de dates'''
        return TDataSet (data = pd.DataFrame.sort_index (self, ascending = ascending, inplace = inplace))

    def calcRiskReturnMetrics (self, tabperiods, strnavcol = 'NAV_pct', change_in_pct = False, annualize = False):
        '''Calcule les métriques de risque/rendement classiques entre deux dates, pour plusieurs durées.'''
        if strnavcol not in self.columns:
            return self
        ohmaxnav = pd.expanding_max (self [strnavcol])
        bnav_is_std = (str(strnavcol).lower() == 'nav_pct')
        strddcurcol = 'DDCur'
        strddlencol = 'DDLen'
        strwddcol = 'WorstDD'
        strddvol = 'DDVol_'
        strretcol = 'AnnRet_'
        strvolcol = 'AnnVol_'
        strshcol = 'Sh_'
        if not bnav_is_std:
            strddcurcol = strddcurcol +'(' + strnavcol + ')'
            strddlencol = strddlencol +'(' + strnavcol + ')'
            strwddcol = strwddcol +'(' + strnavcol + ')'
            strretcol = strretcol +'(' + strnavcol + ')'
            strvolcol = strvolcol +'(' + strnavcol + ')'
            strshcol = strshcol +'(' + strnavcol + ')'
        if change_in_pct:
            self [strddcurcol] = (self [strnavcol] / ohmaxnav ) - 1
        else:
            self [strddcurcol] = (self [strnavcol] - ohmaxnav )
        ohmaxt = self [strnavcol] [self [strddcurcol] == 0.0]
        self [strddlencol] = 0
        ohddlen = TDataSet (ohmaxt).getDeltaT (1)
        self [strddlencol] =  ohddlen.reindex (index = self.index, method = 'pad')
        self [strwddcol] =  pd.expanding_min (self [strddcurcol])
        for period in tabperiods:
            strRet = strretcol + str(period)
            ohdeltat = TDataSet (data = self).getDeltaT (period)
            #fmeandeltat = ohdeltat ['Dt'].mean ()
            if change_in_pct:
                retcol = self [strnavcol].pct_change (period) 
            else:
                retcol = self [strnavcol].diff (period)
            annretcol = retcol * (365.25 / ohdeltat ['Dt'])
            if annualize:
                self [strRet] = annretcol
            else:
                self [strRet] = retcol   
            strVol = strvolcol + str(period)
            if change_in_pct:
                stdevcol = pd.rolling_std (self [strnavcol].pct_change (1), window = period) 
            else:
                stdevcol = pd.rolling_std (self [strnavcol].diff (1), window = period) 
            #CG 30Jan14
            #pour combler le déficit initial d'écart-type
            stdevcol.fillna (method = 'bfill', inplace = True)
            self [strVol] = stdevcol * math.sqrt(260)    
            ohmaxvol = TDataSet (self [strVol] [self [strddcurcol] == 0.0])
            ohmaxvol = ohmaxvol.reindex (index = self.index, method = 'pad')
            strDDVolP = strddvol + str(period)
            try:
                self [strDDVolP] = 0.0
                self [strDDVolP] = self [strddcurcol] / ohmaxvol
            except:
                pass
            strSh = strshcol + str(period)
            self [strSh] = annretcol / self [strVol]
            #les risques inconnus sont nuls
            self.fillna (value = 0.0, inplace = True)
            
        return self

    def getAmplitudeReport (self, dtstart, dtend, cols):
        #pdb.set_trace ()
        dictreport = {}
        #oslice = self.loc [str(dtstart) : str(dtend)]
        oslice = self.take_columns(cols, forceuppercase = False)
        if oslice is None:
            return None
        else:
            oslice = oslice.take_interval (dtstart, dtend)
        dictreport ['DStart'] = dtstart
        dictreport ['DEnd'] = dtend
        for col in self.columns:
            if col in cols:
                dictreport [col + '_Max'] = oslice [col].max ()
                dictreport [col + '_Min'] = oslice [col].min ()
                dictreport [col + '_Mean'] = oslice [col].mean ()
                dictreport [col + '_Last'] = oslice [col] [-1]
        df =  pd.DataFrame (dictreport, index = [0])
        return TDataSet (df)
    

    #pas utile d'affaiblir par dérivation la version DataFrame.plot
    def ds_plot (self, cols, subplots = False):
        plt.figure ()
        #import pdb; pdb.set_trace()
        pd.DataFrame.plot (self [cols], subplots = subplots)
        #plt.legend(loc='best')
        plt.show ()
        
    def to_map (self, path_save, title, format_fig = '.png', normalize = False) :
    
        try:
            self = self.drop ('TIME_V!MOIS')
        except:
            pass
        del self ['Total']
        #self.fillna(value = 0.0, inplace = True)
    
        # Normalize data columns
        if normalize:
            self_norm = (self - self.mean ()) / (self.max () - self.min ())
        else:
            self_norm = self
        
        # Plot it out
        fig = plt.gcf ()
        ax = fig.add_subplot (111)

        #detection si on a que des valeurs positives ou neg ou les deux
        ## on compte combien de valeur non vides
        nbval = sum (np.sum (pd.notnull (self_norm), axis = 1))
        #les valeurs vides
        #nball = (self_norm.shape[0]) * (self_norm.shape[1])
        #nbnan = nball - nbval
        #combien de valeur positives
        nbbool_pos = sum (np.sum (np.greater (self_norm, 0), axis = 1))
        #combien de valeur negatives
        nbbool_neg = nbval - nbbool_pos
        
        if nbbool_pos == nbval:
            color_cmap = plt.cm.Blues  # @UndefinedVariable
            
        elif nbbool_neg == nbval:
            color_cmap = plt.cm.Reds_r  # @UndefinedVariable
        else:
            color_cmap = 'RdBu'
            
        self_norm.fillna (value = 0.0, inplace = True)
        try:
            heatmap = ax.pcolor (self_norm, cmap = color_cmap, alpha = 0.8)
        except:
            return    
        # Format
        fig.set_size_inches (20, 20)
        plt.subplots_adjust(left = 0.35, bottom = -0.01, right = 0.65, top=  None,
                        wspace = None, hspace = None)
        
        # turn off the frame
        ax.set_frame_on (False)
    
        # put the major ticks at the middle of each cell
        ax.set_yticks (np.arange(self_norm.shape [0]) + 0.5, minor = False)
        ax.set_xticks (np.arange(self_norm.shape [1]) + 0.5, minor = False)
    
        # want a more natural, table-like display
        ax.invert_yaxis ()
        ax.xaxis.tick_top ()
    
        # Set the labels
    
        # labels
        labels = self.columns
        # note I could have used self_sort.columns but made "labels" instead
        ax.set_xticklabels (labels, minor = False)
        ax.set_yticklabels (self_norm.index, minor = False)
        #Taille des étiquettes de l'axe des y
        #ax.tick_params(axis='y', labelsize=6)
        #ax.tick_params(axis='x', labelsize=10)
        fig.suptitle (title, fontsize = 25, fontweight = 'bold')
        #rotate the
        #plt.xticks (rotation = 35)
    
        for t in ax.xaxis.get_major_ticks ():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks ():
            t.tick1On = False
            t.tick2On = False
            
        plt.colorbar(mappable=heatmap, shrink=.5, pad=.2, aspect=10)
        fig.set_size_inches (30,25)
        fig.savefig (path_save + format_fig, dpi=100)
        plt.close(fig)
        return

    def rescale(self, dstart, dend, cols, subserie = None, Tobs = None, inplace = False,  
                newmin = 0, newmax = 1, Nbuckets = 5, method = 'normal', N = None):
        ''' La fonction rescale est une fonction de quantilisation
        dstart = date de début
        dend = date de fin
        cols = colonne(s) à quantiliser
        subserie = serie de réference pour calculer les quantiles (inclus en date dans le self)
        Tobs = date de fin de l'ensemble de référence'''
        
        if inplace:
            newdataset = self
        else:
            newdataset = self.copy () 
        
        cols = self.get_columns (cols)
        Nbuckets_start = Nbuckets
        
        if type(method) == type(True):
            method = 'fix'
        else:
            method = method.lower()
            
        #Taille fixe
        if method == 'fix' and N is None:
            for col in cols:
                Nbuckets = Nbuckets_start
                
                data = newdataset [col].copy()
                #data = data.take_interval (dstart = dstart, dend = dend, inplace = True)
                #ATTENTION: dropna change la taille de data
                #data = data.dropna ()
                
                if Tobs is None:
                    Tobs = data.index[-1]
                
                if subserie is None:
                    subserie = data[data.index <= Tobs]
                
                done = False
                while done is False:
                    try:
                        delta = 1.0 / (Nbuckets + 1)
                        quantile_liste = np.array ( [float(i)*delta for i in range (0, Nbuckets + 2)])
                        
                        #détermination des bornes empiriques
                        bornes = subserie.quantile(quantile_liste)

                        #prévoir de l'espace des deux côtés  
                        bornes [0] = -1e6
                        bornes [-1] = 1e6                                
        
                        newdataset [col] = pd.cut (x = data, bins = bornes.values, labels = False)
                        newdataset [col] *= 1.0 / float (Nbuckets)
                        done = True
                        
                    except ValueError:
                        Nbuckets -= 1
                        
        #Taille grandissante        
        elif method == 'increas':
            for col in cols:
                Nbuckets = Nbuckets_start
                
                data = newdataset [col].copy()
                
                #Rajout des positions de la plus vieille stratégie pour remplacer les 0
                if subserie is None:
                    subserie = data[data.index <= Tobs]
                    
                if subserie is not None:
                    serie = subserie.append(data[data.index > Tobs])
                
                n = len(data[data.index <= Tobs])

                index = serie.index[serie.index >=Tobs] 
                
                k = 0
                delta = 1.0 / (Nbuckets+1)
                quantile_liste = np.array([float(i)*delta for i in range(0, Nbuckets+2)])             
                   
                for t in index:
                    #Nbuckets = Nbuckets_start
                    subserie_new = serie[serie.index <= t]
                             
                    bornes = subserie_new.quantile(quantile_liste)
                    bornes [0] = -1e6
                    bornes [-1] = 1e6 
                    
                    temp = pd.cut (x = data, bins = bornes.values, labels = False)
                    temp /= float(Nbuckets)
                                         
                    if k == 0:
                        newdataset[col].iloc[0:n] = temp.iloc[0:n] #new_col
                        ind = n
                    else:
                        newdataset[col].iloc[ind+k-1] = temp.iloc[ind+k-1] #new_col
                    k = k +1 
        
        elif method == 'fix' is True and N is not None:
            for col in cols:
                Nbuckets = Nbuckets_start
                
                data = newdataset [col].copy()
                
                #Rajout des positions de la plus vieille stratégie pour remplacer les 0
                if subserie is None:
                    subserie = data[data.index <= Tobs]
                    
                if subserie is not None:
                    serie = subserie.append(data[data.index > Tobs])
                
                n = N

#               index = serie.index[serie.index >=Tobs]    
                index = serie.iloc[n:].index
                
                k = 0
                delta = 1.0 / (Nbuckets+1)
                quantile_liste = np.array([float(i)*delta for i in range(0, Nbuckets+2)])
                                    
                for t in index:
                    #Nbuckets = Nbuckets_start
                    #subserie_new = serie[serie.index <= t]
                    subserie_new = serie[serie.index <= t]

                    if k > 0:
                        subserie_new = subserie_new.iloc[-n:]
                                  
                    bornes = subserie_new.quantile(quantile_liste)
                    bornes [0] = -1e6
                    bornes [-1] = 1e6 
                    
                    try:
                        temp = pd.cut (x = data, bins = bornes.values, labels = False)
                        temp /= float(Nbuckets)
                    except:
                        pass
                                    
                    if k == 0:
                        newdataset[col].iloc[0:n] = temp.iloc[0:n] #new_col
                        ind = n
                    else:
                        newdataset[col].iloc[ind+k-1] = temp.iloc[ind+k-1] #new_col
                    k = k +1 
        
        elif method == 'normal':
            if N is None:
                N = 200
                
            for col in cols:
                Nbuckets = Nbuckets_start
                
                data = newdataset [col].copy()
                
                #Rajout des positions de la plus vieille stratégie pour remplacer les 0
                if subserie is None:
                    subserie = data[data.index <= Tobs]
                    
                serie = subserie.append(data[data.index > Tobs])
                
                ewm = pd.ewma (serie, com = N, min_periods = 1) #moyenne mobile exponentielle
                ecartewm = serie - ewm
                
                vol = pd.ewmstd (ecartewm, com = N, min_periods = 1)
            
                zscore = ecartewm.astype('float') / vol
                zscore.fillna(0, inplace = True)
                #discretisation des variables
                ## liste des proba
                prob = np.arange(0, 1 , 1/float(Nbuckets+2))
            
                tab_tranche = [norm.ppf (i) for i in prob] #ppf : Percent point function
                tab_tranche [0] = -1e6
                tab_tranche [-1] = 1e6
            
                newdataset[col] =  pd.cut (zscore, bins = tab_tranche, retbins = True, labels = False)[0] / Nbuckets
                      
        return newdataset[cols]
    
    def value_at_risk (self, alpha = 0.05, horizon = 1, nb_sim = 4000, nb_obs = 10000):
        Loss_q = norm.ppf (alpha)
        df_simu = pd.DataFrame ()
        m, vol = np.mean (self), np.std (self)
        df_var = pd.DataFrame (index = [nb_sim])
        for i in range (nb_sim):
            simu = np.random.normal (m, vol, nb_obs)
            df_simu ['sim'+str(i)] = simu
            loss = df_simu['sim'+str(i)].quantile (alpha)
            df_var ['var'+str(i)] = loss * Loss_q * vol
        VaR = np.sqrt (horizon) * np.mean (df_var.values)
        
        return VaR
    
    def calcSpectrum(self, cols, dtend = None, dtstart = None):
        '''Calculation of the Spectrum criterion.
        Ratio of the left and right sides sum (1/4 of the all) of the square module Fourier transform
        and the full sum of the square module Fourier transform'''        
        spectrum_crit = pd.Series(index = cols)
        spectrum_crit.name = 'Spectrum_crit'
        
        newdataset = self.copy()
        
        if dtend is not None:
            newdataset = newdataset.loc[newdataset.index <= dtend]
        if dtstart is not None:
            newdataset = newdataset.loc[newdataset.index >= dtstart]
        
        N = len(newdataset)
        for col in cols:
            active_vect = (newdataset[col] + 1).fillna(0.0)
            active_f = scipy.fftpack.fft(active_vect)
            
            sum_freq_val = sum(pow(1.0/N * np.abs(active_f),2))
            
            idx = N / (4*2)
            spectrum = sum(2.0 * pow(1.0/N * np.abs(active_f[1:idx]),2)) / sum_freq_val
            
            spectrum_crit[col] = spectrum
            
        return spectrum_crit
    
class TDataSet_Tester:

    _PathDir = Init.glbGetPath()
    _DefaultDataDir = Init.glbMissionPath + '/2 Data/2 Calculs/13 01 25 derived/'
    _DefaultOutDir =  Init.glbMissionPath + '/2 Data/2 Calculs/13 01 25 derived/13 01 25 Matrix/'

    #_incols = [0, _TestFile2 + '_Value', _TestFile3 + '_Value']
    _incols = None
    
    def __init__ (self, inpath = None, outpath = None, data = None, index = None, name = None):
        self._testfiles = []
        self._dataset = TDataSet (data, index)
        self._dataset.name = name
        if inpath is None:
            self._dataset.inputpath = self._DefaultDataDir
        else:
            self._dataset.inputpath = inpath
        if outpath is None:
            self._dataset.outputpath = self._DefaultOutDir
        else:
            self._dataset.outputpath = outpath
        pass

    def test_name (self, name):
        self._dataset.name = name
        
    def add_file_to_list (self, filename):
        self._testfiles.append (filename)

    def add_series_from_file (self, filename):
        self._dataset, _ = self._dataset.add_series_from_file (filename)

    def addSeries_fromlist (self, verbose = True):
        self._dataset, _ = self._dataset.add_series_from_fileslist(self._testfiles, verbose = verbose)

    def addSeries_fromdir (self, path = None, namestart = None, \
                            nameend = None, verbose = True, \
                            timecrit = None, compcrit = None):
        if path is None:
            path = self._dataset.inputpath
        self._dataset = self._dataset.add_series_from_dir (path, namestart = namestart,  \
                                            nameend = nameend, verbose = verbose, \
                                           timecrit = timecrit, compcrit = compcrit)
    def reindex_as_idate (self):
        self._dataset.reindex_as_idate ()
        
    def test_to_csv (self, float_format = None, sep = ','):
        self._dataset.to_csv (float_format = float_format, sep = sep)

    def test_diff (self, cols = None, period = 1):
        #cols = self._dataset.get_columns (cols)
        self._dataset.take_diff (period = period, inplace = True, \
                                 cols = cols, inpct = False)

    def test_pctdelta (self, cols = None, period = 1):
        #cols = self._dataset.get_columns (cols)
        pdb.set_trace()
        self._dataset.take_diff(period = period, inplace = True, \
                                cols = cols, inpct = True)        
    
    def test_ema (self, cols = None, emadecay = 0.99, wres = True, \
                  normalize  = True):
        #import pdb; pdb.set_trace()
        #cols = self._dataset.get_columns (cols)
        self._dataset.take_ema (emadecay = emadecay, inplace = True, \
                                cols = cols, wres = wres, normalize = normalize)

    def test_lag (self, lag = 1, freq = 'B', cols = None):
        #cols = self._dataset.get_columns (cols)
        self._dataset.apply_lag (lag = lag, freq = freq, inplace = True,
                                 cols = cols)

    def test_vol (self, window = 20, cols = None):
        #cols = self._dataset.get_columns (cols)
        #import pdb; pdb.set_trace()
        self._dataset.take_vol (inplace = True, window = window, cols = cols)

    def test_cat (self, quanti = False, levels = 2, cols = None):
        #cols = self._dataset.get_columns (cols)
        #import pdb; pdb.set_trace()
        self._dataset.categorize (inplace = True, quantilize = quanti, \
                                  levels = levels, cols = cols)
        

    def test_combi (self, col1 = 0, c1 = 1, col2 = 1, c2 = -1, islinear = False):
        self._dataset.take_combi (idx1 = col1, coeff1 = c1, idx2 = col2, \
                                          coeff2 = c2, inplace = True, islinear = islinear)
            
    def test_plot (self, cols = None):
        plt.figure ()
        #columns = self._dataset.get_columns (cols)
        #self._dataset [columns].plot (subplots = True)
        self._dataset.plot (subplots = True)
        plt.show ()
