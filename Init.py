# -*- coding: cp1252 -*-
####################################################################
#
import os
import socket
import sys, getopt
import pdb
from datetime import datetime
import ConfigParser
import socket
import time

glbLogFile = None
glbLogFilePath = ''
glbVerbose = True
glbVersion = "201409.09"
glbPathDict = {'LT-CGEISSLER':
               {'_PathDir': 'C:/Users/CGEISSLER.QUINTEN/Dropbox/',
                '_ConfigDir': 'C:/Users/CGEISSLER.QUINTEN/Dropbox/',
                '_RootFile': 'glbConfig.csv'},
               'LT-3506':
               {'_PathDir': 'C:/Users/CGEISSLER/Dropbox/',
                '_ConfigDir': 'C:/Users/CGEISSLER/Dropbox/',
                '_RootFile': 'glbConfig.csv'},
            'PAR-DW-J03817':
               {'_PathDir': 'H:/CPE/ADVESTIS/',
                '_ConfigDir': 'H:/CPE/ADVESTIS/',
                '_RootFile': 'glbConfig.csv'},   
            'C-PC':
                 {'_PathDir': 'C:/Users/C/Dropbox/',
                  '_ConfigDir': 'C:/Users/C/Dropbox/',
                  '_RootFile': 'glbConfig.csv'},
            'AdCalcul':
                 {'_PathDir': '/media/advestis/D:/Dropbox/',
                  '_ConfigDir': '/media/advestis/D:/Dropbox/',
                  '_RootFile': 'glbConfig.csv'},    
            'LT-TEST':
               {'_PathDir': 'C:/Users/Raphael/Dropbox/',
                '_ConfigDir': 'C:/Users/Raphael/Dropbox/',
                '_RootFile': 'glbConfig.csv'},
            'NOUR':
               {'_PathDir': 'C:/Users/Noureddine/Dropbox/',
                '_ConfigDir': 'C:/Users/Noureddine/Dropbox/',
                '_RootFile': 'glbConfig.csv'},
            'NBOUMLAIK':
               {'_PathDir': 'E:/Dropbox/',
                '_ConfigDir': 'E:/Dropbox/',
                '_RootFile': 'glbConfig.csv'}, 
            'TOSHIBA_RAF':
               {'_PathDir': 'C:/Users/Administrateur/Dropbox/',
                '_ConfigDir': 'C:/Users/Administrateur/Dropbox/',
                '_RootFile': 'glbConfig.csv'},                 
            'VMARGOT-PC':
               {'_PathDir': 'E:/Dropbox/',
                '_ConfigDir': 'E:/Dropbox/',  
                '_RootFile': 'glbConfig.csv'},
            'PUB-VAIOXP':
                 {'_PathDir': 'C:/Documents and Settings/Administrateur/Mes documents/Dropbox/',
                  '_ConfigDir': 'C:/Documents and Settings/Administrateur/Mes documents/Dropbox/',
                  '_RootFile': 'glbConfig.csv'},
            'DGK7098':
                 {'_PathDir': 'U:/SANCHEZ/',
                  '_ConfigDir': 'D:/Mes Documents/ProOptim/',
                  '_RootFile': 'glbConfig.csv'},
            'DGK7070':
                 {'_PathDir': 'H:/SANCHEZ/',
                  '_ConfigDir': 'D:/Mes Documents/ProOptim/',
                  '_RootFile': 'glbConfig.csv'},
            'DGK7075':
                 {'_PathDir': 'U:/SANCHEZ/',
                  '_ConfigDir': 'D:/Mes Documents/ProOptim/',
                  '_RootFile': 'glbConfig.csv'},
            'DGK7084':
                 {'_PathDir': 'G:/SANCHEZ/',
                 '_ConfigDir': 'D:/Mes Documents/ProOptim/',
                 '_RootFile': 'glbConfig.csv'},
            'DGK7058':
                 {'_PathDir': 'G:/SANCHEZ/',
                 '_ConfigDir': 'D:/Mes Documents/ProOptim/',
                 '_RootFile': 'glbConfig.csv'},
            'DGK7092':
                 {'_PathDir': 'G:/SANCHEZ/',
                 '_ConfigDir': 'D:/Mes Documents/ProOptim/',
                 '_RootFile': 'glbConfig.csv'},
            'DGK7012':
                 {'_PathDir': 'G:/SANCHEZ/',
                 '_ConfigDir': 'D:/Mes Documents/ProOptim/',
                 '_RootFile': 'glbConfig.csv'},               
            'DGK7083':
                 {'_PathDir': 'G:/SANCHEZ/',
                 '_ConfigDir': 'D:/Mes Documents/ProOptim/',
                 '_RootFile': 'glbConfig.csv'},
            'DGK7107':
                 {'_PathDir': 'H:/SANCHEZ/',
                 '_ConfigDir': 'D:/Mes Documents/AdOptim/',
                 '_RootFile': 'glbConfig.csv'},
            'ADCALCUL2':
                 {'_PathDir': '/media/adcalcul2/D/Dropbox/',
                 '_ConfigDir': '/media/adcalcul2/D/Dropbox/',
                 '_RootFile': 'glbConfig.csv'},
            'adcalcul3-All-Series':
                 {'_PathDir': '/media/adcalcul3/D/Dropbox/',
                 '_ConfigDir': '/media/adcalcul3/D/Dropbox/',
                 '_RootFile': 'glbConfig.csv'}            
            }

#le chemin racine de l'arborescence du système (par exemple le répertoire de Dropbox) 
glbRootPath = ''
#le chemin d'accès au fichier global de configuration
glbConfigFilePath = ''
glbConfigFileName = ''
#le répertoire de la mission
glbMissionPath = ''


def glbGetPath ():
    '''Renvoie le chemin racine général en fonction de la machine. '''
    #pdb.set_trace()
    #cname = os.environ ['COMPUTERNAME']
    cname = socket.gethostname ().upper()
    global glbPathDict
    if cname in glbPathDict:
        return glbPathDict [cname] ['_PathDir']
    else:
        return 'D:/Mes Documents/'

def getConfigFilePath ():
    '''Renvoie le répertoire du fichier de configuration général en fonction de la machine. '''
    #pdb.set_trace()
    cname = socket.gethostname ().upper()
    global glbPathDict
    if cname in glbPathDict:
        return glbPathDict [cname] ['_ConfigDir']
    else:
        Log ('Erreur: chemin du fichier de configuration introuvable.')
        sys.exit()
    
def glbGetMissionPath ():
    '''Renvoie le répertoire de la mission lu dans la bonne section du fichier de configuration.'''
    global glbMissionPath
    return glbMissionPath

def openLogFile ():
    #todaystr = time.strftime('%H:%M',time.localtime())
    logfilename = 'Log.csv'
    LogFilePath = glbMissionPath
    LogFile =  open (LogFilePath + logfilename, 'a+')
    return LogFile

def Log (astring, queue = None, doexit = False, Verbose = True):
        
    line = time.strftime('%d/%m/%y %H:%M:%S',time.localtime())
    line = line + ' : ' + astring + '\n'
    
    if queue is not None:
        queue.put (line)
    global LogFile
    
    LogFile = openLogFile ()
            
    LogFile.write (line)
    
    closeLogFile()
    
    if Verbose:
        print line
            
    if doexit:
        os._exit(1)
    else:
        return

def fillPath (path):
    '''Préfixe un chemin relatif (ne commençant pas par /) par le répertoire de la mission. '''
    strpath = str (path)
    if strpath [0] == '/':
        return strpath
    else:
        global glbMissionPath
        if glbMissionPath == '':
            glbInit ()  # @UndefinedVariable
        return glbMissionPath + strpath    

def closeLogFile ():
    global glbLogFile
    if glbLogFile is not None:
        glbLogFile.close ()    

def main (argv):

    try:
        #pdb.set_trace()
        cfgsection = ''
        try:
            opts, args = getopt.getopt(argv,"hp:f:s:",["configpath=", "configfile=", "section="])
        except getopt.GetoptError:
            print 'Init.py -p <path> -f <file> -s <section>'
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print 'Init.py -s <section>'
                sys.exit()
            elif opt in ('-s', '--section'):
                cfgsection = arg.strip()
          
        global glbRootPath
        glbRootPath = glbGetPath()
        global glbConfigFilePath
        glbConfigFilePath = getConfigFilePath ()
        glbZeConfigFileName = 'glbConfig.csv'
        glbConfig = ConfigParser.SafeConfigParser ()
        #Localiser le fichier de configuration de l'application
        #chemin d'accès à ProOptim.cfg
        global glbConfigPath
        try:
            
            #Lire le fichier de configuration global
            glbConfig.read (glbConfigFilePath + glbZeConfigFileName)
            glbConfigPath = glbRootPath + glbConfig.get (cfgsection, 'glbconfigpath')
            global glbConfigFileName
            glbConfigFileName = glbConfig.get (cfgsection, 'glbconfigfile')
            #Localiser le fichier de log
            global glbLogFilePath
            glbLogFilePath = glbRootPath + glbConfig.get (cfgsection, 'glblogfilepath')
            #chemin racine des données de la mission
            global glbMissionPath
            glbMissionPath = glbRootPath + glbConfig.get (cfgsection, 'glbmissionpath')
            #chemin racine des codes source
            glbPySourcePath = glbConfig.get (cfgsection, 'glbpysourcepath')
            global glbVerbose
            glbVerbose = eval (glbConfig.get (cfgsection, 'glbverbose'))
            sys.path.append(glbPySourcePath)
            sys.path.append ('../' + glbPySourcePath)
        except:
            try:
                import pandas as pd
                glbZeConfigFileName = 'glbConfig.csv'
                #Lire le fichier de configuration global
                glbConfig = pd.read_csv (glbConfigFilePath + glbZeConfigFileName, index_col = 0)
                #Localiser le fichier de configuration de l'application
                #chemin d'accès à ProOptim.cfg
                global glbConfigPath
                glbConfigPath = glbRootPath + glbConfig ['glbconfigpath'].loc [cfgsection]
                global glbConfigFileName
                glbConfigFileName = glbConfig ['glbconfigfile'].loc [cfgsection]
                #Localiser le fichier de log
                global glbLogFilePath
                glbLogFilePath = glbRootPath + glbConfig ['glblogfilepath'].loc [cfgsection]
                #chemin racine des données de la mission
                global glbMissionPath
                glbMissionPath = glbRootPath + glbConfig ['glbmissionpath'].loc [cfgsection]
                #chemin racine des codes source
                glbPySourcePath = glbConfig ['glbpysourcepath'].loc [cfgsection]
                global glbVerbose
                try:
                    glbVerbose = eval (glbConfig ['glbverbose'].loc [cfgsection])
                except:
                    glbVerbose = glbConfig ['glbverbose'].loc [cfgsection]
                    
                sys.path.append(glbPySourcePath)
                sys.path.append ('../' + glbPySourcePath)
            
            except:
                Log ('Erreur: ' + str (sys.exc_info()[1]))
            finally:
                closeLogFile ()
    except:
        Log ('Erreur: ' + str (sys.exc_info()[1]))
    finally:
        closeLogFile ()
        

def load_header ():
    head = r'''
    \documentclass{article}
    \usepackage[pdftex]{geometry}
    \geometry{paperheight=400mm, paperwidth=300mm}
    \geometry{portrait}%, nohead, nofoot}
    \geometry{hmargin=15mm,vmargin=12mm}
    \usepackage{tabularx}
    \usepackage{booktabs}
    \usepackage[latin1]{inputenc}  
    \usepackage[T1]{fontenc}  
    \usepackage{graphicx}  
    \usepackage{sidecap}
    
    \title{Rules Reporting}
    \date{\today}
    
    \begin{document}
    
    \maketitle
    
    '''
    return head


                
#pdb.set_trace()
if __name__ == "__main__":
    main(sys.argv[1:])
else:
    main(['-s global'])