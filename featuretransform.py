
import pandas as pd
import numpy as np
from pipeline import *

from importlib import reload
#pdtrain, pdtest = cleanTrain()
#little = pdtrain[:1000000]

def timetransform(pdtrain) :


    print('#### Starting...... ####')
    pdtrain = pdtrain.sort_values(by=['fecha_dato', 'ncodpers'])
    col_list = pdtrain.columns

    start_index = 0
    for field in col_list:
        start_index = start_index + 1
        if field == 'segmento':
                break
    fields_list = col_list[start_index:]

    print('#### Transfrom 0s to -1s for all entries ####')
    # Transform 0's to -1's for all entries
    for field in fields_list:
        print('field type of :' + field)
        print(type(pdtrain[field][1]))
        selector = pdtrain[field] == 0
        pdtrain.loc[selector, field] = -1

    print('#### Divide by months ####')
    # Divide by months
    months = pdtrain['fecha_dato'].unique()
    databymonth = []
    for month in months:
        print(month)
        selector = pdtrain['fecha_dato'] == month
        # Isolate -1s and 1s and append
        databymonth.append(pdtrain[selector])

    print('#### Ripple effects through time.... ####')
    # Ripple Effects through the months
    #print(len(databymonth))
    #print(type(databymonth))
    month_index = 0
    last_index = pdtrain.shape[1] - 1
    transformedmonths = []
    for month in months:
        print(month)
        if month_index == 0:
            print('#### Skipping month 0 ####')
            transformedmonths.append(databymonth[month_index])
            month_index = 1
            continue
        print('#### Transforming month' +  str(month_index) + ' ####')
        print(len(transformedmonths))
        lastmonth = transformedmonths[month_index -1]

        thismonth = databymonth[month_index]
        origshape = str(thismonth.shape)

        lastmonthclients = lastmonth['ncodpers'].unique()

        thismonthclients = thismonth['ncodpers'].unique()

        totransform = np.in1d(thismonthclients, lastmonthclients)
        tokeepforlast = np.in1d(lastmonthclients, thismonthclients)
        clientstotransform = thismonthclients[totransform]
        clientstokeep = lastmonthclients[tokeepforlast]

        selector = (np.in1d(thismonth['ncodpers'], clientstotransform))
        selectorlast = (np.in1d(lastmonth['ncodpers'], clientstokeep))

        matchedthis = (thismonth[selector].sort_values(by='ncodpers')).drop_duplicates(subset='ncodpers')
        #print('matchedthis shape:' + str(matchedthis.shape) + 'In month' + str(month_index))
        matchedlast = lastmonth[selectorlast].sort_values(by='ncodpers').drop_duplicates(subset='ncodpers')
        #print('matchedlast shape:' + str(matchedlast.shape))

        thisfeatures, thistargets = split(matchedthis)
        lastfeatures, lasttargets = split(matchedlast)

        #print('thistargets shape' + str(thistargets.shape))
        #print('lasttargets shape' + str(lasttargets.shape))

        thismatrix = thistargets.as_matrix()
        lastmatrix = lasttargets.as_matrix()
        #print(lastmatrix)
        added = pd.DataFrame(data=(np.add(thismatrix, lastmatrix)), columns=fields_list)
        #print('added shape' + str(added.shape))
        #print('thisfeatures shape' + str(thisfeatures.shape))

        features = thisfeatures.reset_index()
        del features['index']
        targets = added.reset_index()
        del targets['index']

        #print(targets.head())

        objs = [features, targets]
        transformed = (pd.concat(objs, axis=1))
        print('transformed shape:' + str(transformed.shape))
        print(transformed.head())

        nottransformed = thismonth[~selector]

        nottransformed = nottransformed.reset_index()
        transformed = transformed.reset_index()

        del transformed['index']
        del nottransformed['index']

        objs = [ transformed, nottransformed]

        thismonth = pd.concat(objs, axis=0)
        print('returning....')

        #print(thismonth.head())

        print('thismonth shape' + str(thismonth.shape))
        print('origthismonth shape' + origshape)

        thismonth = thismonth.reset_index()
        del thismonth['index']

        transformedmonths.append(thismonth)

        month_index = month_index + 1


    finaldf = pd.concat(transformedmonths, axis=0)
    print('finaldf shape:' + str(finaldf.shape))
    print(finaldf.head())
    return finaldf
