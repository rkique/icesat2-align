# Original LidarShift script written by Aaron Everett - Bureau of Economic Geology - The University of Texas at Austin.
# Modified by Eric Xia - Brown University.
# This script removes the CRS 32616 conversion with Arcpy, allowing it to run on non-Windows systems.

# It can compare the following data:

# -i: A filepath leading to a fixed width text file for the Chiroptera data, with columns:
# Chiroptera_GPS_Time    UTM Easting    UTM Northing    Height

# -si: A filepath leading to a fixed width text file for the ATL data, with columns:
# UTM Easting   UTM Northing    Height.

# The current configuration does not use fixed-width text files to represent the Chiroptera data.
# Instead, it uses a pickle file (representing a Pandas dataframe), which is much faster.
# It reads in CHIROP_PATH as a .pkl file, which has the same parameters as above (t,x,y,z).

from cmath import isnan
from ctypes import ArgumentError
from re import I
from statistics import mean
import pandas as pd
import datetime
import os
import argparse
import io
from csv import writer as csvWriter
import math
import numpy as np
#Need this extra import to provide a local reference to numpy.power when we call it from a dataframe.query
#Apparently, you can't do np.power() in a df.query
from numpy import power as nppower
import datetime
from numba import jit
from scipy import stats
import multiprocessing
import traceback

RMSE = []
#ATL03 Spot size (meters)
ATL07_SPOT_SIZE = 6
IN_BOUNDS_SAMPLE_RATE = 1
CHIROP_PATH = 'data/Chiroptera_1B/0m-4m.pkl'

ATL07_SPOT_SIZE_SELECTION_STRING = "{0} Meters".format(ATL07_SPOT_SIZE)

#The character to use to split each line
SPLIT_CHAR=','


def shiftPoints(df, driftXVel, driftYVel, chiropteraTotalTime, chiropteraT0, chiropteraT1, atl07T0, atl07T1, atl07TotalTime, sameDirection):
    """
    Shifts the points in the given numpy arrays based on the time, speed and direction paraemters.
    """
    
    t = df['t']
    x = df['x']
    y = df['y']

    chiropteraTimeFraction = (t - chiropteraT0) / chiropteraTotalTime

    atl07TimeFraction = 0

    if sameDirection:
        atl07TimeFraction = chiropteraTimeFraction
    else:
        atl07TimeFraction = 1 - chiropteraTimeFraction

    atl07TimeOnTarget = atl07T0 + (atl07TotalTime * atl07TimeFraction)

    #Delta T is the time difference between Chiroptera actually hitting this spot and the ATL07 "hypothetically" hitting it.
    deltaT = t - atl07TimeOnTarget

    #Compute X and Y components of our ice drift vector
    dx = deltaT * driftXVel
    dy = deltaT * driftYVel 
    #Compute the final X and Y position of our point, adjusted for reversal of the sea ice drift.           

    xFinal = x - dx
    yFinal = y - dy

    #Compute a time that represents when IceSAT2 would have hit this point, if it had a Chiroptera installed on it.
    tFinal = t + deltaT
    
    #sets t,x,y
    df['t'] = tFinal
    df['x'] = xFinal
    df['y'] = yFinal

    #This is a shift per point. 
    #This returns the last chiropteraDeltaX, DeltaY in the correct direction.
    dx0 = dx.iloc[0]
    dx1 = dx.iloc[-1]
    dy0 = dy.iloc[0]
    dy1 = dy.iloc[-1]

    return ((dx0, dx1), (dy0, dy1))

def shiftFeatureClass(df, driftMetersPerSecond, driftBearing, chiropteraT0, chiropteraT1, atl07T0, atl07T1, atl07TotalTime, sameDirection):
    
    """
    Performs a shift with the specified parameters on the dataframe.
    """
    
    #print(f"[INFO] Working on shift: {driftMetersPerSecond} m/s with bearing {driftBearing} degrees")
    #Convert our drift bearing to radians, relative to due east.
    driftPolarDir = 360 + (90.0 - driftBearing)
    driftPolarDirRad = math.radians(driftPolarDir)
    
    #Calculate the X and Y components of our drift velocity vector
    driftXVel = math.cos(driftPolarDirRad) * driftMetersPerSecond
    driftYVel = math.sin(driftPolarDirRad) * driftMetersPerSecond
        
    chiropteraTotalTime = chiropteraT1 - chiropteraT0    

    timeBefore = datetime.datetime.now()

    (shiftX, shiftY) = shiftPoints(df, driftXVel, driftYVel, chiropteraTotalTime, chiropteraT0, chiropteraT1, atl07T0, atl07T1, atl07TotalTime, sameDirection)
    
    timeAfter = datetime.datetime.now()
    
    timeDiff = timeAfter - timeBefore
    #ret should be shifted feature class
    
    return (df, shiftX, shiftY)

def meanChiropteraForATL(chiropteraDataFrame, atlX, atlY, spotSize):
    """
    Determines the elevation to be used as the corresponding "chiroptera elevation" for the given ATLAS sensor point.
    This works by first querying the dataframe for any points falling within the spot size radius of the given X and Y, which theoretically represent the center of the area illuminated by the ATLAS sensor's laser.
    """
    
    #shape1 = chiropteraDataFrame.shape
    #shape2 = relevantPoints.shape

    relevantPoints = chiropteraDataFrame.query("sqrt(@nppower((`x` - @atlX), 2) + @nppower((`y` - @atlY), 2)) <= @spotSize")
    #print(len(relevantPoints))
    #for index,row in chiropteraDataFrame.iterrows():
    #    print(f"dist: {np.sqrt(((row['x'] - atlX)**2 + (row['y'] - atlY)**2))}")
    maxZ = relevantPoints["z"].mean()

    return maxZ

def compareToATL07(chiropteraDataFrame, atl03DataFrame):
    """
    Performs the comparison of the given set of chiroptera data to the given set of ATLAS sensor data.
    """

    #print(f"first: {atl03DataFrame.iloc[0]['SHAPE@X']} minX {cMinX} maxX {cMaxX} minY {cMinY} maxY {cMaxY}")

    cMaxX = chiropteraDataFrame["x"].max()
    cMinX = chiropteraDataFrame["x"].min()
    cMaxY = chiropteraDataFrame["y"].max()
    cMinY = chiropteraDataFrame["y"].min()

    #we'd like to do this very quick for many directions...
   
    atl03InBoundsDataFrame = atl03DataFrame[(atl03DataFrame['X'] >= cMinX) & 
                               (atl03DataFrame['X'] <= cMaxX) & 
                               (atl03DataFrame['Y'] >= cMinY) & 
                               (atl03DataFrame['Y'] <= cMaxY)]
    
    #print(f"atl07InBounds has length {len(atl03InBoundsDataFrame)} ")

    chiropteraElevs = []
    atl03Elevs = []

    #this can take a while... calcChiroptera performs a mean over a spot for each corresponding z value in bounds.
    #one approach is to take every Nth Z value...
    for ind, row in atl03InBoundsDataFrame.iterrows():

        if ind % IN_BOUNDS_SAMPLE_RATE  == 0:

            x = row["X"]
            y = row["Y"]
            z = row["Z"]
            #print(f"testing {x,y}")
            chiropteraElev = meanChiropteraForATL(chiropteraDataFrame, x, y, ATL07_SPOT_SIZE)
            #print(f"chiropteraElev: {chiropteraElev}")

            if not isnan(chiropteraElev) and not isnan(z):
                chiropteraElevs.append(chiropteraElev)
                atl03Elevs.append(z)

    return chiropteraElevs, atl03Elevs

def processWithSpeedAndBearing(argSet):

    """
    Performs the iterative process of first shifting the given chiroptera data by the given speed and distance parameters, and then performs the comparison and linear regression operations.
    This function takes its parameters as a dictionary because it is meant to be launched via multiprocessing library.
    """
    
    ret = {}

    try:
    
        curSpeed = argSet["curSpeed"]
        curBearing = argSet["curBearing"]
        chiropteraDfArray = argSet["chiropteraDfArray"]
        atl07DataFrame = argSet["atl07DataFrame"]
        chiropteraT0 = argSet["chiropteraT0"]
        chiropteraT1 = argSet["chiropteraT1"]
        atl07T0 = argSet["atl07T0"]
        atl07T1 = argSet["atl07T1"]
        sameDirection = argSet["sameDirection"]

        atl07TotalTime = atl07T1 - atl07T0
        loopStartTime = datetime.datetime.now()
        
        driftMetersPerSecond = curSpeed / 60.0
    
        #Perform the time/speed/distance derived geometric shift of the chiroptera data, using the given parameters.
        
        shiftedChiroptera, shiftX, shiftY = shiftFeatureClass(chiropteraDfArray,driftMetersPerSecond, curBearing, chiropteraT0, chiropteraT1, atl07T0, atl07T1, atl07TotalTime, sameDirection)

        chiropteraZs, atl07Zs = compareToATL07(shiftedChiroptera, atl07DataFrame)
        
        # print(f"in bounds points for {curBearing} {driftMetersPerSecond} is {len(chiropteraZs)}")
        dfZs = pd.DataFrame(columns=["c","a"])
        dfZs["c"] = chiropteraZs
        dfZs["a"] = atl07Zs
        exportShiftedDataPath = "data/z_compare.csv"
        dfZs.to_csv(exportShiftedDataPath, index=False, header=False)
        #zsFullPath = "data/{0}_zs.csv".format(fcName)
        #dfZs.to_csv(zsFullPath)


        if len(chiropteraZs) > 0:

            #could add here..
            points_compared = len(chiropteraZs)
            dfZs = pd.DataFrame(columns=["c","a"])
            dfZs["c"] = chiropteraZs
            dfZs["a"] = atl07Zs

            #zsFullPath = "data/{0}_zs.csv".format(fcName)
            #dfZs.to_csv(zsFullPath)

            slope, intercept, r_value, p_value, std_err = stats.linregress(atl07Zs, chiropteraZs)
            rsquared = r_value ** 2
            rmse = np.mean(np.sqrt(np.mean(np.array(atl07Zs) - np.array(chiropteraZs))**2))

            loopEndTime = datetime.datetime.now()
            # <class 'int'> <class 'float'> <class 'float'> <class 'pandas.core.series.Series'> <class 'pandas.core.series.Series'> <class 'numpy.float64'> <class 'numpy.float64'>

            #print("Types:", type(points_compared), type(curBearing), type(driftMetersPerSecond*60), type(shiftX), type(shiftY), type(rmse), type(rsquared))

            with open("data/output_log.txt", "a") as file:
                file.write(f"{points_compared},{curBearing},{(driftMetersPerSecond*60)},{shiftX},{shiftY},{rmse},{rsquared}\n")
            shiftX = np.round(shiftX, 3)
            shiftY = np.round(shiftY, 3)
            print(f"({points_compared} pts) bearing:{curBearing:.0f} speed:{(driftMetersPerSecond*60):.3f} X:{shiftX} Y:{shiftY} RMSE:{rmse:.5f} rsquared:{rsquared:.5f}")

            #["driftBearing", "driftSpeed", "shiftX", "shiftY", "rmse", "intersectCount"]
            ret = [curBearing, driftMetersPerSecond, shiftX, shiftY, rmse, r_value, rsquared, points_compared]

        else:
            shiftX = np.round(shiftX, 3)
            shiftY = np.round(shiftY, 3)
            print(f"(no points compared) for shift ({shiftX}, {shiftY})")
    except Exception as e:
        print(e)
        
        st = traceback.format_exc()
        
        print(st)
        exit()
    
    return ret
    
def main():
    #Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, help="Path to chiroptera data file")
    parser.add_argument("-sif", type=str, required=True, help="Path to ATL data file")
    parser.add_argument("-si", type=float, required=True, help="Lowest drift speed in meters per *minute*")
    parser.add_argument("-sf", type=float, required=True, help="Highest drift speed in meters per minute")
    parser.add_argument("-ss", type=float, required=True, help="Drift speed step in meters per minute")
    parser.add_argument("-bi", type=float, required=True, help="Initial drift bearing in degrees")
    parser.add_argument("-bf", type=float, required=True, help="Final drift bearing in degrees")
    parser.add_argument("-bs", type=float, required=True, help="Drift bearing step")
    parser.add_argument("-t0", type=float, required=True, help="Section start time in GPS week seconds for ATL07")
    parser.add_argument("-t1", type=float, required=True, help="Section end time in GPS week seconds for ATL07")
    parser.add_argument("-m", type=str, required=True, help="Number of parallel instances to run")
    parser.add_argument("-sd", type=str, required=True, help="Whether satellite and aircraft pass are in the same direction.  y for same direction, n for opposing directions")

    parser.add_argument("-e", type=float, default=0, required=False, help="Optional extra fixed drift along bearing.  Default 0.")

    args = parser.parse_args()

    iSensorType = 0

    #Input file path
    inputDataPath = args.i

    #Drift speed in meters per minute - this is converted to m/s later
    initialDriftMetersPerMinute = args.si
    finalDriftMetersPerMinute = args.sf
    driftSpeedStep = args.ss

    initialDriftBearing = args.bi
    finalDriftBearing = args.bf
    driftBearingStepSize = args.bs

    if initialDriftBearing < 0 or initialDriftBearing > 360:
        raise ArgumentError("Initial drift bearing is out of bounds.  Must be between 0 and 360.")
    
    if finalDriftBearing < 0 or finalDriftBearing > 360:
        raise ArgumentError("Final drift bearing is out of bounds.  Must be between 0 and 360.")
    
    if initialDriftBearing > finalDriftBearing:
        raise ArgumentError("Initial drift bearing must not be less than final drift bearing.")
    
    if driftSpeedStep <= 0:
        raise ArgumentError("Drift speed step must be greater than zero.") 
    
    #An extra arbitrary adjustment in the drift bearing direction (in m), to allow adjustment for the limitations in accuracy of this approach.
    #We can get in the ballpark by mathematically reversing the drift, but 
    #but what ..... 
    extraDrift = args.e

    #Whether or not we need to export the full shifted data.

    atl07T0 = -1
    atl07T1 = -1

    sameDirection = False
    
    if args.sd == "y":
        sameDirection = True

    #Beginning and end time of IceSAT2 overflight of this area (in GPS seconds)
    atl07T0 = args.t0
    atl07T1 = args.t1

    startTime = datetime.datetime.now()

    print("Starting up at {0}".format(startTime))

    speed_iter = (finalDriftMetersPerMinute - initialDriftMetersPerMinute) / driftSpeedStep
    angle_iter = (finalDriftBearing - initialDriftBearing) / driftBearingStepSize
    print(f"Total iterations to check at {IN_BOUNDS_SAMPLE_RATE} stride: {speed_iter * angle_iter}")
    print("Ingesting ATL07 & Chiroptera data")
    
    with open("data/output_log.txt", "a") as file:
        file.write(f"Begin {inputDataPath} speeds: {initialDriftBearing} {finalDriftBearing} angles: {initialDriftBearing} {finalDriftBearing}")
    
    chirop_names = ["t", "x", "y", "z"]
    atl_names =["X", "Y", "Z"]

    chiropteraDfArray = pd.read_pickle(CHIROP_PATH)
    chiropteraDfArray.columns = chirop_names
    atl07DataFrame = pd.read_fwf(args.sif, names=atl_names)
    
    chiropteraT0 = chiropteraDfArray.iloc[0]["t"]  
    chiropteraT1 = chiropteraDfArray.iloc[-1]["t"]
    print(f"T0: {chiropteraT0} T1: {chiropteraT1}")

    curBearing = initialDriftBearing
    curSpeed = initialDriftMetersPerMinute

    bearingStep = driftBearingStepSize

    dfParamSets = pd.DataFrame(columns=["driftBearing", "driftMetersPerSecond", "shiftX", "shiftY", "rmse", "r_value", "rsquared", "intersectCount"])

    runArgSets = []

    #Iterate over bearing steps
    while curBearing <= finalDriftBearing:
        #Iterate over speed steps.
        while curSpeed <= finalDriftMetersPerMinute:
        
            #Compose a set of arguments to be used for this iteration.  Note that we'll actually launch this later.
            runArgSet = {
                "curSpeed": curSpeed,
                "curBearing": curBearing,
                "chiropteraDfArray": chiropteraDfArray,
                "atl07DataFrame": atl07DataFrame,
                "chiropteraT0": chiropteraT0,
                "chiropteraT1": chiropteraT1,
                "atl07T0": atl07T0,
                "atl07T1": atl07T1,
                "sameDirection": sameDirection
            }
        
            runArgSets.append(runArgSet)

            #Increment our speed.
            curSpeed = curSpeed + driftSpeedStep
    
        #Increment our bearings
        curBearing = curBearing + bearingStep
        curSpeed = initialDriftMetersPerMinute

    # allArgSets = []
    # curSpeed = 1.5
    # while curBearing <= 360:
    #     while curSpeed <= 5:
    #         runArgSet = {
    #             "curSpeed": curSpeed,
    #             "curBearing": curBearing,
    #             "inputSatelliteDataPath": inputSatelliteDataPath,
    #             "chiropteraDfArray": chiropteraDfArray,
    #             "outputDataPath": outputDataPath,
    #             "atl07DataFrame": atl07DataFrame,
    #             "chiropteraT0": chiropteraT0,
    #             "chiropteraT1": chiropteraT1,
    #             "atl07T0": atl07T0,
    #             "atl07T1": atl07T1,
    #             "sameDirection": sameDirection,
    #             "exportShiftedData": exportShiftedData
    #         }
    #         allArgSets.append(runArgSet)
    #         curSpeed += 0.1
    #     curBearing += 10

    # multi-processing, and sequential


    if int(args.m) > 1:
        #Multi-processing code path
        #requires use of multiprocessing.Pool
        finalMulti = int(args.m)
        p = multiprocessing.Pool(finalMulti)
        resultRows = p.map(processWithSpeedAndBearing, runArgSets)
        #removed call to sort_index

        for resultRow in resultRows:
            dfParamSets.loc[-1] = resultRow
            dfParamSets.index = dfParamSets.index + 1

    else:
        #Sequential version.  Slower, but reliable.
        for argSet in runArgSets:
            resultRow = processWithSpeedAndBearing(argSet)
        
            dfParamSets.loc[-1] = resultRow
            dfParamSets.index = dfParamSets.index + 1
        
    #Write out the RMSE sorted results report file.
    dfParamSets = dfParamSets.sort_values(by='rsquared')
    paramSetsCsvPath = 'data/output_results.csv'
    dfParamSets['driftMetersPerMinute'] = dfParamSets['driftMetersPerSecond'] * 60
    dfParamSets.to_csv(paramSetsCsvPath)

    finishTime = datetime.datetime.now()
    print("Finished at {0}".format(finishTime))
    
    timeDiff = finishTime - startTime

    print("running time is: {0}".format(timeDiff))

if __name__ == '__main__':
	#Run the program
    print("Running lidarshift...")
    main()