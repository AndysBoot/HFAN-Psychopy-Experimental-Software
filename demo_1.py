#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.2.4),
    on October 28, 2026, at 15:14
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

"""
Author: Andy Sosa
Contact: andysosa2007@gmail.com
Date: 11/9/2026
Version 1.5
Description: This script was created for the HFAN Research Laboratory. It currently allows the user to see dicom files,
change the anatomical plane, zoom, window position, and window/level of the filer. The next version will allow ther user to properly guess,
where they believe the nodule is located.

"""

# --- Import packages ---
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from psychopy.tools.monitorunittools import convertToPix
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code
import os
import glob
import pydicom
from psychopy import visual
from psychopy import core
from psychopy.hardware import keyboard
from psychopy.event import Mouse

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.2.4'
expName = 'scroll_wheel_test'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = 'data/' + expInfo['participant'] + '_' + expName + '_' + expInfo['date']
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\andys\\Documents\\psychopy_scripts\\scroll_wheel_test.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "trial" ---
    dicom_display = visual.ImageStim(
        win=win,
        name='dicom_display', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.4, 0), draggable=False, size=(0.8, 0.8),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    displayInfo = visual.TextStim(win=win, name='displayInfo',
        text='',
        font='Arial',
        units='height', pos=(0.5, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "trial" ---
    
    
    
    
    
    
    
    # create an object to store info about Routine trial
    trial = data.Routine(
        name='trial',
        components=[dicom_display, displayInfo],
    )
    trial.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code
    

    #-------------------------------------------------------------------------------
    # RANDOM PATIENT SELECTOR AND DICOM VALIDATOR
    #-------------------------------------------------------------------------------
    
    #File Directory to where the dataset is located. It is very important to ensure root_dir ir set to the proper folder destination in order to run the program
    # This script has only been tested with the LIDC Cancer and Imaging dataset for CT Scans
    
    root_dir = r'C:\Users\Public\Downloads\lidc_idri'
    #gathers all valid patient directories and filters out unecessary folders/files
    patient_dirs = [
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('LIDC-IDRI-')
    ]
    if not patient_dirs:
        raise FileNotFoundError(f"No LIDC-IDRI patient folders found in {root_dir}")
    
    #shuffles all the collected folders using numpy
    shuffled_dirs = list(patient_dirs)
    np.random.shuffle(shuffled_dirs)

    folder = None
    series_files = []
    xml_files = []

    for candidate_dir in shuffled_dirs:
        #finds all dicom files within the subfolders
        cand_dcm = glob.glob(os.path.join(candidate_dir, '**', '*.dcm'), recursive=True)
        if not cand_dcm:
            continue
        #once all dicom files are found within the sub folder, we group them together by their SeriesInstanceUID
        cand_series = defaultdict(list)
        for fp in cand_dcm:
            try:
                #this is fairly important for optimization as it ensures we only look at the metadata 
                # rather then loading the entire image each time
                h = pydicom.dcmread(fp, stop_before_pixels=True)
                #this makes sure that we only keep 3d ct slices(filters out everything else like 2d chest x-rays and etc)
                if getattr(h, 'Modality', '') == 'CT' and hasattr(h, 'ImagePositionPatient'):
                    cand_series[h.SeriesInstanceUID].append(fp)
            except Exception:
                continue

        if cand_series:
            #if multiple candidates exist within the CT series(a scan with 2 slices vs one with 100), we return the largest one)
            best_series = max(cand_series.values(), key=len)
            if len(best_series) > 10: # we require 10 slices to make a stackable ct scan(number could be modified to your liking)
                folder = candidate_dir
                series_files = best_series
                # we look for the related xml file(IMPORTANT)
                xml_files = glob.glob(os.path.join(candidate_dir, '**', '*.xml'), recursive=True)
                break

    if folder is None or not series_files:
        raise RuntimeError(f"Checked {len(patient_dirs)} folders in {root_dir}, but none contained a valid 3D CT scan.")
    
    #console logging
    patient_id = os.path.basename(folder)
    print(f"Selected Patient: {patient_id}")


    #-------------------------------------------------------------------------------
    # 3D SLICING PLANE HELPER, WINDOW/LEVEL PRESETS, CENTRAL STATE VARIABLES
    #-------------------------------------------------------------------------------
    
    # extracts the 3d numpy array into a 2d plane depending on the plane given(a,s,c)
    # we return True to call .flip, to ensure proper orientation when displaying ct scans
    def get_slice():
        x, y, z = state['cursor']
        if state['plane'] == 'axial':    return vol[z, :, :], True    # flip: row 0 = anterior -> top
        if state['plane'] == 'coronal':  return vol[:, y, :], True   # row 0 = inferior -> bottom
        return vol[:, :, x], True                                    # sagittal

    # returns number of slices across the x,y,z plane(this is used to prevent the user from scrolling out of bounds)
    def plane_count():
        return {'axial': Z, 'coronal': Y, 'sagittal': X}[state['plane']]
    
    #presets for optimal window/level values when trying to enhance certain features of the CT scan
    PRESETS = [
        ('Lung',        1600, -600),
        ('Mediastinum',  500,   50),
        ('Soft Tissue',  350,   50),
        ('Liver',        160,   60),
        ('Bone',        2000,  500),
        ('Brain',         70,   30),
        ('Subdural',     200,   75),
        ('Stroke',        30,   30),
    ]

    #IMPORTANT
    # This is the central dictionary used when keeping track of important parameters throghout the program
    # Any important features should be added here to keep readibility.
    state = {
        'patient_id': patient_id,
        'plane' : 'axial',
        'center': 0.0,
        'width': 1.0,
        'invert': False,
        'ds': None,
        'default_center': 0.0,
        'default_width': 1.0,
        'preset' : 'File default',
        'zoom' : 1.0,
        'pan' : [0, 0],
        'sop_to_z': {},
        'nodules': [],
        'current_guess': None,
        'show_truth': False,
        'trial_result' : None,
    }
    #used to prevent crashing as python cant handle a float call on multiple variables
    # henceforth, we return the first values as a float if possible.
    def to_float(v, default):
        if v is None:
            return float(default)
        try:
            if hasattr(v, '__iter__') and not isinstance(v, str):
                v = list(v)[0]
            return float(v)
        except (TypeError, ValueError, IndexError):
            return float(default)



    #-------------------------------------------------------------------------------
    # CONSTRUCTING 3D NUMPY ARRAY, XML ANNOTATION, DISTANCE CALCULATOR
    #-------------------------------------------------------------------------------
    
    def build_volume(paths):
        #loads full dicom dataset
        dss = [pydicom.dcmread(p) for p in paths]
        #sort the slices using the z coordinate, from bottom to top
        dss.sort(key=lambda d: float(d.ImagePositionPatient[2]))
        #stacks 2d slices into 3d array and converst values into Hounsfield Units(HU)
        vol = np.stack([d.pixel_array.astype(np.float32) * float(d.get('RescaleSlope', 1))
                        + float(d.get('RescaleIntercept', 0)) for d in dss])
        zs = np.array([float(d.ImagePositionPatient[2]) for d in dss])
        #computes the physical z-spacing (mm between each slice)
        z_sp = float(np.median(np.abs(np.diff(zs)))) if len(zs) > 1 else float(dss[0].get('SliceThickness', 1))
        #we extract the milimeter spacing
        row_sp, col_sp = [float(v) for v in dss[0].PixelSpacing]
        origin = np.array(dss[0].ImagePositionPatient, dtype=float)
        #we create the lookup table to map each slice's UID, to its associated annotated XML file
        sop_to_z = {str(d.SOPInstanceUID).strip(): idx for idx, d in enumerate(dss)}
        
        return vol, dict(x=col_sp, y=row_sp, z=z_sp), origin, dss[0], sop_to_z
    
    # we now build the 3d volume and unpack the 3 dimensions
    vol, spacing, origin, ds0, sop_to_z = build_volume(series_files)
    Z, Y, X = vol.shape
    
    #initialize state with volume metadata and center cursor at the origin
    state['ds'] = ds0
    state['sop_to_z'] = sop_to_z
    state['cursor'] = [X // 2, Y // 2, Z // 2]
    
    #set to monochrome so apply_window() can properly correct it
    state['invert'] = ds0.get('PhotometricInterpretation', '') == 'MONOCHROME1'
    
    #we set the default parameters based off dicom file metedata
    state['default_center'] = to_float(ds0.get('WindowCenter'), (vol.min() + vol.max()) / 2)
    state['default_width'] = max(1.0, to_float(ds0.get('WindowWidth'), vol.max() - vol.min()))
    state['center'], state['width'] = state['default_center'], state['default_width']

    AXIS = {'axial': 2, 'coronal': 1, 'sagittal': 0}


    #-------------------------------------------------------------------------------
    # CONNECTING THE DICOM FILES WITH ITS ASSOCIATED XML FILE
    #-------------------------------------------------------------------------------
    
    # Same as root_dir but this is for the xml files. Make sure to match xml_dir with the proper filepath to the xml folder
    xml_dir = r'C:\Users\asosa02\Downloads\lidc_idri\tcia-lidc-xml'

    # If the patient's DICOM folder did not contain a .xml file, search the external XML archive
    if not xml_files:
        search_roots = [xml_dir, root_dir]
        all_root_xmls = []
        for s_root in search_roots:
            if os.path.exists(s_root):
                all_root_xmls.extend(glob.glob(os.path.join(s_root, '**', '*.xml'), recursive=True))
        
        #extract the UID from the ct scan to guarantee we find the right XML file
        target_series_uid = str(ds0.SeriesInstanceUID).strip()
        sample_sop = next(iter(state['sop_to_z'].keys()), '')

        for xp in all_root_xmls:
            try:
                #optimization: we first check the plain-text substring before fully parsing the xml file
                with open(xp, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if target_series_uid in content or (sample_sop and sample_sop in content):
                        xml_files.append(xp)
                        break  # Found the exact XML for this CT series
            except Exception:
                continue

    print(f"Found {len(xml_files)} XML annotation file(s) for {patient_id}: {xml_files}")

    #-------------------------------------------------------------------------------
    #  LIDC PARSER
    #-------------------------------------------------------------------------------
    
    def parse_lidc_xml(xml_paths, sop_map):
        # parses LIDC-IDRI XML annotatrion files using ElementTree
        # we extract both the large and small nodules and map their 3d indices(x,y,z)
        nodules = []
        for xml_path in xml_paths:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception as e:
                print(f"Could not parse XML {xml_path}: {e}")
                continue

            # Helper to strip XML namespaces from tag names(gets rid of {http://www.nih.gov}roi' -> 'roi' and etc)
            def local_tag(elem):
                return elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            # Find all readingSession elements(each one conatins up to 4 annotations)
            sessions = [el for el in root.iter() if local_tag(el) == 'readingSession']

            for session_idx, session in enumerate(sessions):
                # Extracts the true nodule annotations
                nod_elements = [el for el in session if local_tag(el) == 'unblindedReadNodule']

                for nod in nod_elements:
                    #grabs the radiologist's label for the nodule
                    #IMPORTANT
                    #we coulld use this if you want to specify the exact type of nodule
                    nod_id = 'Unknown'
                    for child in nod:
                        if local_tag(child) == 'noduleID' and child.text:
                            nod_id = child.text.strip()
                    
                    #collects and maps 3d points
                    slices_dict = defaultdict(list)
                    all_x, all_y, all_z = [], [], []
                    nod_type = 'nodule_>=3mm'

                    #since a nodule can span multiple slices, we store them as a block in roi
                    rois = [el for el in nod if local_tag(el) == 'roi']
                    for roi in rois:
                        sop_uid = None
                        for child in roi:
                            if local_tag(child) == 'imageSOP_UID' and child.text:
                                sop_uid = child.text.strip()
                                break
                        if not sop_uid or sop_uid not in sop_map:
                            continue
                        #convert the SOPInstanceUID to its sorted z index
                        z_idx = sop_map[sop_uid]
                        
                        #extract the 2d pixel coordinates for this slice
                        coords = []
                        for child in roi:
                            t = local_tag(child)
                            if t in ('edgeMap', 'locus'):
                                if t == 'locus':
                                    nod_type = 'nodule_<3mm'
                                x_val, y_val = None, None
                                for sub in child:
                                    if local_tag(sub) == 'xCoord' and sub.text:
                                        x_val = int(sub.text.strip())
                                    elif local_tag(sub) == 'yCoord' and sub.text:
                                        y_val = int(sub.text.strip())
                                if x_val is not None and y_val is not None:
                                    coords.append((x_val, y_val))
                                    all_x.append(x_val)
                                    all_y.append(y_val)
                                    all_z.append(z_idx)

                        if coords:
                            slices_dict[z_idx].extend(coords)
                    
                    #store the 3d box for this nodule
                    if all_x:
                        # expanding each small nodule by 5 pixels so it creates a valid square region
                        pad = 5 if nod_type == 'nodule_<3mm' else 0
                        nodules.append({
                            'type': nod_type,
                            'reader': session_idx + 1,
                            'id': nod_id,
                            'bbox': { #3d bounding box for nodule to ensure a proper hit evaluation
                                'x_min': min(all_x) - pad, 'x_max': max(all_x) + pad,
                                'y_min': min(all_y) - pad, 'y_max': max(all_y) + pad,
                                'z_min': min(all_z),       'z_max': max(all_z),
                            },
                            'slices': dict(slices_dict)
                        })
        return nodules

    #console logging
    state['nodules'] = parse_lidc_xml(xml_files, state['sop_to_z'])
    nodule_z_slices = sorted({z for n in state['nodules'] for z in n['slices'].keys()})
    print(f"Found {len(series_files)} CT slices for {patient_id}")
    print(f"Found {len(state['nodules'])} radiologist nodule annotations across Z-slices (1-based): {[z + 1 for z in nodule_z_slices]}")
    
    #-------------------------------------------------------------------------------
    # Window Loader, Level/Window Logic, 3d anatomical converter
    #-------------------------------------------------------------------------------
    
    
    def fit_display(shape):
        rows, cols = shape
        sy, sx = {'axial': (spacing['y'], spacing['x']),
                  'coronal': (spacing['z'], spacing['x']),
                  'sagittal': (spacing['z'], spacing['y'])}[state['plane']]
        h_mm, w_mm = rows * sy, cols * sx
        #calculations to ensure that we are within the window's boundaries
        base_scale = min(win.size[0] * 0.8 / w_mm, win.size[1] * 0.9 / h_mm)
        
        #uses the user's zoom level to update the current window zoom
        final_scale = base_scale * state['zoom']
        
        dicom_display.units = 'pix'
        dicom_display.size = (w_mm * final_scale, h_mm * final_scale)
        dicom_display.pos = (state['pan'][0], state['pan'][1])
    
    #applies window/level contrast to the current HU
    def apply_window():
        img, flip = get_slice()
        low, high = state['center'] - state['width'] / 2, state['center'] + state['width'] / 2
        #normalize Hu values from 0.0 to 1.0
        img = (np.clip(img, low, high) - low) / (high - low)
        if state['invert']:
            img = 1.0 - img
        #we rescale values from -1.0 to 1.0 to properly display image in psychopy
        img = img * 2 - 1
        dicom_display.image = np.flipud(img) if flip else img
        fit_display(img.shape)
        update_text()
    
    #updates the top left box(slice counter and gtuess coordinates)
    def update_text():
        ax = AXIS[state['plane']]
        displayInfo.units = 'pix'
        displayInfo.height = 14
        displayInfo.color = [-0.2, -0.2, -0.2]
        displayInfo.pos = (-win.size[0] / 2 + 110, win.size[1] / 2 - 30)
        base_txt = f"{state['plane'][0].upper()} {state['cursor'][ax] + 1}/{plane_count()}"
        if state['current_guess'] is not None:
            gx, gy, gz = state['current_guess']
            base_txt += f"\nGuess: Voxel ({gx}, {gy}, Slice {gz + 1})"
        displayInfo.text = base_txt

    # Converts the mouse position into coordinates(2d) and then maps them into 3d indices
    # as well as the physical position(mm).
    def screen_to_voxel(mouse):
        x_m, y_m = mouse.getPos()
        px, py = convertToPix(np.array([0, 0]), np.array([x_m, y_m]), win.units, win)
        w, h = dicom_display.size
        cx, cy = dicom_display.pos
        u, v = (px - cx) / w + 0.5, (py - cy) / h + 0.5
        if not (0 <= u < 1 and 0 <= v < 1):
            return None, None #return none if outside the image
        x, y, z = state['cursor']
        plane = state['plane']
        if plane == 'axial': 
            x, y = u * X, (1.0 - v) * Y
        elif plane == 'coronal':
            x, z = u * X, v * Z
        else:  # sagittal
            y, z = u * Y, v * Z
        vox = np.clip([int(x), int(y), int(z)], 0, [X - 1, Y - 1, Z - 1])
        mm = origin + vox * np.array([spacing['x'], spacing['y'], spacing['z']])
        return vox, mm
    
    #transforms the 3d array into a viewable 2d screen
    def voxel_to_screen(vx, vy, vz):
        cx_vox, cy_vox, cz_vox = state['cursor']
        plane = state['plane']

        if plane == 'axial':
            if int(round(vz)) != cz_vox:
                return None
            u = (vx + 0.5) / X
            v = 1.0 - (vy + 0.5) / Y
        elif plane == 'coronal':
            if int(round(vy)) != cy_vox:
                return None
            u = (vx + 0.5) / X
            v = (vz + 0.5) / Z
        else:  # sagittal
            if int(round(vx)) != cx_vox:
                return None
            u = (vy + 0.5) / Y
            v = (vz + 0.5) / Z

        w, h = dicom_display.size
        cx, cy = dicom_display.pos
        px = (u - 0.5) * w + cx
        py = (v - 0.5) * h + cy
        return px, py
        
    # Visual representation of where the user places their guess
    guess_ring = visual.Circle(
        win, radius=12, edges=32, units='pix',
        fillColor=None, lineColor=[0.5, 0, 1.0], lineWidth=2.5
    )
    guess_dot = visual.Circle(
        win, radius=2.5, edges=16, units='pix',
        fillColor=[0.5, 0, 1.0], lineColor=[0.5, 0, 1.0]
    )
    # Mostly for debugging purposes, but pressing t highlights where the nodules are
    truth_rect = visual.Rect(
        win, width=10, height=10, units='pix',
        fillColor=None, lineColor=[0.5, 0, 1.0], lineWidth=1.5
    )
    
    #-------------------------------------------------------------------------------
    # Draw guess marker, annotated nodules and switching contrast presets
    #-------------------------------------------------------------------------------
    

    def draw_overlays():
        # Draw's the current user guess onto the inte4rsecting plane
        if state['current_guess'] is not None:
            gx, gy, gz = state['current_guess']
            scr_pos = voxel_to_screen(gx, gy, gz)
            if scr_pos is not None:
                guess_ring.radius = max(8, 12 * state['zoom'])
                guess_ring.pos = scr_pos
                guess_dot.pos = scr_pos
                guess_ring.draw()
                guess_dot.draw()

        #drawing the overlays onto the nodules(press t to show)
        if state['show_truth']:
            cx_vox, cy_vox, cz_vox = state['cursor']
            plane = state['plane']

            for nod in state['nodules']:
                bb = nod['bbox']

                if plane == 'axial':
                    if cz_vox in nod['slices']:
                        coords = nod['slices'][cz_vox]
                        xs = [c[0] for c in coords]
                        ys = [c[1] for c in coords]
                        p1 = voxel_to_screen(min(xs) - 4, min(ys) - 4, cz_vox)
                        p2 = voxel_to_screen(max(xs) + 4, max(ys) + 4, cz_vox)
                    else:
                        continue

                elif plane == 'coronal':
                    # Check if current coronal Y-slice intersects this nodule's Y-range
                    if (bb['y_min'] - 2) <= cy_vox <= (bb['y_max'] + 2):
                        p1 = voxel_to_screen(bb['x_min'] - 4, cy_vox, bb['z_min'] - 1)
                        p2 = voxel_to_screen(bb['x_max'] + 4, cy_vox, bb['z_max'] + 1)
                    else:
                        continue

                else:  # sagittal
                    # Check if current sagittal X-slice intersects this nodule's X-range
                    if (bb['x_min'] - 2) <= cx_vox <= (bb['x_max'] + 2):
                        p1 = voxel_to_screen(cx_vox, bb['y_min'] - 4, bb['z_min'] - 1)
                        p2 = voxel_to_screen(cx_vox, bb['y_max'] + 4, bb['z_max'] + 1)
                    else:
                        continue

                if p1 is not None and p2 is not None:
                    truth_rect.width = max(10, abs(p2[0] - p1[0]))
                    truth_rect.height = max(10, abs(p2[1] - p1[1]))
                    truth_rect.pos = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
                    truth_rect.draw()
        
    #updates Window/Level to a preset or the dicom file preset, and refreshes the window
    def apply_preset(name):
        if name is None:
            state['width'] = state['default_width']
            state['center'] = state['default_center']
            state['preset'] = 'File default'
        else:
            for n, w, l in PRESETS:
                if n == name:
                    state['width'], state['center'] = float(w), float(l)
                    state['preset'] = n
                    break
        apply_window()


    #-------------------------------------------------------------------------------
    # Contrast Buttons, decision buttons, nouse click detection
    #-------------------------------------------------------------------------------
    
    #sets the buttons dimensions
    BTN_W, BTN_H, BTN_GAP = 160, 34, 8
    btn_x = win.size[0] / 2 - BTN_W / 2 - 15
    btn_y0 = win.size[1] / 2 - 40

    #building preset contrast buttons
    entries = [(f"{i + 1}  {n}", n, 'preset') for i, (n, w, l) in enumerate(PRESETS)]
    entries.append(("0  File default", None, 'preset'))

    buttons = []
    for i, (label, name, btype) in enumerate(entries):
        pos = (btn_x, btn_y0 - i * (BTN_H + BTN_GAP))
        rect = visual.Rect(win, width=BTN_W, height=BTN_H, pos=pos, units='pix',
                           fillColor=[-0.6, -0.6, -0.6], lineColor='white', lineWidth=1)
        txt = visual.TextStim(win, text=label, pos=pos, height=15,
                              units='pix', color='white')
        buttons.append({'name': name, 'type': btype, 'pos': pos, 'rect': rect, 'txt': txt})

    # Build the guess buttons
    action_y0 = btn_y0 - len(entries) * (BTN_H + BTN_GAP) - 24
    action_entries = [
        ("Confirm Guess", "confirm_guess"),
        ("No Nodule", "no_nodule"),
    ]
    
    for j, (label, action_name) in enumerate(action_entries):
        pos = (btn_x, action_y0 - j * (BTN_H + BTN_GAP + 4))
        rect = visual.Rect(win, width=BTN_W, height=BTN_H + 4, pos=pos, units='pix',
                           fillColor=[-0.5, -0.5, -0.5], lineColor='white', lineWidth=1.5)
        txt = visual.TextStim(win, text=label, pos=pos, height=15,
                              units='pix', color='white', bold=True)
        buttons.append({'name': action_name, 'type': 'action', 'pos': pos, 'rect': rect, 'txt': txt})

    #mapping number keys to preset buttons(so we can use the keyboard to change presets rather then mouse)
    key_to_preset = {}
    for i, (n, w, l) in enumerate(PRESETS):
        key_to_preset[str(i + 1)] = n
        key_to_preset[f'num_{i + 1}'] = n
    key_to_preset['0'] = None
    key_to_preset['num_0'] = None

    #returns if the mouse clicked was a button
    def button_at(mouse):
        x, y = mouse.getPos()
        px, py = convertToPix(np.array([0, 0]), np.array([x, y]), win.units, win)
        for b in buttons:
            bx, by = b['pos']
            bh = (BTN_H + 4) if b['type'] == 'action' else BTN_H
            if abs(px - bx) <= BTN_W / 2 and abs(py - by) <= bh / 2:
                return b
        return None

    #renders and colors all buttons on the top right of the screen
    def draw_buttons():
        for b in buttons:
            if b['type'] == 'preset':
                active = (b['name'] == state['preset']) or \
                         (b['name'] is None and state['preset'] == 'File default')
                b['rect'].fillColor = [-0.1, 0.2, 0.6] if active else [-0.6, -0.6, -0.6]
                b['txt'].color = 'white'
            elif b['name'] == 'confirm_guess':
                # Highlight green only when a Shift+Click guess is actively placed
                if state['current_guess'] is not None:
                    b['rect'].fillColor = [-0.2, 0.5, -0.2]
                    b['txt'].color = 'white'
                else:
                    b['rect'].fillColor = [-0.75, -0.75, -0.75]
                    b['txt'].color = [-0.3, -0.3, -0.3]
            elif b['name'] == 'no_nodule':
                b['rect'].fillColor = [0.5, -0.2, -0.2]
                b['txt'].color = 'white'
            b['rect'].draw()
            b['txt'].draw()

    #-------------------------------------------------------------------------------
    # Nodule Guess Decision maker, milimeter distance error difference
    #-------------------------------------------------------------------------------
    
    #compares the user's guess against the annotated xml file
    def evaluate_trial_decision(decision_type, rt_sec):
        """
        Evaluates user's 'confirm_guess' or 'no_nodule' submission against
        the XML ground-truth nodules. Computes both bounding-box hit status
        and 3D physical distance (in mm) to the nearest annotated nodule.
        """
        has_nodules = len(state['nodules']) > 0
        guess = state['current_guess']
        
        #IMPORTANT
        #this is our results from the trial, very useful when moving onto the csv data exportation phase
        result = {
            'patient_id': state['patient_id'],
            'decision': decision_type,
            'rt_sec': round(float(rt_sec), 3),
            'ground_truth_has_nodules': has_nodules,
            'num_reader_annotations': len(state['nodules']),
            'guess_voxel': guess,
            'hit': False,
            'outcome': '',
            'matched_nodule_ids': [],
            'closest_dist_mm': None,
        }

        if decision_type == 'no_nodule':
            result['hit'] = not has_nodules
            result['outcome'] = 'True Negative' if not has_nodules else 'False Negative (Miss)'
            return result

        # If user confirmed a nodule location
        gx, gy, gz = guess
        matched_ids = []
        min_dist_mm = float('inf')
        
        #IMPORTANT 
        # Toleracnce margin for nodule guess, change as deemed best fit by researchers
        xy_tol = 5
        z_tol = 1

        for nod in state['nodules']:
            bb = nod['bbox']
            # check to see if guess  falls within the "nodule box"
            in_x = (bb['x_min'] - xy_tol) <= gx <= (bb['x_max'] + xy_tol)
            in_y = (bb['y_min'] - xy_tol) <= gy <= (bb['y_max'] + xy_tol)
            in_z = (bb['z_min'] - z_tol)  <= gz <= (bb['z_max'] + z_tol)
            if in_x and in_y and in_z:
                matched_ids.append(f"R{nod['reader']}:{nod['id']}")

            # Compute the distance from the nodule's center (mm)
            cx = 0.5 * (bb['x_min'] + bb['x_max'])
            cy = 0.5 * (bb['y_min'] + bb['y_max'])
            cz = 0.5 * (bb['z_min'] + bb['z_max'])
            dx_mm = (gx - cx) * spacing['x']
            dy_mm = (gy - cy) * spacing['y']
            dz_mm = (gz - cz) * spacing['z']
            dist_mm = float(np.sqrt(dx_mm**2 + dy_mm**2 + dz_mm**2))
            if dist_mm < min_dist_mm:
                min_dist_mm = dist_mm

        result['matched_nodule_ids'] = matched_ids
        if min_dist_mm != float('inf'):
            result['closest_dist_mm'] = round(min_dist_mm, 2)

        if matched_ids:
            result['hit'] = True
            result['outcome'] = 'True Positive (Hit)'
        else:
            result['hit'] = False
            result['outcome'] = 'False Positive (Mislocalized)' if has_nodules else 'False Positive (No Nodule in Scan)'

        return result
        
        
    #-------------------------------------------------------------------------------
    # This is where the experiment actually runs(while lop for now, but will change)
    #-------------------------------------------------------------------------------
    
    
    
    
    kb = keyboard.Keyboard()
    mouse = Mouse(visible = True, win = win)
    
    #keyboard shorcuts to view different aqnatomical planes of ct scan
    PLANE_KEYS = {'a': 'axial', 'c': 'coronal', 's': 'sagittal'}   # add this

    # we end the program once timer passes 5 minutes
    # this logic has to be changed so the program ends once the experiment is actuallkly over.
    experiment_time = 300 
    
    #render the initial slice before starting
    apply_window()
    
    #resets mouse movements
    mouse.getWheelRel()
    mouse.getRel()
    timer = core.Clock()
    
    #numerical values for controling zoom/drag sensitivity, change as researchers seem best fit.
    drag_sensitivity = 0.5 * (vol.max() - vol.min())
    zoom_sensitivity = 1
    
    left_was_down = False
    right_was_down = False
    zoom_drag_active = False
    wl_drag_active = False
    
    # program runs while there is time left, or esc is pressed.
    while timer.getTime() < experiment_time:
        key_names = [k.name for k in kb.getKeys()]
        #terminates program if esc is pressed
        if 'escape' in key_names:
            break
        if 'r' in key_names:
            apply_preset(None)    # Resets window/level back to file default
            state['zoom'] = 1.0   # Reset zoom back to default on 'r'
            state['pan'] = [0,0]  # Resets user position back to middle
            apply_window()
        if 't' in key_names:
            state['show_truth'] = not state['show_truth']
            print(f"Ground-truth overlay: {'ON' if state['show_truth'] else 'OFF'}")

        for kn in key_names:
            if kn in key_to_preset:
                apply_preset(key_to_preset[kn]) # keys are 0-8 for the different presets of windows/levels
            if kn in PLANE_KEYS:                       
                # If the user has placed a guess, snap the 3D cross-section cursor to it
                # (or if the mouse is over the image, slice through the mouse's 3D position)
                vox_hover, _ = screen_to_voxel(mouse)
                if state['current_guess'] is not None:
                    state['cursor'] = list(state['current_guess'])
                elif vox_hover is not None:
                    state['cursor'] = [int(vox_hover[0]), int(vox_hover[1]), int(vox_hover[2])]
                state['plane'] = PLANE_KEYS[kn]  # keys are a, s, c for the different anatomical views
                apply_window()

        wheel_x, wheel_y = mouse.getWheelRel()
        if wheel_y != 0: # this is the scroll wheel functionality
            ax = AXIS[state['plane']]
            new_pos = int(np.clip(state['cursor'][ax] + int(wheel_y), 0, plane_count() - 1))
            if new_pos != state['cursor'][ax]:
                state['cursor'][ax] = new_pos
                apply_window()

        dx, dy = mouse.getRel()
        left_pressed, middle_pressed, right_pressed = mouse.getPressed()
        both_pressed = left_pressed and right_pressed
        
        #BOTH PRESSED LOGIC: changing user position
        if both_pressed:
            zoom_drag_active = False
            wl_drag_active = False
            if dx != 0 or dy != 0:
                # Convert dx, dy (height units) to pixels so dragging tracks 1:1 with the mouse
                state['pan'][0] += dx * win.size[1] 
                state['pan'][1] += dy * win.size[1]
                apply_window()
        else:
            # LEFT CLICK LOGIC: Preset Buttons & Zoom
            # Check if Shift is currently held down on the keyboard
            held_keys = [k.name for k in kb.getKeys(waitRelease=False, clear=False)]
            shift_held = ('lshift' in held_keys) or ('rshift' in held_keys) or ('shift' in held_keys)

            # LEFT CLICK LOGIC: Preset Buttons, Shift+ Left Click Guess, & Zoom
            if left_pressed and not left_was_down:          # A new left-click just started
                hit = button_at(mouse)
                if hit is not None:
                    zoom_drag_active = False
                    if hit['type'] == 'preset':
                        apply_preset(hit['name'])           # Clicked a W/L preset button
                    elif hit['name'] == 'confirm_guess':
                        if state['current_guess'] is not None:
                            state['trial_result'] = evaluate_trial_decision('confirm_guess', timer.getTime())
                            print("\n=== TRIAL SUBMITTED: CONFIRM GUESS ===")
                            for k, v in state['trial_result'].items():
                                print(f"  {k}: {v}")
                            break                           # End current trial routine
                        else:
                            print("Cannot confirm guess: Shift + Left-Click on the scan to place a guess first!")
                    elif hit['name'] == 'no_nodule':
                        state['trial_result'] = evaluate_trial_decision('no_nodule', timer.getTime())
                        print("\n=== TRIAL SUBMITTED: NO NODULE ===")
                        for k, v in state['trial_result'].items():
                            print(f"  {k}: {v}")
                        break                               # End current trial routine
                elif shift_held:
                    # logic behind user placing a nodule guess
                    vox, mm = screen_to_voxel(mouse)
                    if vox is not None:
                        state['current_guess'] = [int(vox[0]), int(vox[1]), int(vox[2])]
                        state['cursor'] = [int(vox[0]), int(vox[1]), int(vox[2])]
                        update_text()
                        print(f"Placed guess at Voxel {state['current_guess']} (Axial:{vox[2]+1}, Coronal:{vox[1]+1}, Sagittal:{vox[0]+1})")
                    zoom_drag_active = False
                else:
                    zoom_drag_active = True                 # Clicked the image -> start zooming
            elif not left_pressed:
                zoom_drag_active = False

            if zoom_drag_active and dy != 0:
                # Adjust zoom level (dragging up zooms in, down zooms out)
                state['zoom'] = max(0.1, min(state['zoom'] + dy * zoom_sensitivity, 10.0))
                apply_window()

            # RIGHT CLICK LOGIC: Window/Level Adjustments
            if right_pressed and not right_was_down:        # A new right-click just started
                hit = button_at(mouse)
                if hit is not None:
                    wl_drag_active = False                  # Don't right-click buttons
                else:
                    wl_drag_active = True                   # Clicked the image -> start window/level drag
            elif not right_pressed:
                wl_drag_active = False

            if wl_drag_active and (dx != 0 or dy != 0):
                state['width'] = max(1.0, state['width'] + dx * drag_sensitivity)
                state['center'] += dy * drag_sensitivity
                state['preset'] = 'Custom'
                apply_window()
        left_was_down = left_pressed
        right_was_down = right_pressed
        
        dicom_display.draw()
        draw_overlays()
        displayInfo.draw()
        draw_buttons()
        win.flip()
        
    #ENDS THE PROGRAM
    core.quit()

    # store start times for trial
    trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    trial.tStart = globalClock.getTime(format='float')
    trial.status = STARTED
    thisExp.addData('trial.started', trial.tStart)
    trial.maxDuration = None
    # keep track of which components have finished
    trialComponents = trial.components
    for thisComponent in trial.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "trial" ---
    thisExp.currentRoutine = trial
    trial.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *dicom_display* updates
        
        # if dicom_display is starting this frame...
        if dicom_display.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            dicom_display.frameNStart = frameN  # exact frame index
            dicom_display.tStart = t  # local t and not account for scr refresh
            dicom_display.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dicom_display, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dicom_display.started')
            # update status
            dicom_display.status = STARTED
            dicom_display.setAutoDraw(True)
        
        # if dicom_display is active this frame...
        if dicom_display.status == STARTED:
            # update params
            pass
        
        # if dicom_display is stopping this frame...
        if dicom_display.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > dicom_display.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                dicom_display.tStop = t  # not accounting for scr refresh
                dicom_display.tStopRefresh = tThisFlipGlobal  # on global time
                dicom_display.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dicom_display.stopped')
                # update status
                dicom_display.status = FINISHED
                dicom_display.setAutoDraw(False)
        
        # *displayInfo* updates
        
        # if displayInfo is starting this frame...
        if displayInfo.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            displayInfo.frameNStart = frameN  # exact frame index
            displayInfo.tStart = t  # local t and not account for scr refresh
            displayInfo.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(displayInfo, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'displayInfo.started')
            # update status
            displayInfo.status = STARTED
            displayInfo.setAutoDraw(True)
        
        # if displayInfo is active this frame...
        if displayInfo.status == STARTED:
            # update params
            pass
        
        # if displayInfo is stopping this frame...
        if displayInfo.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > displayInfo.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                displayInfo.tStop = t  # not accounting for scr refresh
                displayInfo.tStopRefresh = tThisFlipGlobal  # on global time
                displayInfo.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'displayInfo.stopped')
                # update status
                displayInfo.status = FINISHED
                displayInfo.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=trial,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            trial.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if trial.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in trial.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "trial" ---
    for thisComponent in trial.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for trial
    trial.tStop = globalClock.getTime(format='float')
    trial.tStopRefresh = tThisFlipGlobal
    thisExp.addData('trial.stopped', trial.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if trial.maxDurationReached:
        routineTimer.addTime(-trial.maxDuration)
    elif trial.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.500000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)
    

def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
