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
import pylink
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
import csv
import json
import xml.etree.ElementTree as ET
import re
from pyglet.window import key as pyglet_key
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
import glob
import pydicom
from psychopy.event import Mouse

DUMMY_NODE = True

if DUMMY_NODE:
    tracker = pylink.EyeLink(None)
else:
    tracker = pylink.EyeLink("100.1.1.1")

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
    
    # Disable default WideText/Pickle so only our two CSVs are generated
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\andys\\Documents\\psychopy_scripts\\scroll_wheel_test.py',
        savePickle=False, saveWideText=False,
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
    # --- Prepare CSV Logging & Multi-Trial Session ---
    NUM_TRIALS = 3               # Hardcoded 3 trials per session
    LOG_EVERY_FRAME = True       # True = log every win.flip(); False = log only on input/state changes
    INPLANE_XY_TOL = 4           # Voxel padding for both hit evaluation and nodule hover detection
    FULL_SWEEP_THRESHOLD = 0.95  # >=95% of slices visited in a plane counts as a full plane sweep

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    trials_csv_path = f"{filename}_trials.csv"
    telemetry_csv_path = f"{filename}_telemetry.csv"

    TRIALS_FIELDNAMES = [
        'participant_id', 'session', 'exp_date', 'psychopy_version', 'pilot_mode',
        'win_width_px', 'win_height_px', 'measured_fps',
        'trial_idx', 'trial_start_global_sec', 'trial_end_global_sec',
        'patient_id', 'series_uid', 'vol_dim_x', 'vol_dim_y', 'vol_dim_z',
        'spacing_x_mm', 'spacing_y_mm', 'spacing_z_mm',
        'origin_x_mm', 'origin_y_mm', 'origin_z_mm',
        'photometric_invert', 'default_wl_width', 'default_wl_center',
        'xml_file_found', 'gt_has_nodules', 'gt_num_consensus_nodules',
        'gt_total_raw_reader_anns', 'gt_nodule_z_slices_0based', 'gt_nodules_json',
        'exit_reason', 'decision', 'rt_sec', 'plane_submitted',
        'cursor_at_submit_x', 'cursor_at_submit_y', 'cursor_at_submit_z',
        'guess_placed', 'num_guess_updates', 'time_to_first_guess_sec', 'time_of_final_guess_sec',
        'guess_vx', 'guess_vy', 'guess_vz', 'guess_mm_x', 'guess_mm_y', 'guess_mm_z',
        'inplane_xy_tol_vox', 'hit', 'outcome', 'matched_nodule_ids',
        'closest_nodule_id', 'closest_dist_mm', 'closest_inplane_dist_mm', 'closest_z_slice_error',
        'total_frames', 'time_axial_sec', 'time_coronal_sec', 'time_sagittal_sec',
        'plane_switches_count', 'scroll_events_count', 'total_slices_traversed',
        'scroll_reversals_total', 'scroll_reversals_axial', 'scroll_reversals_coronal',
        'scroll_reversals_sagittal', 'scroll_reversals_on_nodule_slices',
        'unique_axial_slices_visited', 'axial_coverage_pct', 'axial_min_slice_visited',
        'axial_max_slice_visited', 'axial_span_pct', 'full_sweep_axial',
        'unique_coronal_slices_visited', 'coronal_coverage_pct', 'full_sweep_coronal',
        'unique_sagittal_slices_visited', 'sagittal_coverage_pct', 'full_sweep_sagittal',
        'nodule_slice_ever_visited', 'time_to_first_nodule_slice_sec', 'dwell_time_on_nodule_slices_sec',
        'nodule_mouse_ever_hovered', 'time_to_first_nodule_hover_sec', 'dwell_time_nodule_hover_sec',
        'wl_preset_changes_count', 'wl_drag_frames_count', 'zoom_drag_frames_count',
        'max_zoom_reached', 'min_zoom_reached', 'pan_drag_frames_count',
        'reset_count', 'truth_overlay_toggled', 'truth_overlay_time_sec'
    ]

    TELEMETRY_HEADERS = [
        'participant_id', 'session', 'trial_idx', 'patient_id', 'frame_idx',
        'global_time_sec', 'trial_time_sec', 'dt_sec',
        'plane', 'cursor_x', 'cursor_y', 'cursor_z',
        'active_slice_0based', 'active_slice_1based',
        'nodule_on_current_slice', 'visible_nodule_ids',
        'zoom', 'pan_px_x', 'pan_px_y',
        'img_pos_px_x', 'img_pos_px_y', 'img_width_px', 'img_height_px',
        'wl_width', 'wl_center', 'wl_preset', 'show_truth',
        'mouse_px_x', 'mouse_px_y', 'mouse_in_image',
        'hover_vx', 'hover_vy', 'hover_vz',
        'hover_mm_x', 'hover_mm_y', 'hover_mm_z', 'hover_hu',
        'mouse_over_nodule', 'hovered_nodule_ids',
        'mouse_left', 'mouse_middle', 'mouse_right', 'shift_held',
        'wheel_dy', 'scroll_reversal_on_frame', 'drag_mode',
        'guess_active', 'guess_vx', 'guess_vy', 'guess_vz',
        'gaze_px_x', 'gaze_px_y', 'events'
    ]

    with open(trials_csv_path, mode='w', newline='', encoding='utf-8') as f_tr:
        csv.DictWriter(f_tr, fieldnames=TRIALS_FIELDNAMES).writeheader()

    with open(telemetry_csv_path, mode='w', newline='', encoding='utf-8') as f_tel:
        csv.writer(f_tel).writerow(TELEMETRY_HEADERS)

    active_telemetry_buffer = []

    def flush_telemetry_buffer():
        if active_telemetry_buffer:
            with open(telemetry_csv_path, mode='a', newline='', encoding='utf-8') as f_tel:
                csv.writer(f_tel).writerows(active_telemetry_buffer)
            active_telemetry_buffer.clear()

    runAtExit.append(flush_telemetry_buffer)
    
    #-------------------------------------------------------------------------------
    # EYELINK SETUP, CALIBRATION & EXIT CLEANUP
    #-------------------------------------------------------------------------------
    
    
    # Keep only letters, numbers, and underscores; cap at 8 chars; fallback to "test" if empty
    clean_id = re.sub(r'[^A-Za-z0-9_]', '', str(expInfo['participant']))[:8] or "test"
    edf_host_name = f"{clean_id}.edf"
    edf_local_path = f"{filename}.edf"

    tracker.openDataFile(edf_host_name)
    tracker.sendCommand("sample_rate 1000")
    tracker.sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON")
    tracker.sendCommand("file_sample_data = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS")
    tracker.sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON")
    tracker.sendCommand("link_sample_data = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS")

    # Dynamically match whatever monitor resolution PsychoPy opened
    scn_width, scn_height = int(win.size[0]), int(win.size[1])
    tracker.sendCommand(f"screen_pixel_coords = 0 0 {scn_width - 1} {scn_height - 1}")
    tracker.sendMessage(f"DISPLAY_COORDS 0 0 {scn_width - 1} {scn_height - 1}")

    # Link PsychoPy window for calibration targets
    genv = EyeLinkCoreGraphicsPsychoPy(tracker, win)
    pylink.openGraphicsEx(genv)

    if not DUMMY_NODE:
        tracker.doTrackerSetup()
    

    def cleanup_eyelink():
        if tracker.isConnected():
            tracker.setOfflineMode()
            pylink.pumpDelay(100)
            tracker.closeDataFile()
            if not DUMMY_NODE:
                tracker.receiveDataFile(edf_host_name, edf_local_path)
            tracker.close()

    runAtExit.append(cleanup_eyelink)

    # Session-wide input devices (created once before the trial loop)
    kb = keyboard.Keyboard()
    mouse = Mouse(visible=True, win=win)
    key_state = pyglet_key.KeyStateHandler()
    win.winHandle.push_handlers(key_state)
    
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
    
    user_escaped_session = False
    for trial_idx in range(1, NUM_TRIALS + 1):
        if user_escaped_session:
            break
        
        folder = None
        series_files = []
        xml_files = []

        while shuffled_dirs:
            #finds all dicom files within the subfolders
            candidate_dir = shuffled_dirs.pop(0)
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
            if state['plane'] == 'axial':    return vol[z, :, :], False    # row 0 = anterior -> top
            if state['plane'] == 'coronal':  return vol[:, y, :], True   # flipud so z=0 (inferior) -> bottom
            return vol[:, :, x], True                                    # sagittal: flipud so z=0 -> bottom

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
            'plane': 'axial',
            'center': 0.0,
            'width': 1.0,
            'invert': False,
            'ds': None,
            'default_center': 0.0,
            'default_width': 1.0,
            'preset': 'File default',
            'zoom': 1.0,
            'pan': [0, 0],
            'sop_to_z': {},
            'nodules': [],
            'include_small_locus': False,
            'current_guess': None,
            'show_truth': False,
            'trial_result': None,
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
        
        # grabs the xml directory
        xml_dir = os.path.join(os.path.expanduser('~'), 'Downloads', 'lidc_idri', 'tcia-lidc-xml')

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
        
        def rasterize_polygon_2d(pts):
            """
            Given ordered 2D contour vertices [(x0, y0), (x1, y1), ...] on an axial slice,
            interpolates all polygon edges so no rows/columns are skipped and returns:
              - row_spans: dict mapping each intersected y -> (x_min_at_y, x_max_at_y)
              - col_spans: dict mapping each intersected x -> (y_min_at_x, y_max_at_x)
            """
            if not pts:
                return {}, {}
            if len(pts) == 1:
                x0, y0 = pts[0]
                return {y0: (x0, x0)}, {x0: (y0, y0)}

            edge_pixels = []
            n = len(pts)
            for i in range(n):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % n]
                steps = max(abs(x2 - x1), abs(y2 - y1), 1)
                xs = np.rint(np.linspace(x1, x2, steps + 1)).astype(int)
                ys = np.rint(np.linspace(y1, y2, steps + 1)).astype(int)
                edge_pixels.extend(zip(xs.tolist(), ys.tolist()))

            row_xs = defaultdict(list)
            for px, py in edge_pixels:
                row_xs[py].append(px)

            row_spans = {}
            col_ys = defaultdict(list)
            for py, xs_list in row_xs.items():
                xmin_row, xmax_row = min(xs_list), max(xs_list)
                row_spans[py] = (xmin_row, xmax_row)
                # Fill interior scanline between xmin_row and xmax_row
                for fill_x in range(xmin_row, xmax_row + 1):
                    col_ys[fill_x].append(py)

            col_spans = {px: (min(ys_list), max(ys_list)) for px, ys_list in col_ys.items()}
            return row_spans, col_spans
        
        def parse_lidc_xml(xml_paths, sop_map, include_small_locus=False, min_readers=2):
            raw_annotations = []
            for xml_path in xml_paths:
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                except Exception as e:
                    print(f"Could not parse XML {xml_path}: {e}")
                    continue

                def local_tag(elem):
                    return elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

                sessions = [el for el in root.iter() if local_tag(el) == 'readingSession']

                for session_idx, session in enumerate(sessions):
                    nod_elements = [el for el in session if local_tag(el) == 'unblindedReadNodule']

                    for nod in nod_elements:
                        nod_id = 'Unknown'
                        has_characteristics = False
                        for child in nod:
                            ctag = local_tag(child)
                            if ctag == 'noduleID' and child.text:
                                nod_id = child.text.strip()
                            elif ctag == 'characteristics':
                                has_characteristics = True

                        slices_dict = defaultdict(list)
                        has_polygon_contour = False

                        rois = [el for el in nod if local_tag(el) == 'roi']
                        for roi in rois:
                            inclusion = True
                            sop_uid = None
                            for child in roi:
                                ctag = local_tag(child)
                                if ctag == 'imageSOP_UID' and child.text:
                                    sop_uid = child.text.strip()
                                elif ctag == 'inclusion' and child.text:
                                    inclusion = child.text.strip().upper() != 'FALSE'
                            if not inclusion or not sop_uid or sop_uid not in sop_map:
                                continue

                            z_idx = sop_map[sop_uid]
                            coords = []
                            for child in roi:
                                t = local_tag(child)
                                if t not in ('edgeMap', 'locus'):
                                    continue
                                x_val, y_val = None, None
                                for sub in child:
                                    if local_tag(sub) == 'xCoord' and sub.text:
                                        x_val = int(sub.text.strip())
                                    elif local_tag(sub) == 'yCoord' and sub.text:
                                        y_val = int(sub.text.strip())
                                if x_val is not None and y_val is not None:
                                    coords.append((x_val, y_val))

                            if len(coords) >= 3:
                                has_polygon_contour = True
                                slices_dict[z_idx].extend(coords)
                            elif coords and include_small_locus:
                                slices_dict[z_idx].extend(coords)

                        if not slices_dict:
                            continue

                        is_large_nodule = has_polygon_contour and (has_characteristics or len(slices_dict) >= 2)
                        if (not is_large_nodule or len(slices_dict) < 2) and not include_small_locus:
                            continue

                        # Rasterize each Z-slice contour into filled (z, y) -> (x_min, x_max) spans
                        zy_spans = {}
                        all_x, all_y, all_z = [], [], []
                        for z_k, pts in slices_dict.items():
                            row_spans, _ = rasterize_polygon_2d(pts)
                            for py, (rx0, rx1) in row_spans.items():
                                zy_spans[(z_k, py)] = (rx0, rx1)
                                all_x.extend([rx0, rx1])
                                all_y.append(py)
                                all_z.append(z_k)

                        if not zy_spans:
                            continue

                        raw_annotations.append({
                            'reader': session_idx + 1,
                            'id': nod_id,
                            'bbox': {
                                'x_min': min(all_x), 'x_max': max(all_x),
                                'y_min': min(all_y), 'y_max': max(all_y),
                                'z_min': min(all_z), 'z_max': max(all_z),
                            },
                            'zy_spans': zy_spans,
                            'slices': dict(slices_dict),
                        })

            # Group overlapping reader annotations into physical nodule clusters
            clusters = []
            for ann in raw_annotations:
                b1 = ann['bbox']
                placed = False
                for clust in clusters:
                    b2 = clust['bbox']
                    overlap_x = b1['x_min'] <= b2['x_max'] + 5 and b2['x_min'] <= b1['x_max'] + 5
                    overlap_y = b1['y_min'] <= b2['y_max'] + 5 and b2['y_min'] <= b1['y_max'] + 5
                    overlap_z = b1['z_min'] <= b2['z_max'] + 1 and b2['z_min'] <= b1['z_max'] + 1
                    if overlap_x and overlap_y and overlap_z:
                        clust['members'].append(ann)
                        b2['x_min'] = min(b2['x_min'], b1['x_min'])
                        b2['x_max'] = max(b2['x_max'], b1['x_max'])
                        b2['y_min'] = min(b2['y_min'], b1['y_min'])
                        b2['y_max'] = max(b2['y_max'], b1['y_max'])
                        b2['z_min'] = min(b2['z_min'], b1['z_min'])
                        b2['z_max'] = max(b2['z_max'], b1['z_max'])
                        placed = True
                        break
                if not placed:
                    clusters.append({'bbox': dict(b1), 'members': [ann]})

            # Build consensus nodules where >= min_readers agree on each voxel
            nodules = []
            for c_idx, clust in enumerate(clusters):
                members = clust['members']
                distinct_readers = {m['reader'] for m in members}
                if len(distinct_readers) < min_readers:
                    continue
                voxel_votes = defaultdict(int)
                for m in members:
                    # Count at most 1 vote per reader per voxel
                    reader_voxels = set()
                    for (z_k, py), (rx0, rx1) in m['zy_spans'].items():
                        for px in range(rx0, rx1 + 1):
                            reader_voxels.add((px, py, z_k))
                    for vox_key in reader_voxels:
                        voxel_votes[vox_key] += 1

                consensus_voxels = [v for v, count in voxel_votes.items() if count >= min_readers]
                if not consensus_voxels:
                    continue

                axial_pts = defaultdict(list)
                coronal_z_spans = defaultdict(dict)
                sagittal_z_spans = defaultdict(dict)
                all_x, all_y, all_z = [], [], []

                # Group consensus voxels by (z, y) and (z, x)
                zy_to_xs = defaultdict(list)
                zx_to_ys = defaultdict(list)
                for px, py, pz in consensus_voxels:
                    axial_pts[pz].append((px, py))
                    zy_to_xs[(pz, py)].append(px)
                    zx_to_ys[(pz, px)].append(py)
                    all_x.append(px)
                    all_y.append(py)
                    all_z.append(pz)

                axial_bounds = {}
                for pz, pts in axial_pts.items():
                    xs_p = [p[0] for p in pts]
                    ys_p = [p[1] for p in pts]
                    # Require at least 3x3 pixels on an axial slice
                    if (max(xs_p) - min(xs_p)) >= 2 and (max(ys_p) - min(ys_p)) >= 2:
                        axial_bounds[pz] = (min(xs_p), max(xs_p), min(ys_p), max(ys_p))

                if not axial_bounds:
                    continue

                for (pz, py), xs_list in zy_to_xs.items():
                    if pz in axial_bounds and (max(xs_list) - min(xs_list)) >= 2:
                        coronal_z_spans[py][pz] = (min(xs_list), max(xs_list))

                for (pz, px), ys_list in zx_to_ys.items():
                    if pz in axial_bounds and (max(ys_list) - min(ys_list)) >= 2:
                        sagittal_z_spans[px][pz] = (min(ys_list), max(ys_list))

                coronal_bounds = {}
                for py, z_map in coronal_z_spans.items():
                    xmin_y = min(span[0] for span in z_map.values())
                    xmax_y = max(span[1] for span in z_map.values())
                    zmin_y = min(z_map.keys())
                    zmax_y = max(z_map.keys())
                    coronal_bounds[py] = (xmin_y, xmax_y, zmin_y, zmax_y)

                sagittal_bounds = {}
                for px, z_map in sagittal_z_spans.items():
                    ymin_x = min(span[0] for span in z_map.values())
                    ymax_x = max(span[1] for span in z_map.values())
                    zmin_x = min(z_map.keys())
                    zmax_x = max(z_map.keys())
                    sagittal_bounds[px] = (ymin_x, ymax_x, zmin_x, zmax_x)

                reader_labels = ",".join(f"R{m['reader']}:{m['id']}" for m in members)
                nodules.append({
                    'type': 'nodule_>=3mm',
                    'reader': f"Consensus({len(distinct_readers)}R)",
                    'id': f"Nodule_{c_idx + 1}[{reader_labels}]",
                    'bbox': {
                        'x_min': min(all_x), 'x_max': max(all_x),
                        'y_min': min(all_y), 'y_max': max(all_y),
                        'z_min': min(all_z), 'z_max': max(all_z),
                    },
                    'slices': dict(axial_pts),
                    'z_bounds': axial_bounds,
                    'axial_bounds': axial_bounds,
                    'coronal_bounds': coronal_bounds,
                    'sagittal_bounds': sagittal_bounds,
                    'coronal_z_spans': dict(coronal_z_spans),
                    'sagittal_z_spans': dict(sagittal_z_spans),
                })
            state['raw_reader_anns_count'] = len(raw_annotations)
            return nodules

        state['raw_reader_anns_count'] = 0
        state['nodules'] = parse_lidc_xml(
            xml_files, state['sop_to_z'], include_small_locus=state['include_small_locus']
        )
        nodule_z_slices = sorted({z for n in state['nodules'] for z in n['axial_bounds'].keys()})
        print(f"Found {len(series_files)} CT slices for {patient_id}")
        print(f"Found {len(state['nodules'])} segmented nodule annotations across Z-slices (1-based): {[z + 1 for z in nodule_z_slices]}")
        
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
            dicom_display.image = np.flipud(img) if not flip else img
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
                return None, None  # return none if outside the image
            x, y, z = state['cursor']
            plane = state['plane']
            if plane == 'axial':
                # Axial is flipped with np.flipud, so y=0 is at top (v=1) and y=Y-1 is at bottom (v=0)
                x, y = u * X, (1.0 - v) * Y
            elif plane == 'coronal':
                # Coronal is NOT flipped, so ImageStim renders z=0 at bottom (v=0) and z=Z-1 at top (v=1)
                x, z = u * X, v * Z
            else:  # sagittal
                # Sagittal is NOT flipped, so ImageStim renders z=0 at bottom (v=0) and z=Z-1 at top (v=1)
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
                    if plane == 'axial':
                        if cz_vox in nod['axial_bounds']:
                            xmin, xmax, ymin, ymax = nod['axial_bounds'][cz_vox]
                            p1 = voxel_to_screen(xmin - 4, ymin - 4, cz_vox)
                            p2 = voxel_to_screen(xmax + 4, ymax + 4, cz_vox)
                        else:
                            continue
                    elif plane == 'coronal':
                        if cy_vox in nod['coronal_bounds']:
                            xmin, xmax, zmin, zmax = nod['coronal_bounds'][cy_vox]
                            p1 = voxel_to_screen(xmin - 4, cy_vox, zmin)
                            p2 = voxel_to_screen(xmax + 4, cy_vox, zmax)
                        else:
                            continue
                    else:  # sagittal
                        if cx_vox in nod['sagittal_bounds']:
                            ymin, ymax, zmin, zmax = nod['sagittal_bounds'][cx_vox]
                            p1 = voxel_to_screen(cx_vox, ymin - 4, zmin)
                            p2 = voxel_to_screen(cx_vox, ymax + 4, zmax)
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
        # Helper to check if a 3D voxel (vx, vy, vz) is inside a nodule on a given plane (with padding)
        def is_voxel_in_nodule(vx, vy, vz, plane, nod, tol=INPLANE_XY_TOL):
            if plane == 'axial':
                if vz in nod['axial_bounds']:
                    xmin, xmax, ymin, ymax = nod['axial_bounds'][vz]
                    return (xmin - tol) <= vx <= (xmax + tol) and (ymin - tol) <= vy <= (ymax + tol)
            elif plane == 'coronal':
                if vy in nod['coronal_bounds']:
                    xmin, xmax, zmin, zmax = nod['coronal_bounds'][vy]
                    if zmin <= vz <= zmax and (xmin - tol) <= vx <= (xmax + tol):
                        z_spans = nod['coronal_z_spans'].get(vy, {})
                        if vz in z_spans:
                            rx0, rx1 = z_spans[vz]
                            return (rx0 - tol) <= vx <= (rx1 + tol)
                        return True
            else:  # sagittal
                if vx in nod['sagittal_bounds']:
                    ymin, ymax, zmin, zmax = nod['sagittal_bounds'][vx]
                    if zmin <= vz <= zmax and (ymin - tol) <= vy <= (ymax + tol):
                        z_spans = nod['sagittal_z_spans'].get(vx, {})
                        if vz in z_spans:
                            ry0, ry1 = z_spans[vz]
                            return (ry0 - tol) <= vy <= (ry1 + tol)
                        return True
            return False

        # Returns list of nodule IDs visible on the current slice in the active plane
        def get_visible_nodules_on_slice(cx_vox, cy_vox, cz_vox, plane):
            vis = []
            for nod in state['nodules']:
                if plane == 'axial' and cz_vox in nod['axial_bounds']:
                    vis.append(nod['id'])
                elif plane == 'coronal' and cy_vox in nod['coronal_bounds']:
                    vis.append(nod['id'])
                elif plane == 'sagittal' and cx_vox in nod['sagittal_bounds']:
                    vis.append(nod['id'])
            return vis

        # Compares the user's guess against the annotated XML file
        def evaluate_trial_decision(decision_type, rt_sec):
            has_nodules = len(state['nodules']) > 0
            guess = state['current_guess']
            
            result = {
                'patient_id': state['patient_id'],
                'decision': decision_type,
                'plane_submitted': state['plane'],
                'rt_sec': round(float(rt_sec), 4),
                'ground_truth_has_nodules': has_nodules,
                'num_reader_annotations': len(state['nodules']),
                'guess_voxel': guess,
                'hit': False,
                'outcome': '',
                'matched_nodule_ids': [],
                'closest_nodule_id': None,
                'closest_dist_mm': None,
                'closest_inplane_dist_mm': None,
                'closest_z_slice_error': None,
            }
            
            if decision_type == 'no_nodule':
                result['hit'] = not has_nodules
                result['outcome'] = 'True Negative' if not has_nodules else 'False Negative (Miss)'
                return result

            if guess is None:
                result['hit'] = False
                result['outcome'] = 'No Decision (Timeout/Escape)'
                return result

            gx, gy, gz = guess
            plane = state['plane']
            matched_ids = []
            min_dist_sq = float('inf')
            min_dist_mm = float('inf')
            closest_id = None
            closest_inplane_mm = None
            closest_z_err = None

            for nod in state['nodules']:
                bb = nod['bbox']
                if is_voxel_in_nodule(gx, gy, gz, plane, nod, tol=INPLANE_XY_TOL):
                    matched_ids.append(f"{nod['reader']}:{nod['id']}")

                cx = 0.5 * (bb['x_min'] + bb['x_max'])
                cy = 0.5 * (bb['y_min'] + bb['y_max'])
                cz = 0.5 * (bb['z_min'] + bb['z_max'])
                dx_mm = (gx - cx) * spacing['x']
                dy_mm = (gy - cy) * spacing['y']
                dz_mm = (gz - cz) * spacing['z']
                dist_sq = dx_mm**2 + dy_mm**2 + dz_mm**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    min_dist_mm = float(np.sqrt(dist_sq))
                    closest_id = nod['id']
                    closest_z_err = int(round(gz - cz))
                    if plane == 'axial':
                        closest_inplane_mm = float(np.sqrt(dx_mm**2 + dy_mm**2))
                    elif plane == 'coronal':
                        closest_inplane_mm = float(np.sqrt(dx_mm**2 + dz_mm**2))
                    else:
                        closest_inplane_mm = float(np.sqrt(dy_mm**2 + dz_mm**2))

            result['matched_nodule_ids'] = matched_ids
            if min_dist_mm != float('inf'):
                result['closest_nodule_id'] = closest_id
                result['closest_dist_mm'] = round(min_dist_mm, 2)
                result['closest_inplane_dist_mm'] = round(closest_inplane_mm, 2)
                result['closest_z_slice_error'] = closest_z_err

            if matched_ids:
                result['hit'] = True
                result['outcome'] = 'True Positive (Hit)'
            else:
                result['hit'] = False
                result['outcome'] = 'False Positive (Mislocalized)' if has_nodules else 'False Positive (No Nodule in Scan)'

            return result
        #-------------------------------------------------------------------------------
        # TRIAL INITIALIZATION & TELEMETRY TRACKERS
        #-------------------------------------------------------------------------------
        PLANE_KEYS = {'a': 'axial', 'c': 'coronal', 's': 'sagittal'}
        WATCHED_KEYS = ['escape', 'r', 't'] + list(PLANE_KEYS.keys()) + list(key_to_preset.keys())
        experiment_time = 600

        apply_window()
        timer = core.Clock()
        trial_start_global_sec = globalClock.getTime()

        drag_sensitivity = 0.25 * (vol.max() - vol.min())
        zoom_sensitivity = 0.25
        
        zoom_drag_active = False
        wl_drag_active = False
        
        x_m, y_m = mouse.getPos()
        prev_m_px_x, prev_m_px_y = convertToPix(np.array([0, 0]), np.array([x_m, y_m]), win.units, win)

        # Per-trial search & oscillation accumulators
        frame_idx = 0
        prev_trial_t = 0.0
        exit_reason = 'timeout'

        time_by_plane = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}
        plane_switches_count = 0
        scroll_events_count = 0
        total_slices_traversed = 0

        # Scroll oscillation (direction reversal) tracking per plane
        last_scroll_dir = {'axial': 0, 'coronal': 0, 'sagittal': 0}
        scroll_reversals_by_plane = {'axial': 0, 'coronal': 0, 'sagittal': 0}
        scroll_reversals_on_nodule_slices = 0

        # Unique slices visited for coverage & full-sweep detection
        visited_slices = {
            'axial': {state['cursor'][2]},
            'coronal': {state['cursor'][1]},
            'sagittal': {state['cursor'][0]},
        }

        nodule_slice_ever_visited = False
        time_to_first_nodule_slice_sec = None
        dwell_time_on_nodule_slices_sec = 0.0

        nodule_mouse_ever_hovered = False
        time_to_first_nodule_hover_sec = None
        dwell_time_nodule_hover_sec = 0.0

        num_guess_updates = 0
        time_to_first_guess_sec = None
        time_of_final_guess_sec = None
        final_guess_mm = None

        wl_preset_changes_count = 0
        wl_drag_frames_count = 0
        zoom_drag_frames_count = 0
        pan_drag_frames_count = 0
        max_zoom_reached = state['zoom']
        min_zoom_reached = state['zoom']
        reset_count = 0
        truth_overlay_toggled = False
        truth_overlay_time_sec = 0.0

        prev_logged_state = None
        
        # Start EyeLink recording for this trial
        tracker.setOfflineMode()
        tracker.sendMessage(f"TRIALID {trial_idx}")
        tracker.sendMessage(f"!V TRIAL_VAR patient_id {state['patient_id']}")
        tracker.startRecording(1, 1, 1, 1)
        pylink.pumpDelay(100)  # Let tracker stabilize
        tracker.sendMessage(f"SYNCTIME global_sec={globalClock.getTime():.5f}")

        # Flush any keys, scroll deltas, or held buttons from the previous trial / EyeLink delay
        kb.clearEvents()
        key_state.clear()
        mouse.getWheelRel()
        left_was_down, _, right_was_down = mouse.getPressed()

        #-------------------------------------------------------------------------------
        # INTERACTIVE WHILE LOOP
        #-------------------------------------------------------------------------------
    
        while timer.getTime() < experiment_time:
            frame_events = []
            scroll_reversal_on_frame = False
            trial_ended_this_frame = False

            keys = kb.getKeys(keyList=WATCHED_KEYS, waitRelease=False, clear=True)
            key_names = [k.name for k in keys]
            shift_held = bool(key_state[pyglet_key.LSHIFT] or key_state[pyglet_key.RSHIFT])
            if 'escape' in key_names:
                exit_reason = 'escape'
                user_escaped_session = True
                frame_events.append('ESCAPE_PRESSED')
                trial_ended_this_frame = True

            if not trial_ended_this_frame:
                if 'r' in key_names:
                    apply_preset(None)
                    state['zoom'] = 1.0
                    state['pan'] = [0, 0]
                    apply_window()
                    reset_count += 1
                    frame_events.append('VIEW_RESET')

                if 't' in key_names:
                    state['show_truth'] = not state['show_truth']
                    truth_overlay_toggled = True
                    frame_events.append(f"TRUTH_TOGGLE:{'ON' if state['show_truth'] else 'OFF'}")
                    print(f"Ground-truth overlay: {'ON' if state['show_truth'] else 'OFF'}")

                for kn in key_names:
                    if kn in key_to_preset:
                        apply_preset(key_to_preset[kn])
                        wl_preset_changes_count += 1
                        frame_events.append(f"PRESET_KEY:{state['preset']}")
                    if kn in PLANE_KEYS:
                        new_plane = PLANE_KEYS[kn]
                        if new_plane != state['plane']:
                            plane_switches_count += 1
                        if state['current_guess'] is not None:
                            state['cursor'] = list(state['current_guess'])
                        state['plane'] = new_plane
                        apply_window()
                        frame_events.append(f"PLANE_KEY:{new_plane}")

                wheel_x, wheel_y = mouse.getWheelRel()
                if wheel_y != 0:
                    ax = AXIS[state['plane']]
                    old_pos = state['cursor'][ax]
                    new_pos = int(np.clip(old_pos + int(wheel_y), 0, plane_count() - 1))
                    step_delta = new_pos - old_pos
                    if step_delta != 0:
                        state['cursor'][ax] = new_pos
                        apply_window()
                        scroll_events_count += 1
                        total_slices_traversed += abs(step_delta)

                        # Check for scroll direction reversal (oscillation) in current plane
                        curr_dir = 1 if step_delta > 0 else -1
                        pl = state['plane']
                        if last_scroll_dir[pl] != 0 and curr_dir != last_scroll_dir[pl]:
                            scroll_reversals_by_plane[pl] += 1
                            scroll_reversal_on_frame = True
                            vis_now = get_visible_nodules_on_slice(*state['cursor'], pl)
                            if vis_now:
                                scroll_reversals_on_nodule_slices += 1
                            frame_events.append(f"SCROLL_REVERSAL:{pl}")
                        last_scroll_dir[pl] = curr_dir
            else:
                wheel_x, wheel_y = 0.0, 0.0

            left_pressed, middle_pressed, right_pressed = mouse.getPressed()
            both_pressed = left_pressed and right_pressed
            
            # Track absolute mouse movement from click anchor
            x_m, y_m = mouse.getPos()
            m_px_x, m_px_y = convertToPix(np.array([0, 0]), np.array([x_m, y_m]), win.units, win)

            if not trial_ended_this_frame:
                if both_pressed:
                    zoom_drag_active = False
                    wl_drag_active = False
                    # Pan via absolute delta from previous frame
                    dx_p = m_px_x - prev_m_px_x
                    dy_p = m_px_y - prev_m_px_y
                    if dx_p != 0 or dy_p != 0:
                        state['pan'][0] += dx_p
                        state['pan'][1] += dy_p
                        apply_window()
                else:
                    # LEFT CLICK / ZOOM DRAG
                    if left_pressed and not left_was_down:
                        hit = button_at(mouse)
                        if hit is not None:
                            zoom_drag_active = False
                            if hit['type'] == 'preset':
                                apply_preset(hit['name'])
                                wl_preset_changes_count += 1
                                frame_events.append(f"PRESET_BTN:{state['preset']}")
                            elif hit['name'] == 'confirm_guess':
                                if state['current_guess'] is not None:
                                    exit_reason = 'confirm_guess'
                                    state['trial_result'] = evaluate_trial_decision('confirm_guess', timer.getTime())
                                    frame_events.append('SUBMIT:confirm_guess')
                                    trial_ended_this_frame = True
                                else:
                                    frame_events.append('CONFIRM_REJECTED_NO_GUESS')
                            elif hit['name'] == 'no_nodule':
                                exit_reason = 'no_nodule'
                                state['trial_result'] = evaluate_trial_decision('no_nodule', timer.getTime())
                                frame_events.append('SUBMIT:no_nodule')
                                trial_ended_this_frame = True
                        elif shift_held:
                            vox, mm = screen_to_voxel(mouse)
                            if vox is not None:
                                state['current_guess'] = [int(vox[0]), int(vox[1]), int(vox[2])]
                                state['cursor'] = [int(vox[0]), int(vox[1]), int(vox[2])]
                                final_guess_mm = [round(float(mm[0]), 2), round(float(mm[1]), 2), round(float(mm[2]), 2)]
                                num_guess_updates += 1
                                t_now = timer.getTime()
                                if time_to_first_guess_sec is None:
                                    time_to_first_guess_sec = round(t_now, 4)
                                time_of_final_guess_sec = round(t_now, 4)
                                update_text()
                                frame_events.append(f"GUESS_PLACED:{state['current_guess'][0]}_{state['current_guess'][1]}_{state['current_guess'][2]}")
                            zoom_drag_active = False
                        else:
                            zoom_drag_active = True
                            zoom_anchor_y = m_px_y
                            zoom_start_val = state['zoom']
                    elif not left_pressed:
                        zoom_drag_active = False

                    if zoom_drag_active:
                        # Dragging UP increases zoom, DOWN decreases zoom smoothly
                        drag_delta = m_px_y - zoom_anchor_y
                        state['zoom'] = max(0.1, min(zoom_start_val + (drag_delta * 0.005 * zoom_sensitivity), 10.0))
                        apply_window()

                    # RIGHT CLICK / WINDOW-LEVEL DRAG
                    if right_pressed and not right_was_down:
                        hit = button_at(mouse)
                        if hit is not None:
                            wl_drag_active = False
                        else:
                            wl_drag_active = True
                            wl_anchor_x = m_px_x
                            wl_anchor_y = m_px_y
                            wl_start_w = state['width']
                            wl_start_c = state['center']
                    elif not right_pressed:
                        wl_drag_active = False

                    if wl_drag_active:
                        dx_wl = m_px_x - wl_anchor_x
                        dy_wl = m_px_y - wl_anchor_y
                        # Uses your drag_sensitivity variable (0.5 * HU range)
                        state['width'] = max(1.0, wl_start_w + dx_wl * (drag_sensitivity * 0.002))
                        state['center'] = wl_start_c + dy_wl * (drag_sensitivity * 0.002)
                        state['preset'] = 'Custom'
                        apply_window()

            prev_m_px_x, prev_m_px_y = m_px_x, m_px_y
            left_was_down = left_pressed
            right_was_down = right_pressed


            # Render frame
            dicom_display.draw()
            draw_overlays()
            displayInfo.draw()
            draw_buttons()
            win.flip()
            
            # Push this frame's events to EyeLink EDF
            if frame_events:
                tracker.sendMessage(f"FRAME_{frame_idx} " + "|".join(frame_events))
                
            gaze_x, gaze_y = None, None
            sample = tracker.getNewestSample()
            if sample is not None:
                if sample.isRightSample():
                    gaze_x, gaze_y = sample.getRightEye().getGaze()
                elif sample.isLeftSample():
                    gaze_x, gaze_y = sample.getLeftEye().getGaze()

            #---------------------------------------------------------------------------
            # PER-FRAME IN-MEMORY TELEMETRY & ACCUMULATOR UPDATE (ZERO DISK I/O)
            #---------------------------------------------------------------------------
            now_global = globalClock.getTime()
            now_trial = timer.getTime()
            dt = max(0.0, now_trial - prev_trial_t)
            prev_trial_t = now_trial

            cx_vox, cy_vox, cz_vox = state['cursor']
            curr_plane = state['plane']
            active_ax = AXIS[curr_plane]
            active_slice_0 = state['cursor'][active_ax]

            # Update visited slices and plane dwell timers
            visited_slices[curr_plane].add(active_slice_0)
            time_by_plane[curr_plane] += dt
            max_zoom_reached = max(max_zoom_reached, state['zoom'])
            min_zoom_reached = min(min_zoom_reached, state['zoom'])
            if state['show_truth']:
                truth_overlay_time_sec += dt

            # Drag mode classification
            if both_pressed:
                drag_mode = 'pan_drag'
                pan_drag_frames_count += 1
            elif zoom_drag_active:
                drag_mode = 'zoom_drag'
                zoom_drag_frames_count += 1
            elif wl_drag_active:
                drag_mode = 'wl_drag'
                wl_drag_frames_count += 1
            else:
                drag_mode = 'none'

            # Ground-truth visibility on current slice
            vis_nods = get_visible_nodules_on_slice(cx_vox, cy_vox, cz_vox, curr_plane)
            nodule_on_slice = len(vis_nods) > 0
            if nodule_on_slice:
                if not nodule_slice_ever_visited:
                    nodule_slice_ever_visited = True
                    time_to_first_nodule_slice_sec = round(now_trial, 4)
                dwell_time_on_nodule_slices_sec += dt

            # Mouse 2D pixel & 3D voxel/mm/HU hover telemetry
            x_m, y_m = mouse.getPos()
            m_px_x, m_px_y = convertToPix(np.array([0, 0]), np.array([x_m, y_m]), win.units, win)
            hover_vox, hover_mm = screen_to_voxel(mouse)
            if hover_vox is not None:
                h_vx, h_vy, h_vz = int(hover_vox[0]), int(hover_vox[1]), int(hover_vox[2])
                h_mm_x, h_mm_y, h_mm_z = round(float(hover_mm[0]), 2), round(float(hover_mm[1]), 2), round(float(hover_mm[2]), 2)
                h_hu = round(float(vol[h_vz, h_vy, h_vx]), 1)
                hovered_nods = [
                    nod['id'] for nod in state['nodules']
                    if is_voxel_in_nodule(h_vx, h_vy, h_vz, curr_plane, nod, tol=INPLANE_XY_TOL)
                ]
                mouse_over_nod = len(hovered_nods) > 0
                if mouse_over_nod:
                    if not nodule_mouse_ever_hovered:
                        nodule_mouse_ever_hovered = True
                        time_to_first_nodule_hover_sec = round(now_trial, 4)
                    dwell_time_nodule_hover_sec += dt
            else:
                h_vx = h_vy = h_vz = None
                h_mm_x = h_mm_y = h_mm_z = None
                h_hu = None
                hovered_nods = []
                mouse_over_nod = False

            g_active = state['current_guess'] is not None
            g_vx, g_vy, g_vz = state['current_guess'] if g_active else (None, None, None)

            curr_state_signature = (
                curr_plane, cx_vox, cy_vox, cz_vox,
                round(state['zoom'], 4), round(state['pan'][0], 1), round(state['pan'][1], 1),
                round(state['width'], 1), round(state['center'], 1), state['preset'], state['show_truth'],
                round(float(m_px_x), 1), round(float(m_px_y), 1),
                int(left_pressed), int(middle_pressed), int(right_pressed), int(shift_held),
                int(wheel_y), drag_mode, g_vx, g_vy, g_vz, bool(frame_events)
            )

            if LOG_EVERY_FRAME or (curr_state_signature != prev_logged_state) or bool(frame_events):
                active_telemetry_buffer.append([
                    expInfo['participant'], expInfo['session'], trial_idx, state['patient_id'], frame_idx,
                    round(now_global, 5), round(now_trial, 5), round(dt, 5),
                    curr_plane, cx_vox, cy_vox, cz_vox,
                    active_slice_0, active_slice_0 + 1,
                    nodule_on_slice, "|".join(vis_nods),
                    round(state['zoom'], 4), round(state['pan'][0], 2), round(state['pan'][1], 2),
                    round(float(dicom_display.pos[0]), 2), round(float(dicom_display.pos[1]), 2),
                    round(float(dicom_display.size[0]), 2), round(float(dicom_display.size[1]), 2),
                    round(state['width'], 2), round(state['center'], 2), state['preset'], state['show_truth'],
                    round(float(m_px_x), 2), round(float(m_px_y), 2), hover_vox is not None,
                    h_vx, h_vy, h_vz,
                    h_mm_x, h_mm_y, h_mm_z, h_hu,
                    mouse_over_nod, "|".join(hovered_nods),
                    int(left_pressed), int(middle_pressed), int(right_pressed), int(shift_held),
                    wheel_y, scroll_reversal_on_frame, drag_mode,
                    g_active, g_vx, g_vy, g_vz, gaze_x, gaze_y,
                    "|".join(frame_events)
                ])
                prev_logged_state = curr_state_signature

            frame_idx += 1

            if trial_ended_this_frame:
                break

        #-------------------------------------------------------------------------------
        # POST-TRIAL CSV FLUSH (SUMMARY ROW + TELEMETRY BUFFER)
        #-------------------------------------------------------------------------------
        trial_end_global_sec = globalClock.getTime()
        if state['trial_result'] is None:
            state['trial_result'] = evaluate_trial_decision('none', timer.getTime())
            
        # Stop EyeLink recording for this trial
        tracker.stopRecording()
        tracker.sendMessage("TRIAL_RESULT 0")

        # Serialize ground-truth consensus nodules for the trial summary CSV
        gt_nodules_summary = []
        for nod in state['nodules']:
            bb = nod['bbox']
            cx = 0.5 * (bb['x_min'] + bb['x_max'])
            cy = 0.5 * (bb['y_min'] + bb['y_max'])
            cz = 0.5 * (bb['z_min'] + bb['z_max'])
            mm_c = origin + np.array([cx * spacing['x'], cy * spacing['y'], cz * spacing['z']])
            gt_nodules_summary.append({
                'id': nod['id'],
                'reader': nod['reader'],
                'bbox_vox': bb,
                'center_vox': [round(cx, 1), round(cy, 1), round(cz, 1)],
                'center_mm': [round(float(mm_c[0]), 2), round(float(mm_c[1]), 2), round(float(mm_c[2]), 2)],
            })

        ax_min_vis = min(visited_slices['axial']) if visited_slices['axial'] else 0
        ax_max_vis = max(visited_slices['axial']) if visited_slices['axial'] else 0
        ax_span_pct = ((ax_max_vis - ax_min_vis + 1) / float(Z)) * 100.0 if Z > 0 else 0.0

        ax_cov = (len(visited_slices['axial']) / float(Z)) * 100.0 if Z > 0 else 0.0
        cor_cov = (len(visited_slices['coronal']) / float(Y)) * 100.0 if Y > 0 else 0.0
        sag_cov = (len(visited_slices['sagittal']) / float(X)) * 100.0 if X > 0 else 0.0

        tr_res = state['trial_result']
        g_vox_final = state['current_guess'] if state['current_guess'] is not None else [None, None, None]
        g_mm_final = final_guess_mm if final_guess_mm is not None else [None, None, None]

        trial_row = {
            'participant_id': expInfo['participant'],
            'session': expInfo['session'],
            'exp_date': expInfo['date'],
            'psychopy_version': psychopyVersion,
            'pilot_mode': PILOTING,
            'win_width_px': int(win.size[0]),
            'win_height_px': int(win.size[1]),
            'measured_fps': round(float(expInfo['frameRate']), 2) if expInfo.get('frameRate') else None,
            'trial_idx': trial_idx,
            'trial_start_global_sec': round(trial_start_global_sec, 4),
            'trial_end_global_sec': round(trial_end_global_sec, 4),
            'patient_id': state['patient_id'],
            'series_uid': str(ds0.SeriesInstanceUID).strip(),
            'vol_dim_x': X,
            'vol_dim_y': Y,
            'vol_dim_z': Z,
            'spacing_x_mm': round(spacing['x'], 4),
            'spacing_y_mm': round(spacing['y'], 4),
            'spacing_z_mm': round(spacing['z'], 4),
            'origin_x_mm': round(float(origin[0]), 2),
            'origin_y_mm': round(float(origin[1]), 2),
            'origin_z_mm': round(float(origin[2]), 2),
            'photometric_invert': state['invert'],
            'default_wl_width': round(state['default_width'], 2),
            'default_wl_center': round(state['default_center'], 2),
            'xml_file_found': len(xml_files) > 0,
            'gt_has_nodules': len(state['nodules']) > 0,
            'gt_num_consensus_nodules': len(state['nodules']),
            'gt_total_raw_reader_anns': state['raw_reader_anns_count'],
            'gt_nodule_z_slices_0based': ";".join(str(z) for z in nodule_z_slices),
            'gt_nodules_json': json.dumps(gt_nodules_summary),
            'exit_reason': exit_reason,
            'decision': tr_res['decision'],
            'rt_sec': tr_res['rt_sec'],
            'plane_submitted': tr_res['plane_submitted'],
            'cursor_at_submit_x': state['cursor'][0],
            'cursor_at_submit_y': state['cursor'][1],
            'cursor_at_submit_z': state['cursor'][2],
            'guess_placed': state['current_guess'] is not None,
            'num_guess_updates': num_guess_updates,
            'time_to_first_guess_sec': time_to_first_guess_sec,
            'time_of_final_guess_sec': time_of_final_guess_sec,
            'guess_vx': g_vox_final[0],
            'guess_vy': g_vox_final[1],
            'guess_vz': g_vox_final[2],
            'guess_mm_x': g_mm_final[0],
            'guess_mm_y': g_mm_final[1],
            'guess_mm_z': g_mm_final[2],
            'inplane_xy_tol_vox': INPLANE_XY_TOL,
            'hit': tr_res['hit'],
            'outcome': tr_res['outcome'],
            'matched_nodule_ids': "|".join(tr_res['matched_nodule_ids']),
            'closest_nodule_id': tr_res['closest_nodule_id'],
            'closest_dist_mm': tr_res['closest_dist_mm'],
            'closest_inplane_dist_mm': tr_res['closest_inplane_dist_mm'],
            'closest_z_slice_error': tr_res['closest_z_slice_error'],
            'total_frames': frame_idx,
            'time_axial_sec': round(time_by_plane['axial'], 4),
            'time_coronal_sec': round(time_by_plane['coronal'], 4),
            'time_sagittal_sec': round(time_by_plane['sagittal'], 4),
            'plane_switches_count': plane_switches_count,
            'scroll_events_count': scroll_events_count,
            'total_slices_traversed': total_slices_traversed,
            'scroll_reversals_total': sum(scroll_reversals_by_plane.values()),
            'scroll_reversals_axial': scroll_reversals_by_plane['axial'],
            'scroll_reversals_coronal': scroll_reversals_by_plane['coronal'],
            'scroll_reversals_sagittal': scroll_reversals_by_plane['sagittal'],
            'scroll_reversals_on_nodule_slices': scroll_reversals_on_nodule_slices,
            'unique_axial_slices_visited': len(visited_slices['axial']),
            'axial_coverage_pct': round(ax_cov, 2),
            'axial_min_slice_visited': ax_min_vis,
            'axial_max_slice_visited': ax_max_vis,
            'axial_span_pct': round(ax_span_pct, 2),
            'full_sweep_axial': (ax_cov / 100.0) >= FULL_SWEEP_THRESHOLD,
            'unique_coronal_slices_visited': len(visited_slices['coronal']),
            'coronal_coverage_pct': round(cor_cov, 2),
            'full_sweep_coronal': (cor_cov / 100.0) >= FULL_SWEEP_THRESHOLD,
            'unique_sagittal_slices_visited': len(visited_slices['sagittal']),
            'sagittal_coverage_pct': round(sag_cov, 2),
            'full_sweep_sagittal': (sag_cov / 100.0) >= FULL_SWEEP_THRESHOLD,
            'nodule_slice_ever_visited': nodule_slice_ever_visited,
            'time_to_first_nodule_slice_sec': time_to_first_nodule_slice_sec,
            'dwell_time_on_nodule_slices_sec': round(dwell_time_on_nodule_slices_sec, 4),
            'nodule_mouse_ever_hovered': nodule_mouse_ever_hovered,
            'time_to_first_nodule_hover_sec': time_to_first_nodule_hover_sec,
            'dwell_time_nodule_hover_sec': round(dwell_time_nodule_hover_sec, 4),
            'wl_preset_changes_count': wl_preset_changes_count,
            'wl_drag_frames_count': wl_drag_frames_count,
            'zoom_drag_frames_count': zoom_drag_frames_count,
            'max_zoom_reached': round(max_zoom_reached, 4),
            'min_zoom_reached': round(min_zoom_reached, 4),
            'pan_drag_frames_count': pan_drag_frames_count,
            'reset_count': reset_count,
            'truth_overlay_toggled': truth_overlay_toggled,
            'truth_overlay_time_sec': round(truth_overlay_time_sec, 4),
        }

        with open(trials_csv_path, mode='a', newline='', encoding='utf-8') as f_tr:
            csv.DictWriter(f_tr, fieldnames=TRIALS_FIELDNAMES).writerow(trial_row)

        flush_telemetry_buffer()
        print(f"Saved Trial {trial_idx} summary -> {trials_csv_path}")
        print(f"Flushed Trial {trial_idx} telemetry ({frame_idx} frames) -> {telemetry_csv_path}")

    endExperiment(thisExp, win=win)

def saveData(thisExp):
    """
    Custom CSV files (*_trials.csv and *_telemetry.csv) are already flushed at the end of each trial.
    """
    pass


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
