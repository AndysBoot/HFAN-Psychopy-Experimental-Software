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

# --- Import packages ---
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
    folder = r'C:\Users\Public\Downloads\lidc_idri\LIDC-IDRI-0001'
    

    
    # allows us to gather all of the dicom files within the folder
    files = glob.glob(os.path.join(folder, '**', '*.dcm'), recursive=True)

    def get_slice():
        x, y, z = state['cursor']
        if state['plane'] == 'axial':    return vol[z, :, :], True    # flip: row 0 = anterior -> top
        if state['plane'] == 'coronal':  return vol[:, y, :], False   # row 0 = inferior -> bottom
        return vol[:, :, x], False                                    # sagittal

    def plane_count():
        return {'axial': Z, 'coronal': Y, 'sagittal': X}[state['plane']]
        
        
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
    # a default state for the file that is shared across all functions
    state = {
    'plane' : 'axial',
    'center': 0.0,
    'width': 1.0,
    'invert': False,
    'ds': None,
    'default_center': 0.0,
    'default_width': 1.0,
    'preset' : 'File default',
    }
    def to_float(v, default):
        """Return a float from a number, a list of numbers, or fall back to default."""
        if v is None:
            return float(default)
        try:
            if hasattr(v, '__iter__') and not isinstance(v, str):
                v = list(v)[0]
            return float(v)
        except (TypeError, ValueError, IndexError):
            return float(default)
    
        
    series = defaultdict(list)
    for fp in files:
        h = pydicom.dcmread(fp, stop_before_pixels=True)
        series[h.SeriesInstanceUID].append(fp)
    series_files = max(series.values(), key=len)   # or choose per trial
    
    def build_volume(paths):
        dss = [pydicom.dcmread(p) for p in paths]
        dss.sort(key=lambda d: float(d.ImagePositionPatient[2]))
        vol = np.stack([d.pixel_array.astype(np.float32) * float(d.get('RescaleSlope', 1))
                        + float(d.get('RescaleIntercept', 0)) for d in dss])
        zs = np.array([float(d.ImagePositionPatient[2]) for d in dss])
        z_sp = float(np.median(np.abs(np.diff(zs)))) if len(zs) > 1 else float(dss[0].get('SliceThickness', 1))
        row_sp, col_sp = [float(v) for v in dss[0].PixelSpacing]
        origin = np.array(dss[0].ImagePositionPatient, dtype=float)
        return vol, dict(x=col_sp, y=row_sp, z=z_sp), origin, dss[0]

    vol, spacing, origin, ds0 = build_volume(series_files)
    Z, Y, X = vol.shape
    state['ds'] = ds0
    state['cursor'] = [X // 2, Y // 2, Z // 2]
    state['invert'] = ds0.get('PhotometricInterpretation', '') == 'MONOCHROME1'
    state['default_center'] = to_float(ds0.get('WindowCenter'), (vol.min() + vol.max()) / 2)
    state['default_width'] = max(1.0, to_float(ds0.get('WindowWidth'), vol.max() - vol.min()))
    state['center'], state['width'] = state['default_center'], state['default_width']

    AXIS = {'axial': 2, 'coronal': 1, 'sagittal': 0}
    # get_slice(), plane_count() as before
    
    print(f"found a total of {len(series_files)} files")

    
    def fit_display(shape):
        rows, cols = shape
        sy, sx = {'axial': (spacing['y'], spacing['x']),
                  'coronal': (spacing['z'], spacing['x']),
                  'sagittal': (spacing['z'], spacing['y'])}[state['plane']]
        h_mm, w_mm = rows * sy, cols * sx
        scale = min(win.size[0] * 0.8 / w_mm, win.size[1] * 0.9 / h_mm)
        dicom_display.units = 'pix'
        dicom_display.size = (w_mm * scale, h_mm * scale)
        dicom_display.pos = (0, 0)
        
    def apply_window():
        img, flip = get_slice()
        low, high = state['center'] - state['width'] / 2, state['center'] + state['width'] / 2
        img = (np.clip(img, low, high) - low) / (high - low)
        if state['invert']:
            img = 1.0 - img
        img = img * 2 - 1
        dicom_display.image = np.flipud(img) if flip else img
        fit_display(img.shape)
        update_text()

    def update_text():
        ax = AXIS[state['plane']]
        displayInfo.units = 'pix'
        displayInfo.height = 14
        displayInfo.color = [-0.2, -0.2, -0.2]          # dim
        displayInfo.pos = (-win.size[0] / 2 + 70, win.size[1] / 2 - 25)
        displayInfo.text = f"{state['plane'][0].upper()} {state['cursor'][ax] + 1}/{plane_count()}"

    def screen_to_voxel(mouse):
        x, y = mouse.getPos()
        px, py = convertToPix(np.array([0, 0]), np.array([x, y]), win.units, win)
        w, h = dicom_display.size
        cx, cy = dicom_display.pos
        u, v = (px - cx) / w + 0.5, (py - cy) / h + 0.5
        if not (0 <= u < 1 and 0 <= v < 1):
            return None, None
        x, y, z = state['cursor']
        plane = state['plane']
        if plane == 'axial':
            x, y = u * X, (1 - v) * Y
        elif plane == 'coronal':
            x, z = u * X, v * Z
        else:
            y, z = u * Y, v * Z
        vox = np.clip([int(x), int(y), int(z)], 0, [X - 1, Y - 1, Z - 1])
        mm = origin + vox * np.array([spacing['x'], spacing['y'], spacing['z']])
        return vox, mm
    
        
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

# ---- on-screen buttons (drawn in pixel units, right edge of window) ----
    BTN_W, BTN_H, BTN_GAP = 160, 34, 8
    btn_x = win.size[0] / 2 - BTN_W / 2 - 15
    btn_y0 = win.size[1] / 2 - 40

    entries = [(f"{i + 1}  {n}", n) for i, (n, w, l) in enumerate(PRESETS)]
    entries.append(("0  File default", None))

    buttons = []
    for i, (label, name) in enumerate(entries):
        pos = (btn_x, btn_y0 - i * (BTN_H + BTN_GAP))
        rect = visual.Rect(win, width=BTN_W, height=BTN_H, pos=pos, units='pix',
                           fillColor=[-0.6, -0.6, -0.6], lineColor='white', lineWidth=1)
        txt = visual.TextStim(win, text=label, pos=pos, height=15,
                              units='pix', color='white')
        buttons.append({'name': name, 'pos': pos, 'rect': rect, 'txt': txt})

    # keyboard: 1-8 = presets, 0 = file default (top row or numpad)
    key_to_preset = {}
    for i, (n, w, l) in enumerate(PRESETS):
        key_to_preset[str(i + 1)] = n
        key_to_preset[f'num_{i + 1}'] = n
    key_to_preset['0'] = None
    key_to_preset['num_0'] = None

    def button_at(mouse):
        """Return the button dict under the mouse, or None."""
        x, y = mouse.getPos()  # in win.units
        px, py = convertToPix(np.array([0, 0]), np.array([x, y]), win.units, win)
        for b in buttons:
            bx, by = b['pos']
            if abs(px - bx) <= BTN_W / 2 and abs(py - by) <= BTN_H / 2:
                return b
        return None

    def draw_buttons():
        for b in buttons:
            active = (b['name'] == state['preset']) or \
                     (b['name'] is None and state['preset'] == 'File default')
            b['rect'].fillColor = [-0.1, 0.2, 0.6] if active else [-0.6, -0.6, -0.6]
            b['rect'].draw()
            b['txt'].draw()
    
    kb = keyboard.Keyboard()
    mouse = Mouse(visible = True, win = win)
    
    PLANE_KEYS = {'a': 'axial', 'c': 'coronal', 's': 'sagittal'}   # add this

    experiment_time = 300
    
    apply_window()
    
    mouse.getWheelRel()
    mouse.getRel()
    timer = core.Clock()
    
    drag_sensitivity = 0.5 * (vol.max() - vol.min())
    mouse_was_down = False
    drag_active = False
    
    #this is the logic behind the scroll wheel functionality. As the project develops, this will become a function and it'll allow the user to
    # easily scroll whenever they want, rather then it being for the duration of experiment_time
    while timer.getTime() < experiment_time:
        key_names = [k.name for k in kb.getKeys()]
        if 'escape' in key_names:
            break
        if 'r' in key_names:
            apply_preset(None)

        for kn in key_names:
            if kn in key_to_preset:
                apply_preset(key_to_preset[kn])
            if kn in PLANE_KEYS:                       # NEW: a / c / s
                state['plane'] = PLANE_KEYS[kn]
                apply_window()

        # CHANGED: wheel moves the cursor along the current plane's axis
        wheel_x, wheel_y = mouse.getWheelRel()
        if wheel_y != 0:
            ax = AXIS[state['plane']]
            new_pos = int(np.clip(state['cursor'][ax] + int(wheel_y), 0, plane_count() - 1))
            if new_pos != state['cursor'][ax]:
                state['cursor'][ax] = new_pos
                apply_window()

        dx, dy = mouse.getRel()
        pressed = mouse.getPressed()[2]

        if pressed and not mouse_was_down:          # a new click just started
            hit = button_at(mouse)
            if hit is not None:
                apply_preset(hit['name'])           # button click
                drag_active = False                 # don't start a drag from a button
            else:
                drag_active = True                  # click on the image -> window/level drag
        elif not pressed:
            drag_active = False
        mouse_was_down = pressed

        if drag_active and (dx != 0 or dy != 0):
            state['width'] = max(1.0, state['width'] + dx * drag_sensitivity)
            state['center'] += dy * drag_sensitivity
            state['preset'] = 'Custom'
            apply_window()

        dicom_display.draw()
        displayInfo.draw()
        draw_buttons()
        win.flip()
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
