# HFAN-Psychopy-Experimental-Software
Psychopy based experimental software and data analysis scripts for HFAN, allowing for the tracking of eye movements and logging data during radiologist CT nodule detection trials.


# PERSONAL NOTES
This READ.ME is still a work in progress but it will become more fully fleshed out as the program comes to life. feel free to change anything about the description or how the repository will be properly displayed(so everything before "PERSONAL NOTES")

In order to run the file, you must download it and run it specifically within "Coder View".

I switched from Builder View to Coder View as it proved to be too difficult to properly execute my code as the code grew more complex.

If there are any questions, just send me a message on Discord(server name is andy)


Now I will begin to keep a checklist of things that need to be fully implemented:

  1) Displaying the dicom images(COMPLETED)
  2) Allowing the user to scroll through the images with the scroll wheel(COMPLETED)
  3) Allowing the user to change the window and level of each DICOM image(WORK IN PROGRESS)
  4) Allow the user to change the anatomical plane of the DICOM file(WORK IN PROGRESS)
  5) Allow the user to zoom in/out of the image
  6) Allow the user to guess where they think the nodule is located
  7) Sync all of the data from all previous steps with the eye tracker
  8) Feed this back to HFAN in a presentable way so they can do their research work

Here are some suggested controls to keep in mind(subject to change):
  1) scroll wheel should be for scrolling(accessing each individual slice of the dicom file)
  2) Holding right-click and dragging your mouse in a direction should change the level/window of the DICOM image
  3) Having some pre-built modes for window and level, that easily allow the user to see the different anatomical parts more clearly
  4) An idea for zooming in/out was holding left click and dragging your mouse, but we could try to look for a better solution
  5) Having some sort of visual indication to show the user which slice they are currently on
  6) Allowing the user to left-click and submit their answer.

If there is anything else important I've missed, please feel free to add on.

# Currently working on implementing
  Allowing the user to change the window and level of each DICOM image
    - I have to research the logic behind this, but I already know how we can change the shading of color for the DICOM images, so it shouldn't be hard to implement
  Allow the user to change the anatomical plane of the DICOM file
    - I've looked into this, and it's not hard. It just requires the use of some functions within NumPy since it just involves manipulating arrays.
    The goal is to complete both of these plans by 11/04/26
