::this a script for running the gprMax open source tool on an input file with the
::result of the Bscan output file.

set CONDAPATH=%1
set IN_FULLPATH=%2
set NUM_ASCANS=%3

call %CONDAPATH%Scripts\activate.bat
call conda activate gprMax

:: this is the actual relevent gprMax actions:
:: 1 - create the geometry .vti file of the input file scene.
python geometry_onoff.py %IN_FULLPATH% on
python -m gprMax %IN_FULLPATH%.in --geometry-only
python geometry_onoff.py %IN_FULLPATH% off
:: 2 - running the #NUM_ASCANS Ascans.
python -m gprMax %IN_FULLPATH%.in -n %NUM_ASCANS%
:: 3 - merging the Ascans to Bscan
python -m tools.outputfiles_merge %IN_FULLPATH% --remove-files

::python -m tools.plot_Bscan %IN_PATH%%IN_NAME%_merged.out Ez
pause
